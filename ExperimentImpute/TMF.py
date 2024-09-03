# -*- coding: utf-8 -*-

"""PSMF Experiment 3 - TMF

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import datetime as dt
import time

import matplotlib.pyplot as plt
import numpy as np
from common import (
    RMSEM,
    dump_output,
    matrix_hash,
    parse_args,
    prepare_missing,
    prepare_output,
)
from joblib import Memory
from tqdm import trange

MEMORY = Memory("./cache", verbose=0)


@MEMORY.cache
def temporalRegularizedMF(Y, C, X, d, n, r, M, Mmiss, lam, R, Iter, YorgInt, Einit):

    Epred = np.zeros([1, Iter + 1])
    Efull = np.copy(Epred)
    Epred[:, 0] = Einit
    Efull[:, 0] = Einit

    RunTime = np.zeros([1, Iter + 1])

    Yrec = np.zeros([d, n])

    Ir = np.identity(r)

    RunTimeStart = time.time()

    for i in range(0, Iter):

        gam1 = 1e-6
        nu = 2
        gam = gam1 / ((i + 1) ** 0.7)

        for t in range(n):

            MC = np.diag(M[:, t])
            CM = MC @ C

            Xp = X[:, [n - 1]] if t == 0 else X[:, [t - 1]]

            Yrec[:, [t]] = C @ Xp

            CPinv = np.linalg.inv(CM.T @ CM + nu * Ir)
            X[:, [t]] = CPinv @ (nu * Xp + CM.T @ Y[:, [t]])

            C = C + gam * (MC.T @ (Y[:, [t]] - CM @ Xp) @ Xp.T)

        Yrec2 = C @ X
        Epred[:, i + 1] = RMSEM(Yrec, YorgInt, Mmiss)
        Efull[:, i + 1] = RMSEM(Yrec2, YorgInt, Mmiss)

        RunTime[:, i + 1] = time.time() - RunTimeStart

    return Epred, Efull, RunTime, Yrec2


def main():
    args = parse_args()

    # Set the seed if given, otherwise draw one and print it out
    seed = args.seed or np.random.randint(10000)
    print("Using seed: %r" % seed)
    np.random.seed(seed)

    # Load the data
    original_full = np.genfromtxt(
        "/Users/ljoana/repos/rPSMF/ExperimentImpute/data/original.csv", delimiter=","
    )
    Yorig = np.genfromtxt(args.input, delimiter=",")

    # Create a copy with missings set to zero
    YorigInt = np.copy(Yorig)
    YorigInt[np.isnan(YorigInt)] = 0

    _range = trange if args.fancy else range
    log = print if not args.fancy else lambda *a, **kw: None

    # Extract dimensions and set latent dimensionality
    d, n = Yorig.shape
    ranks = [3, 10]
    Iter = 2
    rho = 10
    R = rho * np.eye(d)

    results_by_rank = {
        rank: {
            "errors_predict": [],
            "errors_full": [],
            "runtimes": [],
            "Y_hashes": [],
            "C_hashes": [],
            "X_hashes": [],
            "res": [],
        }
        for rank in ranks
    }

    # Initialize arrays to keep track of quantaties of interest
    # errors_predict = []
    # errors_full = []
    # runtimes = []
    # Y_hashes = []
    # C_hashes = []
    # X_hashes = []

    for i in _range(args.repeats):
        # Create the missing mask (missMask) and its inverse (M)
        Ymiss = np.copy(Yorig)
        missRatio, missMask = prepare_missing(Ymiss, args.percentage / 100)
        M = np.array(np.invert(np.isnan(Ymiss)), dtype=int)

        # In the data we work with, set missing to 0
        Y = np.copy(Ymiss)
        Y[np.isnan(Y)] = 0

        for r in ranks:
            C = np.random.rand(d, r)
            X = np.random.rand(r, n)

            # store hash of matrices; used to ensure they're the same between
            # scripts
            results_by_rank[r]["Y_hashes"].append(matrix_hash(Y))
            results_by_rank[r]["C_hashes"].append(matrix_hash(C))
            results_by_rank[r]["X_hashes"].append(matrix_hash(X))

            YrecInit = C @ X
            Einit = RMSEM(YrecInit, YorigInt, missMask)

            [ep, ef, rt, res] = temporalRegularizedMF(
                Y, C, X, d, n, r, M, missMask, rho, R, Iter, YorigInt, Einit
            )

            results_by_rank[r]["errors_predict"].append(ep[:, Iter].item())
            results_by_rank[r]["errors_full"].append(ef[:, Iter].item())
            results_by_rank[r]["runtimes"].append(rt[:, Iter].item())
            results_by_rank[r]["res"].append(res)

            log(
                "[%s] Rank %02i, Finished step %04i of %04i"
                % (dt.datetime.now().strftime("%c"), r, i + 1, args.repeats)
            )

    for r in ranks:
        params = {
            "r": r,
            "rho": rho,
            "Iter": Iter,
        }
        hashes = {
            "Y": results_by_rank[r]["Y_hashes"],
            "C": results_by_rank[r]["C_hashes"],
            "X": results_by_rank[r]["X_hashes"],
        }
        results = {
            "error_predict": results_by_rank[r]["errors_predict"],
            "error_full": results_by_rank[r]["errors_full"],
            "runtime": results_by_rank[r]["runtimes"],
            "inside_sig": None,
        }
        output = prepare_output(
            args.input,
            __file__,
            params,
            hashes,
            results,
            seed,
            args.percentage,
            missRatio,
            "TMF",
        )
        fo = f"ExperimentImpute/output/beijing_temperature_{args.percentage}_TMF_{r}.json"
        dump_output(output, fo)

        fig, axs = plt.subplots(12, 1, figsize=(7, 7))
        for i in range(12):
            axs[i].plot(Ymiss[i, :], color="red", linewidth=2, label="Missing Inputs")
            axs[i].plot(
                original_full[i, :],
                color="orange",
                linewidth=2,
                alpha=0.5,
                label="Original",
            )
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, -0.5), fontsize="small")
        plt.show(block=False)
        plt.savefig(f"tmf_input_{args.percentage}_{r}.pdf")

        fig, axs = plt.subplots(12, 1, figsize=(7, 10))
        for i in range(12):
            axs[i].plot(Ymiss[i, :], color="red", linewidth=2, label="Original signal")
            axs[i].plot(
                results_by_rank[r]["res"][-1][i, :],
                color="blue",
                linestyle="--",
                linewidth=1,
                label="Reconstruction",
            )
            axs[i].plot(
                original_full[i, :],
                color="orange",
                linewidth=2,
                alpha=0.5,
                label="Missing segments original signal",
            )
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, -0.4), fontsize="small")
        plt.show(block=False)
        plt.savefig(f"tmf_output_{args.percentage}_{r}.pdf")


if __name__ == "__main__":
    main()
