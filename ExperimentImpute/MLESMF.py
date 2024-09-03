# -*- coding: utf-8 -*-

"""PSMF Experiment 3 - MLE-SMF

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import datetime as dt
import time

import matplotlib.pyplot as plt
import numpy as np
from common import (
    RMSEM,
    compute_number_inside_bars,
    dump_output,
    matrix_hash,
    parse_args,
    prepare_missing,
    prepare_output,
)
from joblib import Memory
from tqdm import trange

MEMORY = Memory("./cache", verbose=0)


def compute_Sinv(CM, PP, Rbar):
    # woodbury, using inv, assuming Rbar diagonal
    Ri = np.diag(1 / np.diag(Rbar))
    RiC = Ri @ CM
    Pi = np.linalg.inv(PP)
    PiCRiC = Pi + CM.T @ RiC
    return Ri - RiC @ np.linalg.inv(PiCRiC) @ RiC.T


@MEMORY.cache
def stochasticGradientStateSpaceMF(
    Y, C, X, d, n, r, M, Mmiss, lam, Q, R, P, sig, Iter, YorgInt, Einit
):

    Epred = np.zeros([1, Iter + 1])
    Efull = np.copy(Epred)
    Epred[:, 0] = Einit
    Efull[:, 0] = Einit

    RunTime = np.zeros([1, Iter + 1])

    Yrec = np.zeros([d, n])
    YrecL = np.copy(Yrec)
    YrecH = np.copy(Yrec)

    RunTimeStart = time.time()

    for i in range(0, Iter):

        gam1 = 1e-6
        gam = gam1 / ((i + 1) ** 0.7)

        for t in range(n):

            MC = np.diag(M[:, t])
            CM = MC @ C

            Xp = X[:, [n - 1]] if t == 0 else X[:, [t - 1]]
            PP = P + Q

            Yrec[:, [t]] = C @ Xp

            CPinv = compute_Sinv(CM, PP, R)
            X[:, [t]] = Xp + PP @ CM.T @ CPinv @ (Y[:, [t]] - CM @ Xp)
            P = PP - PP @ CM.T @ CPinv @ CM @ PP

            MRM = np.diag(np.diag(MC) * np.diag(R))
            eta_k = np.trace(CM @ PP @ CM.T + MRM) / d

            C = C + gam * (1 / eta_k) * (MC.T @ (Y[:, [t]] - CM @ Xp) @ Xp.T)

            YrecL[:, [t]] = Yrec[:, [t]] - sig * np.sqrt(eta_k)
            YrecH[:, [t]] = Yrec[:, [t]] + sig * np.sqrt(eta_k)

        Yrec2 = C @ X
        Epred[:, i + 1] = RMSEM(Yrec, YorgInt, Mmiss)
        Efull[:, i + 1] = RMSEM(Yrec2, YorgInt, Mmiss)

        RunTime[:, i + 1] = time.time() - RunTimeStart

    InsideBars = compute_number_inside_bars(Mmiss, d, n, YorgInt, YrecL, YrecH)

    return Epred, Efull, RunTime, InsideBars, Yrec2


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
    sig = 2
    Iter = 2
    rho = 10
    q = 0.1
    p = 1.0

    results_by_rank = {
        rank: {
            "errors_predict": [],
            "errors_full": [],
            "runtimes": [],
            "inside_bars": [],
            "Y_hashes": [],
            "C_hashes": [],
            "X_hashes": [],
            "res": [],
        }
        for rank in ranks
    }

    # Q = q * np.eye(r)
    # R = rho * np.eye(d)
    # P = p * np.eye(r)

    # Initialize arrays to keep track of quantaties of interest
    # errors_predict = []
    # errors_full = []
    # runtimes = []
    # inside_bars = []
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
            Q = q * np.eye(r)
            R = rho * np.eye(d)
            P = p * np.eye(r)

            C = np.random.rand(d, r)
            X = np.random.rand(r, n)

            # store hash of matrices; used to ensure they're the same between
            # scripts
            results_by_rank[r]["Y_hashes"].append(matrix_hash(Y))
            results_by_rank[r]["C_hashes"].append(matrix_hash(C))
            results_by_rank[r]["X_hashes"].append(matrix_hash(X))

            YrecInit = C @ X
            Einit = RMSEM(YrecInit, YorigInt, missMask)

            [ep, ef, rt, ib, res] = stochasticGradientStateSpaceMF(
                Y, C, X, d, n, r, M, missMask, rho, Q, R, P, sig, Iter, YorigInt, Einit
            )

            error_pred = ep[:, Iter].item()
            error_full = ef[:, Iter].item()
            if np.isnan(error_pred) or np.isnan(error_full):
                runtime = float("nan")
                inside_bar = float("nan")
            else:
                runtime = rt[:, Iter].item()
                inside_bar = ib

            results_by_rank[r]["errors_predict"].append(error_pred)
            results_by_rank[r]["errors_full"].append(error_full)
            results_by_rank[r]["runtimes"].append(runtime)
            results_by_rank[r]["inside_bars"].append(inside_bar)
            results_by_rank[r]["res"].append(res)

            log(
                "[%s] Rank %02i, Finished step %04i of %04i"
                % (dt.datetime.now().strftime("%c"), r, i + 1, args.repeats)
            )

    for r in ranks:
        params = {
            "r": r,
            "sig": sig,
            "rho": rho,
            "q": q,
            "p": p,
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
            "inside_sig": results_by_rank[r]["inside_bars"],
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
            "MLESMF",
        )
        fo = f"ExperimentImpute/output/beijing_temperature_{args.percentage}_MLESMF_{r}.json"
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
        plt.savefig(f"mlesmf_input_{args.percentage}_{r}.pdf")

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
        plt.savefig(f"mlesmf_output_{args.percentage}_{r}.pdf")


if __name__ == "__main__":
    main()
