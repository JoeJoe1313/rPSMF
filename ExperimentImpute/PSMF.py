# -*- coding: utf-8 -*-

"""PSMF Experiment 3 - PSMF

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
def ProbabilisticSequentialMatrixFactorizer(
    Y, C, X, d, n, r, M, Mmiss, lam, V, Q, R, P, sig, Iter, YorgInt, Einit
):

    Epred = np.zeros([1, Iter + 1])
    Efull = np.copy(Epred)
    Epred[:, 0] = Einit
    Efull[:, 0] = Einit

    RunTime = np.zeros([1, Iter + 1])

    Yrec = np.zeros([d, n])
    YrecL = np.copy(Yrec)
    YrecH = np.copy(Yrec)

    Id = np.eye(d)

    RunTimeStart = time.time()

    for i in range(0, Iter):
        for t in range(n):

            Mk = np.diag(M[:, t])
            CM = Mk @ C

            Xp = X[:, [n - 1]] if t == 0 else X[:, [t - 1]]
            PP = P + Q

            Yrec[:, [t]] = C @ Xp

            # Assumes R is diagonal. Otherwise: MRM = Mk @ R @ Mk.T
            MRM = np.diag(np.diag(Mk) * np.diag(R))
            Rbar = MRM + Xp.T @ V @ Xp * Id
            CPinv = compute_Sinv(CM, PP, Rbar)
            X[:, [t]] = Xp + PP @ CM.T @ CPinv @ (Y[:, [t]] - CM @ Xp)
            P = PP - PP @ CM.T @ CPinv @ CM @ PP

            eta_k = np.trace(CM @ PP @ CM.T + MRM) / d
            Nt = Xp.T @ V @ Xp + eta_k

            C = C + ((Y[:, [t]] - CM @ Xp) @ Xp.T @ V) / (Nt)
            V = V - (V @ Xp @ Xp.T @ V) / (Nt)

            YrecL[:, [t]] = Yrec[:, [t]] - sig * np.sqrt(Nt)
            YrecH[:, [t]] = Yrec[:, [t]] + sig * np.sqrt(Nt)

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

    # Initialize dimensions, hyperparameters, and noise covariances
    d, n = Yorig.shape
    ranks = [3, 10]
    sig = 2
    Iter = 2
    rho = 10
    v = 2
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

    # V = v * np.eye(r)
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
            V = v * np.eye(r)
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
            # fmt: off
            [ep, ef, rt, ib, res] = ProbabilisticSequentialMatrixFactorizer(
                Y, C, X, d, n, r, M, missMask, rho, V, Q, R, P, sig, Iter, YorigInt, Einit
            )
            # fmt: on

            results_by_rank[r]["errors_predict"].append(ep[:, Iter].item())
            results_by_rank[r]["errors_full"].append(ef[:, Iter].item())
            results_by_rank[r]["runtimes"].append(rt[:, Iter].item())
            results_by_rank[r]["inside_bars"].append(ib)
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
            "v": v,
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
            "PSMF",
        )
        fo = f"ExperimentImpute/output/beijing_temperature_{args.percentage}_PSMF_{r}.json"
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
        plt.savefig(f"psmf_input_{args.percentage}_{r}.pdf")

        fig, axs = plt.subplots(12, 1, figsize=(7, 7))
        for i in range(12):
            axs[i].plot(Ymiss[i, :], color="red", linewidth=2, label="Missing Inputs")
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
                label="Original",
            )
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, -0.4), fontsize="small")
        plt.show(block=False)
        plt.savefig(f"psmf_output_{args.percentage}_{r}.pdf")


if __name__ == "__main__":
    main()
