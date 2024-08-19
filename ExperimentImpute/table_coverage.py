#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PSMF Experiment 3 - Generate coverage table

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import argparse
import json
import os
import sys

import numpy as np
from table_common import (
    DATASET_NAMES,
    DATASETS,
    EPILOGUE,
    PERCENTAGES_APP,
    PREAMBLE,
    PROB_METHODS,
    RANKS,
    make_filepath,
)

FILE_LENGTHS = set()
FILES_MISSING = set()
FILES_NAN = set()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        help="Input directory with result files",
        required=True,
    )
    parser.add_argument(
        "-o", "--output-file", help="Output file for the table", required=True
    )
    parser.add_argument(
        "-p",
        "--percentage",
        help="Missing percentage to ues",
        required=True,
        type=int,
        choices=[20, 30, 40],
    )
    parser.add_argument(
        "-s",
        "--standalone",
        help="Build a standalone latex document",
        action="store_true",
    )
    return parser.parse_args()


def warn_missing(path):
    if path in FILES_MISSING:
        return
    print(f"Warning: result file {path} not found.", file=sys.stderr)
    FILES_MISSING.add(path)


def warn_nan(path, count):
    if path in FILES_NAN:
        return
    print(
        f"Warning: result file {path} contains {count} NaN value(s)",
        file=sys.stderr,
    )
    FILES_NAN.add(path)


def get_mean_coverage(path, method):
    with open(path, "r") as fp:
        data = json.load(fp)
    assert data["method"] == method
    values = data["results"]["inside_sig"]
    FILE_LENGTHS.add(len(values))
    assert len(FILE_LENGTHS) == 1
    if np.any(np.isnan(values)):
        warn_nan(path, np.sum(np.isnan(values)))
        return np.nanmean(values)
    return np.mean(values)


def get_coverage(method, dataset, perc, rank, result_dir):
    path = make_filepath(method, dataset, perc, rank, result_dir)
    if not os.path.exists(path):
        warn_missing(path)
        return "TODO"

    cov = get_mean_coverage(path, method)

    others = {}
    for other_method in PROB_METHODS:
        if other_method == method:
            continue

        other_path = make_filepath(other_method, dataset, perc, rank, result_dir)
        if not os.path.exists(other_path):
            continue

        others[other_method] = get_mean_coverage(other_path, other_method)

    max_cov = max(list(others.values()) + [cov])
    if cov == max_cov:
        return "\\textbf{%.2f}" % cov
    return "%.2f" % cov


def build_table(result_dir, perc, rank):
    tex = []
    tex.append("%% DO NOT EDIT - AUTOMATICALLY GENERATED FROM RESULTS!")
    tex.append("%% This table requires booktabs and multirow!")
    tex.append(f"%% Table for missing percentage {perc}")

    names = {"MLESMF": "MLE-SMF", "PMF": "PMF*", "BPMF": "BPMF*"}
    methname = lambda m: names.get(m, m)
    dataset_name = lambda d: DATASET_NAMES.get(d, d)

    fmt = "l" + "c" * len(DATASETS)
    tex.append("\\begin{tabular}{%s}" % fmt)
    tex.append("\\toprule")
    header = " & ".join([""] + [dataset_name(dset) for dset in DATASETS]) + " \\\\"
    tex.append(header)
    tex.append("\\cmidrule(lr){2-%i}" % (len(DATASETS) + 1))

    for method in PROB_METHODS:
        vrow = [methname(method)]
        for dataset in DATASETS:
            value = get_coverage(method, dataset, perc, rank, result_dir)
            vrow.append(value)
        tex.append(" & ".join(vrow) + " \\\\")

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    return tex


def main():
    args = parse_args()
    for r in RANKS:
        tex = build_table(args.input_dir, args.percentage, r)
        tex = PREAMBLE + tex + EPILOGUE if args.standalone else tex
        fo = f"ExperimentImpute/tables/table_coverage_{args.percentage}_{r}.tex"
        with open(fo, "w") as fp:
            fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
