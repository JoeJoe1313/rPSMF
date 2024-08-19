# -*- coding: utf-8 -*-

"""PSMF Experiment 3 - Shared code for the table generation scripts

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import argparse
import os

METHODS = ["PSMF", "rPSMF", "MLESMF", "TMF"]
PROB_METHODS = ["PSMF", "rPSMF", "MLESMF"]
DATASETS = [
    "beijing_temperature",
]
PERCENTAGES_APP = [20, 40]
PERCENTAGE_MAIN = 30
RANKS = [3, 10]

DATASET_NAMES = {
    "beijing_temperature": "ECG",
}

PREAMBLE = [
    "\\documentclass[11pt, preview=true]{standalone}",
    "\\usepackage{booktabs}",
    "\\usepackage{multirow}",
    "\\usepackage{amsmath}",
    "\\begin{document}",
]
EPILOGUE = ["\\end{document}"]


def make_filepath(method, dataset, perc, rank, result_dir):
    filename = "%s_%i_%s_%s.json" % (dataset, perc, method, rank)
    return os.path.join(result_dir, filename)
