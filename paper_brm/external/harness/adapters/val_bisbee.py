"""Validation: Bisbee et al. (2024) ChatGPT feeling thermometers, level-2 rows as stored by
external/scripts/bisbee_ladder.py (results/bisbee/level2_contrasts_rr1.csv), one family per prompt."""
import os

import pandas as pd

NAME = "val_bisbee"
KIND = "language model (validation of the harness against bisbee_ladder.py)"
SOURCE = "Bisbee et al. (2024); rows from paper_brm/external/results/bisbee/level2_contrasts_rr1.csv"
FILE = r"C:\Research\PsychBench\UpdatedRun\paper_brm\external\results\bisbee\level2_contrasts_rr1.csv"


def load():
    d = pd.read_csv(FILE)
    g = pd.DataFrame(dict(family=d.framing, contrast=d.contrast, outcome=d.outcome, g=d.g, se_g=d.se_g,
                          gamma=d.p, se_gamma=d.se_p, cov=d["cov"], family_size=d.m_family,
                          stored_verdict=d.verdict))
    return dict(mode="gaps", gaps=g.reset_index(drop=True), inputs=[FILE])
