"""Twin-2K-500 Mega Study read with one Bonferroni family per specification and study.

twin2k_mega.py put every study of a specification in one family (about 2,400 contrasts), which
makes kept unreachable: its own split-half control reads none of 1,598 contrasts. Each study is a
separate survey, so this reading takes the family to be one specification within one study. The
gaps, SEs and covariances are the stored ones in results/twin2k_mega/l2_contrasts.csv, unchanged;
only the family and its size change.
"""
import os

import pandas as pd

NAME = "twin2k_mega_bystudy"
KIND = "digital twins (LLM, 23 specifications) and a split-half control, families per study"
SOURCE = "Twin-2K-500 Mega Study (arXiv 2509.19088); gaps from harness/results/twin2k_mega/l2_contrasts.csv"

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "twin2k_mega", "l2_contrasts.csv")


def load():
    d = pd.read_csv(SRC)
    keep = ["family", "contrast", "study", "outcome", "attribute", "n_a", "n_b",
            "g", "se_g", "df_g", "gamma", "se_gamma", "df_gamma", "cov"]
    g = d[keep].copy()
    g["specification"] = g.family
    g["family"] = g.family + " : " + g.study
    return dict(mode="gaps", gaps=g, inputs=[SRC], notes="families = specification x study")
