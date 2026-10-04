"""Shared loader for the validation adapters: rows the paper's own scripts already read, put into
the harness's gaps format so the harness can be checked against the published verdicts."""
import os

import pandas as pd

ROOT = r"C:\Research\PsychBench\UpdatedRun"
EXT = os.path.join(ROOT, "analysis", "brm", "l2_external_r3.csv")
TWIN = os.path.join(ROOT, "paper_brm", "external", "results", "twin2k", "level2_contrasts.csv")


def external(source):
    d = pd.read_csv(EXT)
    d = d[(d.config == "headline") & (d.source == source)]
    g = pd.DataFrame(dict(family=d.family, contrast=d.contrast, row=d.row, g=d.g, se_g=d.se_g, gamma=d.p,
                          se_gamma=d.se_p, df_g=d.df, df_gamma=d.df, cov=d["cov"], family_size=d.family_size,
                          stored_verdict=d.verdict))
    return g.reset_index(drop=True), [EXT]


def twin(config):
    d = pd.read_csv(TWIN)
    d = d[d.config == config]
    g = pd.DataFrame(dict(family=d.run, contrast=d.contrast, outcome=d.outcome, g=d.g, se_g=d.se_g, gamma=d.p,
                          se_gamma=d.se_p, cov=d["cov"], family_size=d.m_family, stored_verdict=d.verdict,
                          stored_label=d.label))
    return g.reset_index(drop=True), [TWIN]
