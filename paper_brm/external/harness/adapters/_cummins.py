"""Shared code for the Cummins (2025) adapters (arXiv 2509.13397, OSF skbqm, Study 1).

The contrast list below was fixed on 3 Oct 2026 BEFORE any ratio was computed and is not to be edited
after results are seen:

  political_identity  Conservative (Strongly, Moderately, Slightly Conservative) minus
                      Liberal (Strongly, Moderately, Slightly Liberal); "Neutral (Moderate)" in neither.
                      The extreme levels alone are not estimable (Strongly Conservative n = 2).
  sex                 f minus m
  education           Bachelor's degree or Graduate degree  minus  all lower levels
                      (Not a high school graduate, High school graduate, Some college or associate's)
  age                 older minus younger; older = age > 27, younger = age <= 27, where 27 is the
                      median of the 85 human participants. Simulated personas carry the matched
                      human's age and take the same cut.

Outcomes: bjw_score (sum of 6 items, 6 to 36) and gut_diff (gut_y minus gut_x, -9 to 9).
All gaps are marginal mean differences (N = 85 leaves no room to standardise over other attributes).
"""
import os

import numpy as np
import pandas as pd

RAW = r"C:\Research\PsychBench\UpdatedRun\paper_brm\external\raw\cummins2025"
FULL = os.path.join(RAW, "full_data.csv")
GT = os.path.join(RAW, "ground_truth_data.csv")
COMPLETION = os.path.join(RAW, "configuration_completion_summary.csv")
CFG = ["model", "temperature", "reasoning_effort", "prompt_content", "demographics_used"]
OUTCOMES = ["bjw_score", "gut_diff"]
CONTRASTS = ["political_identity", "sex", "education", "age"]
B = 2000
SEED_MAIN = 20261003

CONSERVATIVE = ["Strongly Conservative", "Moderately Conservative", "Slightly Conservative"]
LIBERAL = ["Strongly Liberal", "Moderately Liberal", "Slightly Liberal"]
HIGHER_EDU = ["Bachelor's degree", "Graduate degree or graduate education"]
# attributes the LLM was told, by demographics_used (paper, Method: none / age-and-gender only / extensive)
INFORMED = {"none": set(), "minimal": {"sex", "age"}, "extensive": set(CONTRASTS)}


def load_human():
    gt = pd.read_csv(GT).sort_values("participant_id").reset_index(drop=True)
    med = float(gt.age.median())
    masks = {
        "political_identity": (gt.political_identity.isin(CONSERVATIVE).to_numpy(), gt.political_identity.isin(LIBERAL).to_numpy()),
        "sex": ((gt.sex == "f").to_numpy(), (gt.sex == "m").to_numpy()),
        "education": (gt.education.isin(HIGHER_EDU).to_numpy(), (~gt.education.isin(HIGHER_EDU)).to_numpy()),
        "age": ((gt.age > med).to_numpy(), (gt.age <= med).to_numpy()),
    }
    return gt, masks, med


def load_sim():
    """One (85, 2) array per configuration: bjw_score, gut_diff for participants with complete_data == 1,
    NaN elsewhere. Returns dict cfg_tuple -> array, plus the completion table."""
    fd = pd.read_csv(FULL, keep_default_na=False)
    ai = fd[(fd.dgp == "ai") & (fd.complete_data == 1)]
    out = {}
    for key, d in ai.groupby(CFG, sort=True):
        arr = np.full((85, 2), np.nan)
        for j, sc in enumerate(OUTCOMES):
            s = d[d.scale == sc]
            if s.participant_id.duplicated().any():
                raise ValueError(f"duplicate rows in {key} {sc}")
            arr[s.participant_id.to_numpy() - 1, j] = s.score.to_numpy(float)
        out[key] = arr
    comp = pd.read_csv(COMPLETION, keep_default_na=False)
    return out, comp


def mean_gap(x, ok, hi, lo):
    """Point gap; NaN when a side has no available person."""
    a, b = hi & ok, lo & ok
    if not a.any() or not b.any():
        return np.nan
    return float(x[a].mean() - x[b].mean())


def boot_gap(x, ok, hi, lo, counts):
    """Gap in every bootstrap sample. counts: (B, n) multinomial resample counts."""
    xi = np.where(ok, x, 0.0)
    out = []
    for m in (hi, lo):
        w = (m & ok).astype(float)
        num, den = counts @ (xi * w), counts @ w
        out.append(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan))
    return out[0] - out[1]


def resample_counts(rng, n, nboot=B):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=nboot).astype(float)
