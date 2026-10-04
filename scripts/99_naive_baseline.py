"""Naive baseline checks on the worked example, for comparison with the ladder.

Two checks common in silicon-sample studies, run on the same 48 matched persona cells the ladder reads:
  1. Correlation of simulated persona means with the NHANES means of matching adults (the
     "algorithmic fidelity" style check). Pearson and Spearman, with a cell bootstrap interval.
  2. Direction of the seven level-2 group gaps (does each simulated gap have the population's sign?).
Persona means average both framings, as in the ladder's headline level-2 and level-3 readings.

Inputs: analysis/brm/80b_sim_cells.csv (persona means, corpus 'all'), analysis/brm/80a_cell_counts.csv
(NHANES 2005-2018 cell means, poverty-ratio income bands, married/single), analysis/brm/l2_verdicts.csv
(headline standardised gaps). Output: analysis/brm/99_naive_baseline.csv.
"""
import os

import numpy as np
import pandas as pd

BRM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "brm")
MODELS = {"deepseek-chat-v3": "DeepSeek-V3", "gemini-3-flash-preview": "Gemini-3-Flash",
          "gpt-4o-mini": "GPT-4o-mini", "glm-4.7": "GLM-4.7"}
L2_NAMES = {"deepseek/deepseek-chat-v3": "DeepSeek-V3", "google/gemini-3-flash-preview": "Gemini-3-Flash",
            "openai/gpt-4o-mini": "GPT-4o-mini", "z-ai/glm-4.7": "GLM-4.7"}
KEYS = ["race", "sex", "inc", "mar"]
RNG = np.random.default_rng(20261004)
B = 2000


def model_name(raw):
    for k, v in MODELS.items():
        if k in raw:
            return v
    raise ValueError(raw)


def boot_ci(x, y, fn):
    n = len(x)
    stats = []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        stats.append(fn(x[i], y[i]))
    return np.percentile(stats, [2.5, 97.5])


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(pd.Series(x).rank().values, pd.Series(y).rank().values)


def main():
    sim = pd.read_csv(os.path.join(BRM, "80b_sim_cells.csv"))
    sim = sim[sim.corpus == "all"].copy()
    sim["model"] = sim.model.map(model_name)
    sim = sim.groupby(["model"] + KEYS, as_index=False)["mean"].mean()

    ref = pd.read_csv(os.path.join(BRM, "80a_cell_counts.csv"))
    ref = ref[(ref.window == "2005-2018") & (ref.income == "pir_band") & (ref.marital == "mar")]
    ref = ref[KEYS + ["wmean_phq8"]]

    rows = []
    for m, d in sim.groupby("model"):
        j = d.merge(ref, on=KEYS, how="inner")
        x, y = j["mean"].values, j["wmean_phq8"].values
        lo, hi = boot_ci(x, y, pearson)
        slo, shi = boot_ci(x, y, spearman)
        rows.append(dict(model=m, check="cell_correlation", n=len(j), pearson=pearson(x, y),
                         pearson_lo=lo, pearson_hi=hi, spearman=spearman(x, y), spearman_lo=slo,
                         spearman_hi=shi, sim_mean=x.mean(), ref_mean=y.mean()))

    v = pd.read_csv(os.path.join(BRM, "l2_verdicts.csv"))
    v = v[(v.analysis == "headline") & (v.corpus == "all") & (v.framing == "combined")
          & (v.estimand == "standardised") & (v.ref_variant == "primary") & (v.scope.isin(L2_NAMES))]
    for scope, d in v.groupby("scope"):
        same = int((np.sign(d.simulated) == np.sign(d.population)).sum())
        rows.append(dict(model=L2_NAMES[scope], check="gap_direction", n=len(d), same_direction=same,
                         kept=int((d.verdict == "kept").sum())))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(BRM, "99_naive_baseline.csv"), index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
