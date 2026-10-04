"""Test C. Does higher individual accuracy predict better partisan-gap fidelity?

x = the authors' individual accuracy (summary.csv, paper_accuracy; 12 specifications, the
demographics-only run has none). A second x is recomputed here for all 13 specifications with the authors'
task-level definition (per column 1 - |twin - waves-1-3 answer| / range, range over waves 1-3, wave 4 and
twin answers; mean over a task's columns, over respondents, over the 17 tasks). It matches the paper only
where free-text anchoring and probability-matching are not involved, so it is used for ranking only.
y = mean over the 8 items of the Republican - Democrat ratio g/p (00_gaps_all_specs.csv), and the mean
absolute distance of the ratio from 1.

Outputs
  C_spec_table.csv      per spec: paper_accuracy, own_accuracy, own accuracy on the 8 items, mean ratio, mean |ratio - 1|
  C_correlations.csv    Pearson and Spearman (with p and a bootstrap CI over specifications, B = 5000)
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import *

TASK = {"QID287": "false consensus", "QID290": "false consensus", "QID156": "base rate", "QID154": "base rate",
        "QID158": "framing problem", "QID157": "framing problem", "QID160": "conjunction problem (Linda)",
        "QID159": "conjunction problem (Linda)", "QID162": "outcome bias", "QID161": "outcome bias",
        "QID181": "sunk cost fallacy", "QID182": "sunk cost fallacy", "QID183": "absolute vs. relative savings",
        "QID184": "absolute vs. relative savings", "QID189": "WTA/WTP-Thaler", "QID190": "WTA/WTP-Thaler",
        "QID191": "WTA/WTP-Thaler", "QID192": "Allais", "QID193": "Allais", "QID194": "myside", "QID195": "myside",
        "QID198": "prob matching vs. max", "QID203": "prob matching vs. max", "QID288": "non-separability of risks and benefits",
        "QID289": "non-separability of risks and benefits", "QID291": "omission", "QID196": "denominator neglect",
        "QID9": "pricing", **{f"QID{k}": "anchoring and adjustment" for k in range(163, 171)},
        **{f"QID{k}": "less is more" for k in range(171, 180)}}
SEED = 20261003
cols = list(F2C.values())
sims, h4, h13, demo = load_all(cols)


def own_accuracy(S):
    acc = {}
    for cc in cols:
        a, b = S[cc], h13.loc[S.index, cc]
        stack = np.r_[b.values, h4.loc[S.index, cc].values, a.values]
        lo, hi = np.nanmin(stack), np.nanmax(stack)
        acc[cc] = 1 - (a - b).abs() / (hi - lo)
    acc = pd.DataFrame(acc)
    task = pd.Series({t: acc[[c for c in cols if TASK[c.split("_")[0]] == t]].mean(axis=1).mean() for t in sorted(set(TASK.values()))})
    items8 = acc[ITEMS].mean(axis=1).mean()
    return 100 * task.mean(), 100 * items8


gaps = pd.read_csv(os.path.join(OUT, "00_gaps_all_specs.csv"))
rows = []
for k, S in sims.items():
    t, i8 = own_accuracy(S)
    g = gaps[gaps.run == k]
    rows.append(dict(run=k, respondents=len(S), paper_accuracy=ACC.get(k), own_accuracy=t, own_accuracy_8items=i8,
                     mean_ratio=g.ratio.mean(), min_ratio=g.ratio.min(), max_ratio=g.ratio.max(),
                     mean_abs_ratio_minus_1=(g.ratio - 1).abs().mean()))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "C_spec_table.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(T.round(3).to_string(index=False))

llm = T[T.run != CEILING]
sets = {
    "12 specs with paper accuracy": (llm.dropna(subset=["paper_accuracy"]), "paper_accuracy"),
    "11 prompted specs (fine-tuned excluded), paper accuracy": (llm.dropna(subset=["paper_accuracy"]).query("run != 'finetune500_gpt41mini'"), "paper_accuracy"),
    "12 specs + ceiling, paper accuracy": (T.dropna(subset=["paper_accuracy"]), "paper_accuracy"),
    "13 specs, recomputed accuracy": (llm, "own_accuracy"),
    "13 specs, accuracy on the 8 items": (llm, "own_accuracy_8items"),
    "4 text-persona GPT-4.1-mini variants (n too small for inference), paper accuracy": (llm[llm.run.isin(["text_gpt41mini", "text_reasoning_gpt41mini", "text_repeatq_gpt41mini", "text_t07_gpt41mini"])], "paper_accuracy"),
}
rng = np.random.default_rng(SEED)
out = []
for name, (d, xc) in sets.items():
    for yc in ("mean_ratio", "mean_abs_ratio_minus_1"):
        x, y = d[xc].values, d[yc].values
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        bs_p, bs_s = [], []
        for _ in range(5000):
            i = rng.integers(0, len(x), len(x))
            if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
                continue
            bs_p.append(stats.pearsonr(x[i], y[i])[0])
            bs_s.append(stats.spearmanr(x[i], y[i])[0])
        out.append(dict(set=name, n=len(x), accuracy=xc, fidelity=yc, pearson=pr, pearson_p=pp,
                        pearson_lo=np.percentile(bs_p, 2.5), pearson_hi=np.percentile(bs_p, 97.5),
                        spearman=sr, spearman_p=sp, spearman_lo=np.percentile(bs_s, 2.5), spearman_hi=np.percentile(bs_s, 97.5)))
C = pd.DataFrame(out)
C.to_csv(os.path.join(OUT, "C_correlations.csv"), index=False)
print(C.round(3).to_string(index=False))
# note: the ceiling row of C_spec_table.csv has own_accuracy 100 by construction (the twin is the waves-1-3 answer); its paper value 81.72 is the test-retest accuracy
# accuracy check: recomputed versus the paper
d = llm.dropna(subset=["paper_accuracy"])
print("recomputed vs paper accuracy: Pearson", round(stats.pearsonr(d.paper_accuracy, d.own_accuracy)[0], 3),
      "max abs diff", round((d.paper_accuracy - d.own_accuracy).abs().max(), 2))
