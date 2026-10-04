"""Receipt for cummins2025: numbers Cummins (2025) prints for Study 1, recomputed from the OSF files.
Writes results/cummins2025/receipt.csv (item, ours, published, source_location)."""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapters import _cummins as C  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cummins2025", "receipt.csv")
PDF = "arXiv 2509.13397 PDF (In Press, AMPPS version, fetched 3 Oct 2026)"

fd = pd.read_csv(C.FULL, keep_default_na=False)
gt = pd.read_csv(C.GT)
comp = pd.read_csv(C.COMPLETION, keep_default_na=False)
ai = fd[fd.dgp == "ai"]

rows = []


def add(item, ours, published, loc):
    rows.append(dict(item=item, ours=ours, published=published, source_location=loc))


add("configurations", ai.groupby(C.CFG).ngroups, 252, f"{PDF}, Study 1 Method, Analytic Features and Silicon Samples, p.4")
add("humans", len(gt), 85, f"{PDF}, Study 1 Method, Sample, p.4")
comp = comp.assign(n_complete=(comp.completion_rate * 85).round().astype(int))
add("simulated participants (configurations x 85)", ai.groupby(C.CFG).ngroups * 85, 21420, f"{PDF}, Study 1 Method, p.4")
cc = ai[ai.complete_data == 1].drop_duplicates(C.CFG + ["participant_id"])
add("simulated participants with complete data", len(cc), 21153, f"{PDF}, Study 1 Method, p.4")
add("completion rate minimum", round(float(comp.completion_rate.min()), 2), 0.21, f"{PDF}, Results, Success completion rates, p.6")
add("completion rate mean", round(float(comp.completion_rate.mean()), 2), 0.99, f"{PDF}, Results, Success completion rates, p.6")
add("completion rate median", round(float(comp.completion_rate.median()), 2), 1.00, f"{PDF}, Results, Success completion rates, p.6")
add("configurations below 50% completion", int((comp.completion_rate < 0.5).sum()), 1, f"{PDF}, Results, Success completion rates, p.6")

# Data Feature 1: Spearman rho between human and simulated scores per configuration and scale,
# complete participants, configurations at or above 50% completion (03_analysis.qmd lines 745-812)
keep = comp[comp.completion_rate >= 0.5][C.CFG]
a = ai[(ai.complete_data == 1)].merge(keep, on=C.CFG)
h = fd[fd.dgp == "human"][["scale", "participant_id", "score"]].rename(columns={"score": "human"})
a = a.merge(h, on=["scale", "participant_id"])
rho = []
for kk, d in a.groupby(C.CFG + ["scale"]):
    key, sc = kk[:-1], kk[-1]
    if d.score.nunique() < 2 or d.human.nunique() < 2:
        r = np.nan
    else:
        r = stats.spearmanr(d.human, d.score)[0]
    rho.append(dict(zip(C.CFG, key), scale=sc, rho=r, n=len(d)))
rho = pd.DataFrame(rho)
for sc, lo, hi, nconf in [("bjw_score", -0.24, 0.40, 246), ("gut_diff", -0.29, 0.22, 233)]:
    r = rho[rho.scale == sc].rho.dropna()
    add(f"Spearman rho range, {sc}: min", round(float(r.min()), 2), lo, f"{PDF}, Results, Data Feature 1, p.6")
    add(f"Spearman rho range, {sc}: max", round(float(r.max()), 2), hi, f"{PDF}, Results, Data Feature 1, p.6")
    add(f"configurations with a defined rho, {sc}", int(len(r)), nconf, f"{PDF}, Study 1 Method, Data Feature 1, p.5")

# human BJW-gut correlation, Pearson with Fisher 95% interval
r = stats.pearsonr(gt.bjw_score, gt.gut_diff)[0]
z, se = np.arctanh(r), 1 / np.sqrt(len(gt) - 3)
add("human BJW-Gut Feelings correlation r", round(float(r), 2), 0.26, f"{PDF}, Results, Data Feature 3, p.6")
add("human BJW-Gut Feelings correlation 95% CI low", round(float(np.tanh(z - 1.96 * se)), 2), 0.05, f"{PDF}, Results, Data Feature 3, p.6")
add("human BJW-Gut Feelings correlation 95% CI high", round(float(np.tanh(z + 1.96 * se)), 2), 0.45, f"{PDF}, Results, Data Feature 3, p.6")

# simulated between-scale correlation range (Data Feature 3), rule of 03_analysis.qmd 1204-1260 is
# not reproduced here (needs the authors' pairing and exclusions); the range is shown for reference
w = a.pivot_table(index=C.CFG + ["participant_id"], columns="scale", values="score").reset_index()
cr = []
for key, d in w.groupby(C.CFG):
    if d.bjw_score.nunique() > 1 and d.gut_diff.nunique() > 1:
        cr.append(stats.pearsonr(d.bjw_score, d.gut_diff)[0])
cr = np.array(cr)
add("simulated BJW-Gut correlation, min over configurations", round(float(cr.min()), 2), -0.40, f"{PDF}, Results, Data Feature 3, p.6")
add("simulated BJW-Gut correlation, max over configurations", round(float(cr.max()), 2), 0.78, f"{PDF}, Results, Data Feature 3, p.6")
add("simulated BJW-Gut correlation, mean over configurations", round(float(cr.mean()), 2), 0.21, f"{PDF}, Results, Data Feature 3, p.6")
add("configurations in that correlation (ours: variance in both scales)", int(len(cr)), 232, f"{PDF}, Study 1 Method, Data Feature 3, p.5")

pd.DataFrame(rows).to_csv(OUT, index=False)
print(pd.DataFrame(rows)[["item", "ours", "published"]].to_string(index=False))
