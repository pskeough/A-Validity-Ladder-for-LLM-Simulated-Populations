"""Test D. Do the fine-tuned twins shrink the partisan gap because they pull every answer toward the grand
mean (a smaller overall SD)?

Sample: the 1,558 respondents of the fine-tuned release, for every specification (so default, demographics-only
and the human wave-4 answers are compared on the same people). Republican and Democrat respondents only
(item-wise complete pairs). Items oriented as in s02b (y -> 6 - y where the human R - D gap is negative), then
item means of each statistic over the 8 items. Respondent bootstrap, B = 1000.

Per specification:
  gap_ratio             mean twin gap / mean human gap (ratio of item means)
  sd_ratio_all          item-mean overall SD of twin answers / of human wave-4 answers
  sd_ratio_within       the same for pooled within-party SD
  d_ratio               standardised gap (gap / overall SD) twin over human
  gap_if_sd_matched     twin gap rescaled by (human SD / twin SD) about the grand mean, as a ratio to the human gap
  share_log_shrink_by_sd  ln(sd_ratio_all) / ln(gap_ratio): the share of the proportional gap shrinkage that a
                        pure overall-SD compression would account for (read only where gap_ratio < 1)
  r2_party              share of answer variance that is between party
  tv_marginal           total variation distance between the twin and human answer distributions (all R/D)
  tv_within_party       the same within party, averaged over the two parties
  corr_w4               correlation of twin with the person's own wave-4 answer
  grand_mean_diff       twin grand mean minus human grand mean (oriented)
"""
import os

import numpy as np
import pandas as pd

from common import *

B = 1000
SEED = 20261003
sims, h4, h13, demo = load_all()
ft_ids = sims["finetune500_gpt41mini"].index
party = demo.party
rd = party.isin(["Republican", "Democrat"])
ids = np.array([i for i in ft_ids if rd.get(i, False)])
REP = (party.loc[ids] == "Republican").astype(int).values

SIGN = {}
for it in ITEMS:
    x = h13[it][rd]
    SIGN[it] = np.sign(x[party[rd] == "Republican"].mean() - x[party[rd] == "Democrat"].mean())


def orient(M):
    M = M.copy()
    for j, it in enumerate(ITEMS):
        if SIGN[it] < 0:
            M[:, j] = 6.0 - M[:, j]
    return M


Y = orient(h4.reindex(ids)[ITEMS].values)
specs = {k: orient(S.reindex(ids)[ITEMS].values) for k, S in sims.items()}


def tv(a, b):
    ca = np.array([(a == c).mean() for c in (1, 2, 3, 4, 5)])
    cb = np.array([(b == c).mean() for c in (1, 2, 3, 4, 5)])
    return 0.5 * np.abs(ca - cb).sum()


def stats(T, idx):
    t_all, y_all, rep_all = T[idx], Y[idx], REP[idx]
    out = {k: [] for k in ("g", "p", "sd_t", "sd_y", "w_t", "w_y", "r2_t", "r2_y", "tvm", "tvw", "corr", "gm_t", "gm_y")}
    for j in range(len(ITEMS)):
        ok = ~(np.isnan(t_all[:, j]) | np.isnan(y_all[:, j]))
        t, y, rep = t_all[ok, j], y_all[ok, j], rep_all[ok]
        out["g"].append(t[rep == 1].mean() - t[rep == 0].mean())
        out["p"].append(y[rep == 1].mean() - y[rep == 0].mean())
        out["sd_t"].append(t.std(ddof=1)); out["sd_y"].append(y.std(ddof=1))
        wt = np.r_[t[rep == 1] - t[rep == 1].mean(), t[rep == 0] - t[rep == 0].mean()]
        wy = np.r_[y[rep == 1] - y[rep == 1].mean(), y[rep == 0] - y[rep == 0].mean()]
        out["w_t"].append(np.sqrt((wt ** 2).sum() / (len(t) - 2))); out["w_y"].append(np.sqrt((wy ** 2).sum() / (len(y) - 2)))
        out["r2_t"].append(1 - (wt ** 2).sum() / ((t - t.mean()) ** 2).sum() if t.var() > 0 else np.nan)
        out["r2_y"].append(1 - (wy ** 2).sum() / ((y - y.mean()) ** 2).sum())
        out["tvm"].append(tv(t, y))
        out["tvw"].append(0.5 * (tv(t[rep == 1], y[rep == 1]) + tv(t[rep == 0], y[rep == 0])))
        out["corr"].append(np.corrcoef(t, y)[0, 1] if t.var() > 0 else np.nan)
        out["gm_t"].append(t.mean()); out["gm_y"].append(y.mean())
    m = {k: np.nanmean(v) for k, v in out.items()}
    r = dict(gap_twin=m["g"], gap_human=m["p"], gap_ratio=m["g"] / m["p"],
             sd_ratio_all=m["sd_t"] / m["sd_y"], sd_ratio_within=m["w_t"] / m["w_y"],
             d_ratio=(m["g"] / m["sd_t"]) / (m["p"] / m["sd_y"]),
             gap_if_sd_matched=(m["g"] * m["sd_y"] / m["sd_t"]) / m["p"],
             r2_party_twin=m["r2_t"], r2_party_human=m["r2_y"], tv_marginal=m["tvm"], tv_within_party=m["tvw"],
             corr_w4=m["corr"], grand_mean_diff=m["gm_t"] - m["gm_y"], sd_twin=m["sd_t"], sd_human=m["sd_y"])
    lr = np.log(r["gap_ratio"]) if r["gap_ratio"] > 0 else np.nan
    r["share_log_shrink_by_sd"] = np.log(r["sd_ratio_all"]) / lr if (lr < 0 and r["sd_ratio_all"] > 0) else np.nan
    return r


rng = np.random.default_rng(SEED)
n = len(ids)
boot_idx = [rng.integers(0, n, n) for _ in range(B)]
rows = []
for k, T in specs.items():
    est = stats(T, np.arange(n))
    bs = pd.DataFrame([stats(T, i) for i in boot_idx])
    r = dict(run=k, respondents=n)
    for nm, v in est.items():
        r[nm] = v
        if nm in ("gap_ratio", "sd_ratio_all", "sd_ratio_within", "d_ratio", "gap_if_sd_matched", "share_log_shrink_by_sd", "corr_w4", "r2_party_twin"):
            r[nm + "_lo"], r[nm + "_hi"] = np.nanpercentile(bs[nm], [2.5, 97.5])
    rows.append(r)
    print("D", k, flush=True)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "D_finetuned_shrinkage.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
show = ["run", "gap_ratio", "gap_ratio_lo", "gap_ratio_hi", "sd_ratio_all", "sd_ratio_within", "d_ratio", "gap_if_sd_matched",
        "share_log_shrink_by_sd", "r2_party_twin", "r2_party_human", "tv_marginal", "tv_within_party", "corr_w4", "grand_mean_diff"]
print(R[show].round(3).to_string(index=False))
