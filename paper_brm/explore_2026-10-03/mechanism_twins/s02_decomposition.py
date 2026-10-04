"""Test B. Where the twin partisan gap comes from.

Sample per specification and item: Republican or Democrat respondents with a twin answer, a wave-4
answer and a waves-1-3 answer. Rep = 1 for Republican, 0 for Democrat.

For an outcome y (twin answer, or the human wave-4 answer as the benchmark) fit
    y = a + b_party * Rep + b_own * h13            (M1, primary)
    y = a + b_party * Rep + b_own * h13 + b_ide * ideology   (M2, adds the 1-5 political-views item
                                                       the twin persona also contains)
The Republican - Democrat difference in means of y is an exact identity of the fit:
    gap = b_party + b_own * (mean h13 in R - mean h13 in D) [+ b_ide * (mean ideology R - D)]
so gap = "own earlier answers" part + "party" part (+ ideology part). Components are signed so that the
human gap is positive (items are oriented by the sign of the human waves-1-3 R - D gap).

M0 (univariate): the part of the twin gap predicted by the person's own earlier answer alone,
    b0 * (mean h13 R - D), with b0 the slope of y on h13 without party; the rest is "added beyond".

Outputs
  B_by_item.csv     per spec x item x outcome (twin / human) the components for M0, M1, M2
  B_pooled.csv      per spec: item-averaged oriented components, twin and human, excess = twin - human,
                    respondent-bootstrap 95% CI (B = 500)
"""
import os

import numpy as np
import pandas as pd

from common import *

B = 500
SEED = 20261003
sims, h4, h13, demo = load_all()
rd = demo.party.isin(["Republican", "Democrat"])
ids_rd = demo.index[rd]
REP = (demo.party == "Republican").astype(float)
IDE = demo.ideology

# orientation: sign of the human waves-1-3 R - D gap over all Republican and Democrat respondents
SIGN = {}
for it in ITEMS:
    x = h13[it]
    SIGN[it] = np.sign(x[REP.index[(REP == 1) & rd]].mean() - x[REP.index[(REP == 0) & rd]].mean())


def fit(y, rep, h, ide, with_ide):
    cols = [np.ones(len(y)), rep, h] + ([ide] if with_ide else [])
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    dh = h[rep == 1].mean() - h[rep == 0].mean()
    di = ide[rep == 1].mean() - ide[rep == 0].mean()
    return beta, dh, di


def components(y, rep, h, ide):
    """gap decomposition of outcome y; returns dict."""
    gap = y[rep == 1].mean() - y[rep == 0].mean()
    dh = h[rep == 1].mean() - h[rep == 0].mean()
    # M0: univariate slope on own earlier answer
    x = h - h.mean()
    b0 = (x * (y - y.mean())).sum() / (x * x).sum()
    out = dict(gap=gap, dh13=dh, m0_own=b0 * dh, m0_beyond=gap - b0 * dh, m0_slope=b0)
    beta, dh, di = fit(y, rep, h, ide, False)
    out.update(m1_party=beta[1], m1_own=beta[2] * dh, m1_slope=beta[2])
    beta, dh, di = fit(y, rep, h, ide, True)
    out.update(m2_party=beta[1], m2_own=beta[2] * dh, m2_ide=beta[3] * di, m2_slope=beta[2], m2_ide_slope=beta[3])
    return out


def wide(S):
    """arrays (n x items) of twin, w4, w13 over Republican/Democrat respondents; NaN where missing."""
    ids = [i for i in S.index if i in ids_rd]
    T = S.loc[ids, ITEMS].values
    Y4 = h4.loc[ids, ITEMS].values
    Y13 = h13.loc[ids, ITEMS].values
    return np.array(ids), T, Y4, Y13


KEYS = ["gap", "m0_own", "m0_beyond", "m1_party", "m1_own", "m2_party", "m2_own", "m2_ide"]


def item_stats(rows_idx, ids, T, Y4, Y13, j):
    """components for item j on the resampled rows (index array into ids)."""
    t, y4, y13 = T[rows_idx, j], Y4[rows_idx, j], Y13[rows_idx, j]
    rid = ids[rows_idx]
    rep, ide = REP.loc[rid].values, IDE.loc[rid].values
    ok = ~(np.isnan(t) | np.isnan(y4) | np.isnan(y13) | np.isnan(ide))
    # ideology missing is not expected; the mask keeps M0 and M1 on the same rows as M2
    t, y4, y13, rep, ide = t[ok], y4[ok], y13[ok], rep[ok], ide[ok]
    s = SIGN[ITEMS[j]]
    res = {}
    for who, y in (("twin", t), ("human", y4)):
        c = components(y, rep, y13, ide)
        for key in KEYS:
            res[(who, key)] = s * c[key]
        res[(who, "m1_slope")] = c["m1_slope"]
        res[(who, "m0_slope")] = c["m0_slope"]
        res[(who, "dh13")] = s * c["dh13"]
    return res


by_item, pooled = [], []
rng = np.random.default_rng(SEED)
for k, S in sims.items():
    ids, T, Y4, Y13 = wide(S)
    n = len(ids)
    full = np.arange(n)
    point = [item_stats(full, ids, T, Y4, Y13, j) for j in range(len(ITEMS))]
    for j, it in enumerate(ITEMS):
        for who in ("twin", "human"):
            r = dict(run=k, outcome=it, who=who)
            r.update({key: point[j][(who, key)] for key in KEYS + ["m1_slope", "m0_slope", "dh13"]})
            by_item.append(r)

    def agg(res_list):
        o = {}
        for who in ("twin", "human"):
            for key in KEYS + ["m1_slope", "m0_slope"]:
                o[f"{who}_{key}"] = np.mean([r[(who, key)] for r in res_list])
        for key in KEYS:
            o[f"excess_{key}"] = o[f"twin_{key}"] - o[f"human_{key}"]
        return o

    est = agg(point)
    boots = pd.DataFrame([agg([item_stats(idx, ids, T, Y4, Y13, j) for j in range(len(ITEMS))])
                          for idx in (rng.integers(0, n, n) for _ in range(B))])
    r = dict(run=k, respondents=n)
    for nm, v in est.items():
        r[nm] = v
        r[nm + "_lo"], r[nm + "_hi"] = np.percentile(boots[nm], [2.5, 97.5])
    pooled.append(r)
    print("decomp", k, flush=True)

pd.DataFrame(by_item).to_csv(os.path.join(OUT, "B_by_item.csv"), index=False)
P = pd.DataFrame(pooled)
P.to_csv(os.path.join(OUT, "B_pooled.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
show = ["run", "twin_gap", "human_gap", "twin_m1_own", "twin_m1_party", "human_m1_own", "human_m1_party",
        "excess_gap", "excess_m1_own", "excess_m1_party", "twin_m1_slope", "human_m1_slope"]
print(P[show].round(3).to_string(index=False))
show = ["run", "twin_m0_own", "twin_m0_beyond", "human_m0_own", "human_m0_beyond", "twin_m2_party", "twin_m2_own", "twin_m2_ide",
        "human_m2_party", "human_m2_own", "human_m2_ide"]
print(P[show].round(3).to_string(index=False))
