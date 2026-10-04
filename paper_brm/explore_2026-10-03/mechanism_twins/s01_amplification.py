"""Test A. Amplification on the 8 certifiable items (Republican and Democrat respondents).

Sample per specification and item: Republican or Democrat respondents with a twin answer, a wave-4
answer and a waves-1-3 answer (the "triple sample").

Outputs
  A_slopes_by_item.csv     per spec x item: n, slope of twin on wave-4 answer, on waves-1-3 answer, joint
                           slopes, corr, human test-retest slope (wave 4 on waves 1-3)
  A_slopes_pooled.csv      per spec: item-centred pooled slopes with respondent bootstrap CIs (B = 500)
  A_within_party.csv       per spec x item x party: SD (twin, wave 4, waves 1-3), share at the human
                           party mode, share at either endpoint, number of answer values used
  A_within_party_summary.csv   per spec: item means of the above
  A_distributions.csv      per spec x item x party x answer code: share of twins and of humans (wave 4)
"""
import os

import numpy as np
import pandas as pd

from common import *

B = 500
SEED = 20261003
sims, h4, h13, demo = load_all()
party = demo.party


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def triple(S, it):
    d = pd.DataFrame(dict(t=S[it], y4=h4.loc[S.index, it], y13=h13.loc[S.index, it]), index=S.index)
    d["party"] = party.reindex(d.index)
    return d.dropna().query("party in ['Republican','Democrat']")


rows, wrows, drows = [], [], []
for k, S in sims.items():
    for it in ITEMS:
        d = triple(S, it)
        t, y4, y13 = d.t.values, d.y4.values, d.y13.values
        vt = t.var(ddof=1)
        row = dict(run=k, outcome=it, n=len(d),
                   slope_on_w4=np.cov(t, y4)[0, 1] / y4.var(ddof=1),
                   slope_on_w13=np.cov(t, y13)[0, 1] / y13.var(ddof=1),
                   corr_w4=np.corrcoef(t, y4)[0, 1] if vt > 0 else np.nan,
                   corr_w13=np.corrcoef(t, y13)[0, 1] if vt > 0 else np.nan,
                   retest_slope_w4_on_w13=np.cov(y4, y13)[0, 1] / y13.var(ddof=1),
                   retest_corr=np.corrcoef(y4, y13)[0, 1],
                   sd_twin=t.std(ddof=1), sd_w4=y4.std(ddof=1), sd_w13=y13.std(ddof=1))
        if vt > 0:
            b = ols(np.column_stack([y4, y13]), t)
            row.update(joint_b_w4=b[1], joint_b_w13=b[2])
        rows.append(row)
        for pty in ("Republican", "Democrat"):
            e = d[d.party == pty]
            mode = e.y4.value_counts().sort_values(ascending=False)
            mode_val = mode.index[0]
            wrows.append(dict(run=k, outcome=it, party=pty, n=len(e),
                              sd_twin=e.t.std(ddof=1), sd_w4=e.y4.std(ddof=1), sd_w13=e.y13.std(ddof=1),
                              human_mode=mode_val,
                              share_human_at_mode=(e.y4 == mode_val).mean(),
                              share_twin_at_human_mode=(e.t == mode_val).mean(),
                              share_twin_at_own_mode=e.t.value_counts(normalize=True).iloc[0],
                              share_human_at_ends=e.y4.isin([1, 5]).mean(),
                              share_twin_at_ends=e.t.isin([1, 5]).mean(),
                              mean_twin=e.t.mean(), mean_w4=e.y4.mean(), mean_w13=e.y13.mean(),
                              values_used_twin=e.t.nunique(), values_used_w4=e.y4.nunique()))
            for code in (1, 2, 3, 4, 5):
                drows.append(dict(run=k, outcome=it, party=pty, code=code, share_twin=(e.t == code).mean(),
                                  share_w4=(e.y4 == code).mean(), share_w13=(e.y13 == code).mean()))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "A_slopes_by_item.csv"), index=False)
W = pd.DataFrame(wrows)
W.to_csv(os.path.join(OUT, "A_within_party.csv"), index=False)
pd.DataFrame(drows).to_csv(os.path.join(OUT, "A_distributions.csv"), index=False)
sm = (W.groupby("run")
      .agg(sd_twin=("sd_twin", "mean"), sd_w4=("sd_w4", "mean"), sd_w13=("sd_w13", "mean"),
           share_human_at_mode=("share_human_at_mode", "mean"),
           share_twin_at_human_mode=("share_twin_at_human_mode", "mean"),
           share_twin_at_own_mode=("share_twin_at_own_mode", "mean"),
           share_human_at_ends=("share_human_at_ends", "mean"), share_twin_at_ends=("share_twin_at_ends", "mean"),
           values_used_twin=("values_used_twin", "mean"), values_used_w4=("values_used_w4", "mean")).reset_index())
sm["sd_ratio_twin_over_w4"] = sm.sd_twin / sm.sd_w4
sm.to_csv(os.path.join(OUT, "A_within_party_summary.csv"), index=False)

# pooled, item-centred slopes with a respondent bootstrap (a respondent is resampled once for all items)
rng = np.random.default_rng(SEED)
ids_all = sorted(set(demo.index[party.isin(["Republican", "Democrat"])]))
prow = []
for k, S in sims.items():
    # wide frame: respondent x item, keep respondents with all three values on at least one item
    frames = {it: triple(S, it) for it in ITEMS}
    ids = sorted(set().union(*[set(f.index) for f in frames.values()]))
    pos = {r: i for i, r in enumerate(ids)}
    n = len(ids)
    T = np.full((n, len(ITEMS)), np.nan); Y4 = T.copy(); Y13 = T.copy()
    for j, it in enumerate(ITEMS):
        f = frames[it]
        ii = [pos[r] for r in f.index]
        T[ii, j], Y4[ii, j], Y13[ii, j] = f.t.values, f.y4.values, f.y13.values

    def pooled(idx):
        t, a, b = T[idx], Y4[idx], Y13[idx]
        ok = ~np.isnan(t)
        # item-centre using the resampled data
        out = []
        for M in (t, a, b):
            M = M.copy()
            M[~ok] = np.nan
            M = M - np.nanmean(M, axis=0)
            out.append(M)
        tc, ac, bc = out
        m = ok.ravel()
        x4, x13, yy = ac.ravel()[m], bc.ravel()[m], tc.ravel()[m]
        s4 = (x4 * yy).sum() / (x4 * x4).sum()
        s13 = (x13 * yy).sum() / (x13 * x13).sum()
        X = np.column_stack([x4, x13])
        j4, j13 = np.linalg.lstsq(X, yy, rcond=None)[0] if yy.var() > 0 else (np.nan, np.nan)
        rt = (x4 * x13).sum() / (x13 * x13).sum()
        return s4, s13, j4, j13, rt, yy.std(ddof=1) / x4.std(ddof=1)

    est = pooled(np.arange(n))
    boots = np.array([pooled(rng.integers(0, n, n)) for _ in range(B)])
    names = ["slope_on_w4", "slope_on_w13", "joint_b_w4", "joint_b_w13", "retest_slope_w4_on_w13", "sd_ratio_twin_over_w4"]
    r = dict(run=k, respondents=n)
    for i, nm in enumerate(names):
        r[nm] = est[i]
        r[nm + "_lo"], r[nm + "_hi"] = np.nanpercentile(boots[:, i], [2.5, 97.5])
    prow.append(r)
    print("pooled", k, flush=True)
P = pd.DataFrame(prow)
P.to_csv(os.path.join(OUT, "A_slopes_pooled.csv"), index=False)

pd.set_option("display.width", 250, "display.max_columns", 40)
print(P[["run", "respondents", "slope_on_w4", "slope_on_w4_lo", "slope_on_w4_hi", "slope_on_w13", "joint_b_w4", "joint_b_w13",
         "retest_slope_w4_on_w13", "sd_ratio_twin_over_w4"]].round(3).to_string(index=False))
print(sm.round(3).to_string(index=False))
