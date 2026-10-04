"""Test C. Variance decomposition and pattern concentration.

C1  Between-persona against within-persona variance of the PHQ-8 total and of each item, on the 48
    matched cells, against NHANES between-cell and within-cell variance. Per cell c: mean m_c,
    within-cell variance sigma2_c (draws: ddof 1; NHANES: weighted, unbiased reliability form), and the
    sampling variance tau2_c of the cell mean (draws: sigma2_c / n_c; NHANES: delete-one-PSU
    jackknife). With cell weights pi_c (equal, or the NHANES population shares N_c):
        within  = sum pi_c sigma2_c
        between = sum pi_c (m_c - mbar)^2  -  sum (pi_c - pi_c^2) tau2_c      (noise removed)
        share   = between / (within + between)
    Also the within-cell covariance structure of the items: total within variance = sum of item
    variances + 2 sum of covariances; mean inter-item correlation within cells.
C2  Pattern concentration. Distinct vectors, share of draws on the top 1, 10 and 50 vectors, the
    unbiased probability that two draws are the identical vector (any two draws, and pairs from
    different personas), and a decomposition of the log collision ratio against NHANES into
      pattern part   K(pi_model, c_model) / K(pi_model, c_nhanes)     same total mix, model patterns
      total-mix part K(pi_model, c_nhanes) / K(pi_nhanes, c_nhanes)    NHANES patterns, model total mix
    with K(pi, c) = sum_t pi_t^2 c_t over the totals both populations have (cross-persona collision c_t
    from script 79e). The script's own reuse ratio (count-weighted) is reproduced beside it.

Outputs: C_variance_decomposition.csv, C_variance_summary.csv, C_within_cov.csv, C_patterns.csv,
C_collision_decomposition.csv, C_top_vectors.csv
"""
import os

import numpy as np
import pandas as pd

from common import (DPQ, FRAMINGS, ITEMS, MODELS, OUT, SHORT, CellRef, cis_matched, load_corpus, load_nhanes,
                    sim_cells, wvar)

nh = load_nhanes()
corpus = load_corpus()
x = cis_matched(corpus)
sims = sim_cells(x)
d = nh[nh.c48 >= 0].reset_index(drop=True)
Nf_pop = CellRef(nh).Nf
pi_pop = Nf_pop / Nf_pop.sum()
pi_eq = np.full(48, 1 / 48)

# ---------------------------------------------------------------- NHANES cell moments (total + items)
cols = ["phq8"] + DPQ
labels = ["total"] + [f"item{i}" for i in range(1, 9)]
nh_mean, nh_within, nh_tau2 = {}, {}, {}
for col, lab in zip(cols, labels):
    cr = CellRef(nh, ycol=col)
    nh_mean[lab] = cr.mf.copy()
    nh_tau2[lab] = np.array([cr.des.jk_var(cr.mf[c], cr.mr[:, c]) for c in range(48)])
    w_in = np.empty(48)
    for c in range(48):
        g = d[d.c48 == c]
        w_in[c] = wvar(g[col].to_numpy(float), g.w.to_numpy(float))
    nh_within[lab] = w_in


def decomp(m, s2, tau2, pi):
    mbar = float(np.sum(pi * m))
    within = float(np.sum(pi * s2))
    b_obs = float(np.sum(pi * (m - mbar) ** 2))
    b = b_obs - float(np.sum((pi - pi ** 2) * tau2))
    return within, b, b_obs


rows = []
for wname, pi in (("equal weight over 48 cells", pi_eq), ("NHANES population shares", pi_pop)):
    for lab in labels:
        w_, b_, bo = decomp(nh_mean[lab], nh_within[lab], nh_tau2[lab], pi)
        rows.append(dict(source="NHANES", framing="", weighting=wname, quantity=lab, within=w_,
                         between=b_, between_observed=bo, total=w_ + b_, between_share=b_ / (w_ + b_)))
    for mdl in MODELS:
        for fr in FRAMINGS:
            g = x[(x.short == mdl) & (x.framing == fr)]
            for lab, col in zip(labels, ["total"] + ITEMS):
                grp = g.groupby("c48")[col]
                m = grp.mean().reindex(range(48)).to_numpy()
                s2 = grp.var(ddof=1).reindex(range(48)).to_numpy()
                n = grp.size().reindex(range(48)).to_numpy(float)
                w_, b_, bo = decomp(m, s2, s2 / n, pi)
                rows.append(dict(source=SHORT[mdl], framing=fr, weighting=wname, quantity=lab, within=w_,
                                 between=b_, between_observed=bo, total=w_ + b_, between_share=b_ / (w_ + b_)))
dec = pd.DataFrame(rows)
dec.to_csv(os.path.join(OUT, "C_variance_decomposition.csv"), index=False)

# summary: models averaged over the two framings; ratios to NHANES
summ = []
for wname in dec.weighting.unique():
    for lab in labels:
        nr = dec[(dec.source == "NHANES") & (dec.weighting == wname) & (dec.quantity == lab)].iloc[0]
        for mdl in MODELS:
            q = dec[(dec.source == SHORT[mdl]) & (dec.weighting == wname) & (dec.quantity == lab)]
            summ.append(dict(weighting=wname, quantity=lab, model=SHORT[mdl], within=q.within.mean(),
                             between=q.between.mean(), between_share=q.between_share.mean(),
                             within_ratio_to_nhanes=q.within.mean() / nr.within,
                             between_ratio_to_nhanes=q.between.mean() / nr["between"],
                             total_ratio_to_nhanes=q.total.mean() / nr.total,
                             nhanes_within=nr.within, nhanes_between=nr["between"],
                             nhanes_between_share=nr["between_share"]))
pd.DataFrame(summ).to_csv(os.path.join(OUT, "C_variance_summary.csv"), index=False)

# ---------------------------------------------------------------- within-cell covariance of items
cov_rows = []


def pooled_within_cov(Xc, cellid, wts, pi):
    S = np.zeros((8, 8))
    for c in range(48):
        m = cellid == c
        X = Xc[m]
        ww = wts[m]
        mu = np.average(X, axis=0, weights=ww)
        Z = X - mu
        den = ww.sum() - np.sum(ww ** 2) / ww.sum()
        S += pi[c] * (Z * ww[:, None]).T @ Z / den
    return S


def cov_row(source, framing, weighting, S):
    sd = np.sqrt(np.diag(S))
    R = S / np.outer(sd, sd)
    off = R[np.triu_indices(8, 1)]
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]
    return dict(source=source, framing=framing, weighting=weighting, total_within_var=float(S.sum()),
                sum_item_var=float(np.trace(S)), sum_2cov=float(S.sum() - np.trace(S)),
                share_total_var_from_covariance=float((S.sum() - np.trace(S)) / S.sum()),
                mean_interitem_r=float(off.mean()), lambda1_over_lambda2=float(ev[0] / ev[1]),
                var_ratio_if_items_independent=float(S.sum() / np.trace(S)))


Xn = d[DPQ].to_numpy(float)
for wname, pi in (("equal weight over 48 cells", pi_eq), ("NHANES population shares", pi_pop)):
    cov_rows.append(cov_row("NHANES", "", wname, pooled_within_cov(Xn, d.c48.to_numpy(), d.w.to_numpy(float), pi)))
    for mdl in MODELS:
        for fr in FRAMINGS:
            g = x[(x.short == mdl) & (x.framing == fr)]
            S = pooled_within_cov(g[ITEMS].to_numpy(float), g.c48.to_numpy(), np.ones(len(g)), pi)
            cov_rows.append(cov_row(SHORT[mdl], fr, wname, S))
pd.DataFrame(cov_rows).to_csv(os.path.join(OUT, "C_within_cov.csv"), index=False)

# ---------------------------------------------------------------- C2 patterns
pw = 4 ** np.arange(8)
key_n_all = (nh[DPQ].to_numpy(int) * pw).sum(axis=1)
tot_n_all = nh.phq8.to_numpy(int)
w_n_all = nh.w7.to_numpy(float)


def unbiased_collision(keys, weights=None):
    if weights is None:
        weights = np.ones(len(keys))
    _, inv = np.unique(keys, return_inverse=True)
    sk = np.bincount(inv, weights=weights)
    s2k = np.bincount(inv, weights=weights ** 2)
    sw, sw2 = weights.sum(), (weights ** 2).sum()
    return float(((sk ** 2).sum() - s2k.sum()) / (sw ** 2 - sw2))


def cross_collision(keys, persona, totals):
    out = {}
    df = pd.DataFrame(dict(k=keys, p=persona, t=totals))
    for t, g in df.groupby("t"):
        n = len(g)
        if n < 2:
            continue
        nk = g.k.value_counts().to_numpy(float)
        npk = g.groupby(["p", "k"]).size().to_numpy(float)
        np_ = g.p.value_counts().to_numpy(float)
        denom = n ** 2 - (np_ ** 2).sum()
        if denom <= 0:
            continue
        out[int(t)] = ((nk ** 2).sum() - (npk ** 2).sum()) / denom, n
    return out


nh_cc = cross_collision(key_n_all, np.arange(len(nh)), tot_n_all)
pat, dec2, topv = [], [], []
pat.append(dict(source="NHANES", framing="", n=len(nh), distinct=len(np.unique(key_n_all)),
                top1_share=pd.Series(key_n_all).value_counts(normalize=True).iloc[0],
                top10_share=pd.Series(key_n_all).value_counts(normalize=True).iloc[:10].sum(),
                top50_share=pd.Series(key_n_all).value_counts(normalize=True).iloc[:50].sum(),
                collision_unweighted=unbiased_collision(key_n_all),
                collision_weighted=unbiased_collision(key_n_all, w_n_all),
                effective_n_vectors=1 / unbiased_collision(key_n_all, w_n_all),
                total_simpson_effective_n=1 / np.sum(np.bincount(tot_n_all, minlength=25) ** 2 /
                                                      len(nh) ** 2)))
vc = pd.Series(key_n_all).value_counts(normalize=True).iloc[:5]
for rank, (k, v) in enumerate(vc.items(), 1):
    topv.append(dict(source="NHANES", framing="", rank=rank, share=v,
                     vector="".join(str((k // 4 ** j) % 4) for j in range(8))))
for mdl in MODELS:
    for fr in FRAMINGS:
        g = corpus[(corpus.short == mdl) & (corpus.framing == fr)]
        key = (g[ITEMS].to_numpy() * pw).sum(axis=1)
        tot = g.total.to_numpy()
        vcnt = pd.Series(key).value_counts(normalize=True)
        pat.append(dict(source=SHORT[mdl], framing=fr, n=len(g), distinct=len(vcnt), top1_share=vcnt.iloc[0],
                        top10_share=vcnt.iloc[:10].sum(), top50_share=vcnt.iloc[:50].sum(),
                        collision_unweighted=unbiased_collision(key), collision_weighted=np.nan,
                        effective_n_vectors=1 / unbiased_collision(key),
                        total_simpson_effective_n=1 / np.sum((np.bincount(tot, minlength=25) / len(g)) ** 2)))
        for rank, (k, v) in enumerate(vcnt.iloc[:5].items(), 1):
            topv.append(dict(source=SHORT[mdl], framing=fr, rank=rank, share=v,
                             vector="".join(str((k // 4 ** j) % 4) for j in range(8))))
        cc = cross_collision(key, g.profile_id.to_numpy(), tot)
        ts = sorted(t for t in cc if t in nh_cc)
        n_m = np.array([cc[t][1] for t in ts], float)
        c_m = np.array([cc[t][0] for t in ts])
        c_h = np.array([nh_cc[t][0] for t in ts])
        n_h = np.array([nh_cc[t][1] for t in ts], float)
        pi_m, pi_h = n_m / n_m.sum(), n_h / n_h.sum()
        K = lambda pi, c: float(np.sum(pi ** 2 * c))  # noqa: E731
        pat_part = K(pi_m, c_m) / K(pi_m, c_h)
        mix_part = K(pi_m, c_h) / K(pi_h, c_h)
        script_ratio = np.average(c_m, weights=n_m) / np.average(c_h, weights=n_m)
        
        dec2.append(dict(model=SHORT[mdl], framing=fr, reuse_ratio_script_weights=script_ratio,
                         pattern_part_ratio=pat_part, total_mix_part_ratio=mix_part,
                         overall_ratio=pat_part * mix_part,
                         n_totals=len(ts),
                         model_total_sd=float(np.std(tot)), nhanes_total_sd=float(np.std(tot_n_all))))
pd.DataFrame(pat).to_csv(os.path.join(OUT, "C_patterns.csv"), index=False)
pd.DataFrame(dec2).to_csv(os.path.join(OUT, "C_collision_decomposition.csv"), index=False)
pd.DataFrame(topv).to_csv(os.path.join(OUT, "C_top_vectors.csv"), index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
tt = pd.DataFrame(summ)
print(tt[(tt.quantity == "total")].round(3).to_string(index=False))
print(tt[(tt.weighting == "equal weight over 48 cells") & (tt.quantity != "total")].pivot(
    index="quantity", columns="model", values="between_share").round(3).to_string())
print(dec[(dec.source == "NHANES")].round(3).to_string(index=False))
print(pd.DataFrame(cov_rows).round(3).to_string(index=False))
print(pd.DataFrame(pat).round(3).to_string(index=False))
print(pd.DataFrame(dec2).round(3).to_string(index=False))
