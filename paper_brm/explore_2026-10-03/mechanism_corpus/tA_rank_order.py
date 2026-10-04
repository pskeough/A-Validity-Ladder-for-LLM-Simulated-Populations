"""Test A. Rank-order fidelity of the persona cell means.

Unit: the 48 matched persona cells (race x sex x income x marital). For each model and framing the
simulated cell mean s_c is correlated with the NHANES 2005-2018 weighted cell mean y_c (Spearman and
Pearson), with a percentile bootstrap over cells (B = 1000, 90% limits).

Benchmark with real respondents (the 83_controls_lib logic): 200 random 50/50 splits of NHANES adults,
stratified by cell. The reference half supplies y_c. Three things are correlated with it:
  donor half     the donor half's own weighted cell means (split-half reliability)
  pseudo-model   30 draws per cell and framing from the donor half, draws proportional to weight, two
                 framings pooled (the 83b pseudo-model, combined framing)
  each model     the model's real cell means (so models and people face the same noisy reference)
The Spearman-Brown correction of the split-half Pearson r gives the reliability of the full-sample
NHANES cell means, and its square root is the largest correlation any noise-free simulator can
reach with them (the noise ceiling).

Also: the same correlations within income band (cell means centred on their income band), the
marginal level means for each attribute, and each attribute's main-effect share of the variance of the
48 cell means.

Outputs: A_rank_order.csv, A_realpeople_benchmark.csv, A_split_reps.csv, A_attribute_effects.csv,
A_main_effect_shares.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import (CELLS, FRAMINGS, MODELS, OUT, SHORT, CellRef, attr, cis_matched, load_corpus,
                    load_nhanes, pearson, sim_cells, spearman)

B = 1000
NSPLIT = 200
rng = np.random.default_rng(20261003)

nh = load_nhanes()
cref = CellRef(nh)
y = cref.mf.copy()
sims = sim_cells(cis_matched(load_corpus()))
inc = attr(2)


def center_within(v, g):
    out = v.copy()
    for lv in np.unique(g):
        out[g == lv] -= v[g == lv].mean()
    return out


def corr_pair(a, b):
    return spearman(a, b), pearson(a, b)


def boot_ci(a, b, B, rng):
    n = len(a)
    sp, pe = np.empty(B), np.empty(B)
    for k in range(B):
        i = rng.integers(0, n, n)
        sp[k] = spearman(a[i], b[i])
        pe[k] = pearson(a[i], b[i])
    q = lambda v: (float(np.nanquantile(v, 0.05)), float(np.nanquantile(v, 0.95)))  # noqa: E731
    return q(sp), q(pe)


# ------------------------------------------------------------------ models against NHANES
rows = []
for mdl in MODELS + ["pooled"]:
    for fr in FRAMINGS + ["combined"]:
        s = sims[(mdl, fr)]["s"]
        r = dict(model=SHORT.get(mdl, mdl), framing=fr)
        sp, pe = corr_pair(s, y)
        (sl, sh), (pl, ph) = boot_ci(s, y, B, rng)
        r.update(spearman=sp, spearman_lo90=sl, spearman_hi90=sh, pearson=pe, pearson_lo90=pl,
                 pearson_hi90=ph)
        sc, yc = center_within(s, inc), center_within(y, inc)
        sp2, pe2 = corr_pair(sc, yc)
        (sl, sh), (pl, ph) = boot_ci(sc, yc, B, rng)
        r.update(within_income_spearman=sp2, within_income_spearman_lo90=sl,
                 within_income_spearman_hi90=sh, within_income_pearson=pe2,
                 within_income_pearson_lo90=pl, within_income_pearson_hi90=ph,
                 sd_sim_cells=s.std(ddof=1), sd_nhanes_cells=y.std(ddof=1),
                 sd_ratio=s.std(ddof=1) / y.std(ddof=1),
                 ols_slope_y_on_s=np.polyfit(s, y, 1)[0], mean_resid=float(np.mean(s - y)))
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "A_rank_order.csv"), index=False)

# ------------------------------------------------------------------ real-respondent benchmark
ok = nh.c48.to_numpy() >= 0
d = nh[ok].reset_index(drop=True)
cell = d.c48.to_numpy()
w = d.w.to_numpy(float)
tot = d.phq8.to_numpy(float)
idx_cell = [np.flatnonzero(cell == c) for c in range(48)]


def split_half(rng):
    donor = np.zeros(len(d), bool)
    for ix in idx_cell:
        p = rng.permutation(ix)
        k = (len(p) + int(rng.integers(0, 2))) // 2
        donor[p[:k]] = True
    return donor


def wmeans(mask):
    num = np.bincount(cell[mask], weights=(w * tot)[mask], minlength=48)
    den = np.bincount(cell[mask], weights=w[mask], minlength=48)
    return num / den


reps = []
for r in range(NSPLIT):
    donor = split_half(rng)
    mref, mdon = wmeans(~donor), wmeans(donor)
    # pseudo-model: 30 draws per cell and framing from the donor half, pooled over two framings
    ps = np.empty(48)
    for c in range(48):
        pool = idx_cell[c][donor[idx_cell[c]]]
        pr = w[pool] / w[pool].sum()
        draws = rng.choice(pool, 60, replace=True, p=pr)
        ps[c] = tot[draws].mean()
    row = dict(rep=r)
    for name, v in [("donor_half", mdon), ("pseudo_model", ps)] + \
            [(SHORT.get(m, m), sims[(m, "combined")]["s"]) for m in MODELS + ["pooled"]]:
        sp, pe = corr_pair(v, mref)
        sp2, pe2 = corr_pair(center_within(v, inc), center_within(mref, inc))
        row.update({f"{name}|spearman": sp, f"{name}|pearson": pe,
                    f"{name}|within_income_spearman": sp2, f"{name}|within_income_pearson": pe2})
    reps.append(row)
reps = pd.DataFrame(reps)
reps.to_csv(os.path.join(OUT, "A_split_reps.csv"), index=False)

bench = []
for name in ["donor_half", "pseudo_model"] + [SHORT.get(m, m) for m in MODELS + ["pooled"]]:
    for stat in ["spearman", "pearson", "within_income_spearman", "within_income_pearson"]:
        v = reps[f"{name}|{stat}"]
        bench.append(dict(source=name, statistic=stat, mean=v.mean(), p05=v.quantile(.05),
                          p95=v.quantile(.95)))
bench = pd.DataFrame(bench)
r_half = reps["donor_half|pearson"].mean()
rel_full = 2 * r_half / (1 + r_half)
inc_rank = np.select([inc == "Low", inc == "Middle"], [2.0, 1.0], 0.0)     # income-only predictor
base_sp, base_pe = corr_pair(inc_rank, y)
base_sp_h = float(np.mean([spearman(inc_rank, wmeans(~split_half(rng))) for _ in range(NSPLIT)]))
bench = pd.concat([bench, pd.DataFrame([
    dict(source="income-only predictor (Low 2, Middle 1, High 0) vs full NHANES", statistic="spearman",
         mean=base_sp, p05=np.nan, p95=np.nan),
    dict(source="income-only predictor vs full NHANES", statistic="pearson", mean=base_pe,
         p05=np.nan, p95=np.nan),
    dict(source="income-only predictor vs reference half (mean of 200 splits)", statistic="spearman",
         mean=base_sp_h, p05=np.nan, p95=np.nan),
    dict(source="noise ceiling vs reference half (sqrt of split-half r)", statistic="pearson",
         mean=float(np.sqrt(r_half)), p05=np.nan, p95=np.nan),
    dict(source="noise ceiling", statistic="pearson", mean=float(np.sqrt(rel_full)), p05=np.nan, p95=np.nan),
    dict(source="reliability of full-sample NHANES cell means (Spearman-Brown)", statistic="pearson",
         mean=rel_full, p05=np.nan, p95=np.nan)])], ignore_index=True)
bench.to_csv(os.path.join(OUT, "A_realpeople_benchmark.csv"), index=False)

# ------------------------------------------------------------------ attribute effects
names = ["race", "sex", "income", "marital"]
levels = [["White", "Black", "Asian", "Hispanic"], ["Men", "Women"], ["Low", "Middle", "High"],
          ["Married", "Single"]]
eff = []
shares = []
for src, v in [("NHANES", y)] + [(SHORT.get(m, m), sims[(m, "combined")]["s"]) for m in MODELS + ["pooled"]]:
    tot_ss = np.sum((v - v.mean()) ** 2)
    ss_main = 0.0
    for k in range(4):
        a = attr(k)
        lm = {lv: v[a == lv].mean() for lv in levels[k]}
        for lv in levels[k]:
            eff.append(dict(source=src, attribute=names[k], level=lv, marginal_mean=lm[lv],
                            minus_grand=lm[lv] - v.mean()))
        ss = sum((a == lv).sum() * (lm[lv] - v.mean()) ** 2 for lv in levels[k])
        ss_main += ss
        shares.append(dict(source=src, term=names[k], share_of_cell_mean_variance=ss / tot_ss))
    shares.append(dict(source=src, term="all four main effects", share_of_cell_mean_variance=ss_main / tot_ss))
    shares.append(dict(source=src, term="interaction + residual", share_of_cell_mean_variance=1 - ss_main / tot_ss))
    shares.append(dict(source=src, term="total SS of 48 cell means", share_of_cell_mean_variance=tot_ss))
pd.DataFrame(eff).to_csv(os.path.join(OUT, "A_attribute_effects.csv"), index=False)
pd.DataFrame(shares).to_csv(os.path.join(OUT, "A_main_effect_shares.csv"), index=False)

pd.set_option("display.width", 250)
print(res[["model", "framing", "spearman", "spearman_lo90", "spearman_hi90", "pearson", "pearson_lo90",
           "pearson_hi90", "within_income_spearman", "within_income_pearson", "sd_ratio",
           "ols_slope_y_on_s"]].round(3).to_string(index=False))
print(bench.round(3).to_string(index=False))
print(pd.DataFrame(shares).pivot(index="term", columns="source", values="share_of_cell_mean_variance").round(3).to_string())
