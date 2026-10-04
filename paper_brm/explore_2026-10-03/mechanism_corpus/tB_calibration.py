"""Test B. Does recalibration fix the level-3 failure, and does it rescue level 2?

Unit: the 48 matched persona cells, framings combined (the level-3 primary unit). Calibration maps a
simulated cell mean s_c to  s_c + a  (additive offset) or  alpha + beta s_c  (linear), fitted by weighted
least squares of the NHANES cell mean y_c on s_c with the NHANES cell population counts N_c as weights,
so the offset is the model's overall post-stratified residual.

B1  Where the residual lives. Decomposition of the 48 cell residuals r_c = s_c - y_c (equal weight) into
    the uniform shift, the four attribute main effects and the remainder; the share of the N-weighted
    mean squared residual that is the squared overall residual.
B2  In-sample (oracle) calibration on all 48 cells: group residuals raw, after the offset, after the
    linear map. Passing in sample is optimistic by construction; the group spread that survives is not.
B3  Out of sample. 200 random splits of the 48 cells into halves, stratified by race x sex x income (one
    of the two marital cells of each triple goes to the calibration half, the other to the evaluation
    half, so every group keeps half its cells in each). The offset or the linear map is fitted on half A
    and level 3 is read on half B with the frozen estimator (post-stratified residual, simulation
    variance sum p^2 var, delete-one-PSU jackknife of the reference, TOST at the 90% level), with the
    calibration coefficients refitted inside every jackknife replicate so that reference error in the
    coefficients is carried. Simulation noise in the coefficients is approximated by the offset formula
    sum_A (N_c / sum_A N)^2 beta^2 var(s_c). Pass = all ten group intervals inside the tolerance.
B4  Leave-one-group-out: fit on the cells outside a level of an attribute, read the residual in the
    left-out level.
B5  Level 2 after linear recalibration. The gap of a persona pair scales by beta (the offset cancels), so
    the simulated gap and its SE are multiplied by beta and the frozen r3 (78c, Bonferroni over seven
    contrasts, asymmetric band) is rerun against the unchanged NHANES gap. Variants: beta from all 48
    cells (WLS and OLS), beta from the calibration half of each of the 200 splits, and the beta that
    sets the standardised Low minus High ratio to exactly 1 (the best case for a single slope).

Outputs: B_cell_residual_decomposition.csv, B_insample_groups.csv, B_halfsplit_summary.csv,
B_halfsplit_groups.csv, B_leave_group_out.csv, B_slopes.csv, B_l2_recalibrated.csv,
B_l2_split_summary.csv
"""
import os

import numpy as np
import pandas as pd

from common import (FULLNAME, GROUPS, MODELS, OUT, SHORT, TOLS, CellRef, attr, cis_matched, l2_model_verdict,
                    l2_reference, l2_sim_gaps, load_corpus, load_nhanes, r3_verdict, sim_cells, tost)

NSPLIT = 200
rng = np.random.default_rng(20261003)
nh = load_nhanes()
cref = CellRef(nh)
sims = sim_cells(cis_matched(load_corpus()))
UNITS = MODELS + ["pooled"]
ALL = np.ones(48, bool)
Nf = cref.Nf


def fit_coef(N, y, s, A, method):
    """alpha, beta for the calibrated value alpha + beta s; N, y are (48,) or (R, 48)."""
    if method == "none":
        z = np.zeros(N.shape[:-1] + (1,))
        return z, z + 1.0
    NA = N[..., A]
    W = NA.sum(-1, keepdims=True)
    sA, yA = s[A], y[..., A]
    if method == "offset":
        a = (NA * (yA - sA)).sum(-1, keepdims=True) / W
        return a, np.ones_like(a)
    sbar = (NA * sA).sum(-1, keepdims=True) / W
    ybar = (NA * yA).sum(-1, keepdims=True) / W
    beta = (NA * (sA - sbar) * (yA - ybar)).sum(-1, keepdims=True) / (NA * (sA - sbar) ** 2).sum(-1, keepdims=True)
    return ybar - beta * sbar, beta


def eval_groups(s, v, A, Bm, method, groups=GROUPS):
    """Level-3 rows on the cells Bm after calibration fitted on the cells A."""
    af, bf = fit_coef(cref.Nf, cref.mf, s, A, method)
    ar, br = fit_coef(cref.Nr, cref.mr, s, A, method)
    shat_f = (af + bf * s).ravel()
    shat_r = ar + br * s[None, :]
    beta = float(bf.ravel()[0])
    if method == "none":
        vcoef = 0.0
    else:
        wA = Nf[A] / Nf[A].sum()
        vcoef = beta ** 2 * float(np.sum(wA ** 2 * v[A]))
    rows = []
    for gname, dim, val in groups:
        g = np.ones(48, bool) if dim is None else attr(dim) == val
        ga = g & Bm
        if not ga.any():
            continue
        Ng, Ngr = cref.Nf[ga].sum(), cref.Nr[:, ga].sum(1)
        p, pr = cref.Nf[ga] / Ng, cref.Nr[:, ga] / Ngr[:, None]
        sim, simr = float(p @ shat_f[ga]), (pr * shat_r[:, ga]).sum(1)
        ref, refr = cref.Yf[ga].sum() / Ng, cref.Yr[:, ga].sum(1) / Ngr
        res, resr = sim - ref, simr - refr
        vref = float(cref.des.jk_var(res, resr))
        vsim = float(np.sum((p * beta) ** 2 * v[ga])) + vcoef
        se, df, tc, lo, hi = tost(res, vsim, vref, cref.df)
        r = dict(group=gname, n_cells=int(ga.sum()), resid=res, se=se, ci90_lo=lo, ci90_hi=hi)
        for k, tol in TOLS.items():
            r[f"pass_{k}"] = bool(lo > -tol and hi < tol)
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------- B1 decomposition
dec = []
levels = [["White", "Black", "Asian", "Hispanic"], ["Men", "Women"], ["Low", "Middle", "High"], ["Married", "Single"]]
names = ["race", "sex", "income", "marital"]
for u in UNITS:
    s = sims[(u, "combined")]["s"]
    r = s - cref.mf
    ss_unc = float(np.sum(r ** 2))
    gm = float(r.mean())
    parts = {"uniform shift (grand mean)": 48 * gm ** 2}
    ss_main = 0.0
    for k in range(4):
        a = attr(k)
        ss = sum((a == lv).sum() * (r[a == lv].mean() - gm) ** 2 for lv in levels[k])
        parts[f"{names[k]} main effect"] = float(ss)
        ss_main += ss
    parts["interaction and remainder"] = ss_unc - parts["uniform shift (grand mean)"] - ss_main
    p = Nf / Nf.sum()
    nw_share = float((p @ r) ** 2 / (p @ r ** 2))
    for k, vv in parts.items():
        dec.append(dict(model=SHORT.get(u, u), term=k, share_of_uncentered_SS=vv / ss_unc,
                        mean_cell_residual=gm, sd_cell_residual=float(r.std(ddof=1)),
                        nweighted_share_uniform=nw_share, cell_resid_min=float(r.min()),
                        cell_resid_max=float(r.max())))
pd.DataFrame(dec).to_csv(os.path.join(OUT, "B_cell_residual_decomposition.csv"), index=False)

# --------------------------------------------------------------------------------- B2 in sample
ins = []
slopes = []
for u in UNITS:
    s, v = sims[(u, "combined")]["s"], sims[(u, "combined")]["v"]
    for method in ("none", "offset", "linear"):
        rows = eval_groups(s, v, ALL, ALL, method)
        for r in rows:
            ins.append(dict(model=SHORT.get(u, u), calibration=method, **r))
    a, b = fit_coef(cref.Nf, cref.mf, s, ALL, "linear")
    b_ols = np.polyfit(s, cref.mf, 1)[0]
    slopes.append(dict(model=SHORT.get(u, u), alpha_wls=float(a.ravel()[0]), beta_wls_all48=float(b.ravel()[0]),
                       beta_ols_all48=float(b_ols), offset_all48=float(fit_coef(cref.Nf, cref.mf, s, ALL, "offset")[0].ravel()[0])))
ins = pd.DataFrame(ins)
ins.to_csv(os.path.join(OUT, "B_insample_groups.csv"), index=False)

# ------------------------------------------------------------------------------- B3 half splits
trip = np.arange(48) // 2
splits = []
for _ in range(NSPLIT):
    A = np.zeros(48, bool)
    for t in range(24):
        A[2 * t + int(rng.integers(0, 2))] = True
    splits.append(A)
grp_rows, sum_rows = [], []
beta_split = {u: [] for u in UNITS}
for u in UNITS:
    s, v = sims[(u, "combined")]["s"], sims[(u, "combined")]["v"]
    for method in ("none", "offset", "linear"):
        allpass = {k: [] for k in TOLS}
        maxabs = []
        per_group = {}
        for A in splits:
            Bm = ~A
            rows = eval_groups(s, v, A, Bm, method)
            maxabs.append(max(abs(r["resid"]) for r in rows))
            for k in TOLS:
                allpass[k].append(all(r[f"pass_{k}"] for r in rows))
            for r in rows:
                per_group.setdefault(r["group"], []).append(r)
            if method == "linear":
                beta_split[u].append(float(fit_coef(cref.Nf, cref.mf, s, A, "linear")[1].ravel()[0]))
        row = dict(model=SHORT.get(u, u), calibration=method, n_splits=NSPLIT,
                   mean_max_abs_group_resid=float(np.mean(maxabs)))
        for k in TOLS:
            row[f"share_splits_all10_pass_{k}"] = float(np.mean(allpass[k]))
        sum_rows.append(row)
        for gname, lst in per_group.items():
            g = dict(model=SHORT.get(u, u), calibration=method, group=gname,
                     mean_resid=float(np.mean([r["resid"] for r in lst])),
                     sd_resid_over_splits=float(np.std([r["resid"] for r in lst])),
                     mean_ci90_hi=float(np.mean([r["ci90_hi"] for r in lst])),
                     mean_ci90_lo=float(np.mean([r["ci90_lo"] for r in lst])))
            for k in TOLS:
                g[f"share_pass_{k}"] = float(np.mean([r[f"pass_{k}"] for r in lst]))
            grp_rows.append(g)
halfsum = pd.DataFrame(sum_rows)
halfsum.to_csv(os.path.join(OUT, "B_halfsplit_summary.csv"), index=False)
pd.DataFrame(grp_rows).to_csv(os.path.join(OUT, "B_halfsplit_groups.csv"), index=False)

# --------------------------------------------------------------------------- B4 leave group out
lo_rows = []
for u in UNITS:
    s, v = sims[(u, "combined")]["s"], sims[(u, "combined")]["v"]
    for k in range(4):
        a = attr(k)
        for lv in levels[k]:
            Bm = a == lv
            A = ~Bm
            for method in ("none", "offset", "linear"):
                r = eval_groups(s, v, A, Bm, method, groups=[(lv, k, lv)])[0]
                lo_rows.append(dict(model=SHORT.get(u, u), attribute=names[k], left_out=lv,
                                    calibration=method, **{kk: r[kk] for kk in
                                    ("resid", "ci90_lo", "ci90_hi", "pass_0.2SD", "pass_1pt", "pass_2pt")}))
pd.DataFrame(lo_rows).to_csv(os.path.join(OUT, "B_leave_group_out.csv"), index=False)

# ----------------------------------------------------------------------------------- B5 level 2
gaps = l2_sim_gaps()
rf = l2_reference()
contrasts = [c for (e, c) in rf.index if e == "standardised"]
LOWHIGH = "Low minus High SES"


def l2_all(mdl, beta, estimands=("standardised", "marginal")):
    out = []
    for est in estimands:
        for c in contrasts:
            g, se, df = gaps[(FULLNAME[mdl], c)]
            o = r3_verdict(beta * g, abs(beta) * se, df, rf.loc[(est, c)])
            out.append(dict(estimand=est, contrast=c, beta=beta, sim_gap=beta * g, ratio=o["ratio"],
                            ci_lo=o["ci_lo"], ci_hi=o["ci_hi"], ci_unbounded=o["ci_unbounded"],
                            verdict=o["verdict"], r3_unstopped=o["r3_unstopped"]))
    return out


l2rows = []
for mdl in MODELS:
    s = sims[(mdl, "combined")]["s"]
    b_wls = float(fit_coef(cref.Nf, cref.mf, s, ALL, "linear")[1].ravel()[0])
    b_ols = float(np.polyfit(s, cref.mf, 1)[0])
    g, se, df = gaps[(FULLNAME[mdl], LOWHIGH)]
    ratio_lh = r3_verdict(g, se, df, rf.loc[("standardised", LOWHIGH)])["ratio"]
    b_inc = 1.0 / ratio_lh
    for name, beta in [("raw (beta = 1)", 1.0), ("slope, all 48 cells, WLS", b_wls),
                       ("slope, all 48 cells, OLS", b_ols), ("slope that sets Low minus High to 1", b_inc)]:
        for r in l2_all(mdl, beta):
            l2rows.append(dict(model=SHORT[mdl], calibration=name, **r))
l2df = pd.DataFrame(l2rows)
l2df.to_csv(os.path.join(OUT, "B_l2_recalibrated.csv"), index=False)

# split-half slopes: verdict distribution
split_rows, model_v = [], []
for mdl in MODELS:
    s = sims[(mdl, "combined")]["s"]
    per = {}
    mv = []
    for A in splits:
        b = float(fit_coef(cref.Nf, cref.mf, s, A, "linear")[1].ravel()[0])
        rows = l2_all(mdl, b, estimands=("standardised",))
        for r in rows:
            per.setdefault(r["contrast"], []).append(r)
        mv.append(l2_model_verdict([r["verdict"] for r in rows]))
    for c, lst in per.items():
        vc = pd.Series([r["verdict"] for r in lst]).value_counts(normalize=True)
        split_rows.append(dict(model=SHORT[mdl], contrast=c,
                               mean_beta=float(np.mean([r["beta"] for r in lst])),
                               mean_ratio=float(np.mean([r["ratio"] for r in lst])),
                               ratio_p05=float(np.quantile([r["ratio"] for r in lst], .05)),
                               ratio_p95=float(np.quantile([r["ratio"] for r in lst], .95)),
                               verdict_shares="; ".join(f"{k}: {v:.2f}" for k, v in vc.items()),
                               share_kept_in_name=float(np.mean(["kept" in r["verdict"].split(" or ") for r in lst])),
                               share_excludes_kept=float(np.mean([r["verdict"] not in
                                                                  ("undetermined", "reference too imprecise",
                                                                   "no population gap") and
                                                                  "kept" not in r["verdict"].split(" or ") for r in lst]))))
    vv = pd.Series(mv).value_counts(normalize=True)
    model_v.append(dict(model=SHORT[mdl], contrast="MODEL VERDICT (std, 7 contrasts)",
                        mean_beta=float(np.mean([r["beta"] for r in per[LOWHIGH]])),
                        verdict_shares="; ".join(f"{k}: {v:.2f}" for k, v in vv.items())))
pd.DataFrame(split_rows + model_v).to_csv(os.path.join(OUT, "B_l2_split_summary.csv"), index=False)

# raw model verdicts for reference
raw_mv = {}
for mdl in MODELS:
    for name in l2df.calibration.unique():
        x = l2df[(l2df.model == SHORT[mdl]) & (l2df.calibration == name) & (l2df.estimand == "standardised")]
        raw_mv[(SHORT[mdl], name)] = l2_model_verdict(list(x.verdict))
slopes = pd.DataFrame(slopes)
slopes["beta_wls_halfsplit_mean"] = [float(np.mean(beta_split[u])) for u in UNITS]
slopes["beta_wls_halfsplit_sd"] = [float(np.std(beta_split[u])) for u in UNITS]
slopes.to_csv(os.path.join(OUT, "B_slopes.csv"), index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
print(pd.DataFrame(dec).pivot(index="term", columns="model", values="share_of_uncentered_SS").round(3).to_string())
print(pd.DataFrame(dec).groupby("model")[["mean_cell_residual", "sd_cell_residual", "nweighted_share_uniform"]].first().round(3).to_string())
print(slopes.round(3).to_string(index=False))
print(halfsum.round(3).to_string(index=False))
print("L2 model verdicts at all-cell slopes:")
for k, v in raw_mv.items():
    print(k, v)
x = l2df[(l2df.estimand == "standardised") & (l2df.contrast == LOWHIGH)]
print(x[["model", "calibration", "beta", "ratio", "ci_lo", "ci_hi", "verdict"]].round(3).to_string(index=False))
print(pd.DataFrame(split_rows + model_v)[lambda d: d.contrast.isin([LOWHIGH, "MODEL VERDICT (std, 7 contrasts)"])].round(3).to_string(index=False))
