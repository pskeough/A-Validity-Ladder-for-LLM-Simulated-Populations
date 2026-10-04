"""Level 3: population calibration.

The post-stratified residual of a group G is R_G = sum_c p_c (s_c - y_c), where c runs over the
persona cells (attribute combinations) in G, s_c is the simulated cell value, y_c the reference
weighted mean of the cell and p_c the reference weighted share of cell c within G. The simulation
SE is sqrt(sum p_c^2 var(s_c)); the reference SE is the delete-one-PSU jackknife of the whole
residual, weights included; df by Satterthwaite with the design df. TOST at alpha .05: the 90%
interval must lie inside (-delta, +delta) for every group row.

Ported from scripts/80c_l3_estimates.py (estimand "PS", tost, jk_moments, fieller),
scripts/80_l3_lib.py (Design: here _survey.JKnDesign), scripts/80b_l3_sim_cells.py (plus-four
variance of a prevalence) and scripts/83_controls_lib.py (l3_eval, l3_pass).
"""
import numpy as np
import pandas as pd
from scipy import stats

from ._survey import JKnDesign
from .thresholds import L3_ALPHA, L3_PREV_BANDS, L3_RATIO_BAND, L3_REPORT_SD, L3_TOLERANCE_SD


def read_tost(lo, hi, tolerance):
    """pass: interval inside (-tol, tol); fail: interval wholly outside it; else unresolved."""
    if lo > -tolerance and hi < tolerance:
        return "pass"
    if lo > tolerance or hi < -tolerance:
        return "fail"
    return "unresolved"


# Source: scripts/80c_l3_estimates.py, tost
def level3_tost(resid, var_sim, var_ref, df_ref, tolerance, *, alpha=L3_ALPHA):
    """TOST of one residual. Returns dict(se, df, tcrit, ci90_lo, ci90_hi, delta_min, verdict).

    df = var^2 / (var_ref^2 / df_ref) (Satterthwaite, simulation side with infinite df)."""
    var = var_sim + var_ref
    se = float(np.sqrt(var))
    df = var ** 2 / (var_ref ** 2 / df_ref) if var_ref > 0 else 1e9
    tc = float(stats.t.ppf(1 - alpha, df))
    lo, hi = resid - tc * se, resid + tc * se
    return dict(se=se, df=float(df), tcrit=tc, ci90_lo=float(lo), ci90_hi=float(hi),
                delta_min=float(max(abs(lo), abs(hi))), verdict=read_tost(lo, hi, tolerance))


# Source: scripts/80c_l3_estimates.py, jk_moments
def _jk_moments(des, a, ar, b, br):
    c = (des.rep_nh - 1.0) / des.rep_nh
    da, db = ar - a, br - b
    return float(np.sum(c * da * da)), float(np.sum(c * da * db)), float(np.sum(c * db * db))


# Source: scripts/80c_l3_estimates.py, fieller (band as an argument)
def prevalence_ratio(sim, ref, vsim, mom, df, *, band=L3_RATIO_BAND, alpha=L3_ALPHA):
    """90% Fieller interval of sim / ref and the two one-sided tests of the ratio band, run as
    linear contrasts sim - b ref; var(sim - b ref) = vsim + A - 2 b B + b^2 C with (A, B, C) the
    jackknife moments of the composition-weighted simulated value and the reference."""
    A, B, C = mom
    t = float(stats.t.ppf(1 - alpha, df))
    qa = ref ** 2 - t ** 2 * C
    qb = -2.0 * (sim * ref - t ** 2 * B)
    qc = sim ** 2 - t ** 2 * (vsim + A)
    disc = qb ** 2 - 4 * qa * qc
    if qa > 0 and disc >= 0:
        lo, hi = (-qb - np.sqrt(disc)) / (2 * qa), (-qb + np.sqrt(disc)) / (2 * qa)
        lo = max(lo, 0.0)
    else:
        lo, hi = np.nan, np.nan
    out = dict(ratio=sim / ref if ref else np.nan, ratio_ci90_lo=lo, ratio_ci90_hi=hi)
    for b, side in [(band[0], "lo"), (band[1], "hi")]:
        se_b = np.sqrt(vsim + A - 2 * b * B + b * b * C)
        out[f"ratio_z_{side}"] = (sim - b * ref) / se_b
    out["ratio_pass"] = bool(out["ratio_z_lo"] > t and out["ratio_z_hi"] < -t)
    return out


def level3_from_cells(cell_values, cell_variances, ref_cell, ref_y, ref_weight, groups, *,
                      tolerance, psu=None, stratum=None, alpha=L3_ALPHA, ratio_band=None):
    """Post-stratified residuals for given cell values (the core of level 3).

    cell_values, cell_variances : (C,) simulated cell values s_c and var(s_c); NaN = no personas.
    ref_cell : (n,) index 0..C-1 of each reference respondent's cell, -1 when unclassified.
    ref_y, ref_weight : (n,) reference outcome and weight.
    groups : dict name -> (C,) boolean mask of the cells in the group.
    psu, stratum : (n,) design arrays (None: iid respondents, delete-one jackknife).
    ratio_band : when given (prevalence rows), adds the Fieller ratio test.

    Returns a DataFrame with one row per group.
    """
    s = np.asarray(cell_values, float)
    v = np.asarray(cell_variances, float)
    C = len(s)
    ci = np.asarray(ref_cell, int)
    y = np.asarray(ref_y, float)
    w = np.asarray(ref_weight, float)
    des = JKnDesign(len(y), psu, stratum)
    ok = ci >= 0
    onehot = np.zeros((len(y), C))
    onehot[np.where(ok)[0], ci[ok]] = 1.0
    ncell = onehot[ok].sum(0)
    Nf, Nr = des.replicate(des.psu_totals(onehot * w[:, None]))
    Yf, Yr = des.replicate(des.psu_totals(onehot * (w * np.nan_to_num(y))[:, None]))
    avail = ~np.isnan(s)
    rows = []
    for gname, g in groups.items():
        g = np.asarray(g, bool)
        ga = g & avail & (Nf > 0)
        base = dict(group=gname, n_cells=int(g.sum()), n_cells_missing=int((g & ~avail).sum()),
                    n_cells_empty_in_reference=int((g & avail & ~(Nf > 0)).sum()),
                    n_ref=int(ncell[ga].sum()))
        if not ga.any():
            rows.append(dict(base, verdict="not read", reason="no cell with both simulated and reference data"))
            continue
        # Source: scripts/80c_l3_estimates.py, PS block
        Ng, Ngr = Nf[ga].sum(), Nr[:, ga].sum(1)
        p, pr = Nf[ga] / Ng, Nr[:, ga] / Ngr[:, None]
        simps, simps_r = float(p @ s[ga]), pr @ s[ga]
        ref, refr = Yf[ga].sum() / Ng, Yr[:, ga].sum(1) / Ngr
        res, resr = simps - ref, simps_r - refr
        vref = float(des.jk_var(res, resr))
        vsim = float(np.sum(p ** 2 * v[ga]))
        t = level3_tost(res, vsim, vref, des.dfree, tolerance, alpha=alpha)
        r = dict(base, sim=simps, ref=float(ref), ref_se=float(np.sqrt(des.jk_var(ref, refr))),
                 resid=float(res), se_sim=float(np.sqrt(vsim)), se_ref_part=float(np.sqrt(vref)),
                 design_df=des.dfree, tolerance=tolerance, **t)
        if ratio_band is not None:
            r.update(prevalence_ratio(simps, ref, vsim, _jk_moments(des, simps, simps_r, ref, refr),
                                      t["df"], band=ratio_band, alpha=alpha))
        rows.append(r)
    return pd.DataFrame(rows)


def level3_model_verdict(verdicts):
    """pass when every group row passes, fail when any fails, else unresolved."""
    verdicts = [v for v in verdicts if v != "not read"]
    if not verdicts:
        return "unresolved"
    if any(v == "fail" for v in verdicts):
        return "fail"
    if all(v == "pass" for v in verdicts):
        return "pass"
    return "unresolved"


def _persona_cells(sim, persona, score, cell_attributes, prevalence_cut=None):
    """Per persona: mean and var(mean) = s2 / n (pooled within variance when n = 1)."""
    cols = [persona] + list(cell_attributes)
    x = sim[score].astype(float)
    if prevalence_cut is not None:
        x = (x >= prevalence_cut).astype(float)
    d = sim[cols].assign(_x=x.to_numpy())
    pm = d.groupby(cols, observed=True, dropna=False)._x.agg(["size", "mean", "var", "sum"]).reset_index()
    if pm[persona].duplicated().any():
        raise ValueError("persona attributes must be constant within a persona")
    if prevalence_cut is not None:
        # Source: scripts/80b_l3_sim_cells.py: plus-four, p~ = (x + 2) / (n + 4)
        pt = (pm["sum"] + 2) / (pm["size"] + 4)
        pm["v"] = pt * (1 - pt) / (pm["size"] + 4)
    else:
        multi = pm["size"] >= 2
        pooled = ((pm["var"] * (pm["size"] - 1))[multi].sum() / (pm["size"] - 1)[multi].sum()
                  if multi.any() else 0.0)
        pm["v"] = np.where(multi, pm["var"].fillna(0) / pm["size"], pooled / pm["size"])
    return pm


def level3(sim, ref, cell_attributes, *, score="score", persona="persona", ref_score=None,
           weight=None, psu=None, stratum=None, group_attributes=None, sd_ref=None, tolerance=None,
           tolerance_sd=L3_TOLERANCE_SD, report_sd=L3_REPORT_SD, prevalence_cut=None,
           prevalence_bands=L3_PREV_BANDS, ratio_band=L3_RATIO_BAND, alpha=L3_ALPHA):
    """Level 3 for one model and framing: does the simulated population reproduce the reference's
    mean, overall and in each group, after post-stratification to the reference's composition?

    Parameters
    ----------
    sim : long DataFrame, one row per draw (persona, score, cell attribute columns).
    ref : reference DataFrame, one row per respondent (score, the same cell attribute columns with
        the same labels, optional weight / psu / stratum columns).
    cell_attributes : the persona attributes that define a post-stratification cell.
    group_attributes : attributes whose levels form the group rows (default: cell_attributes);
        an "Overall" row is always included.
    tolerance : delta in score points. Default: tolerance_sd x sd_ref (0.5 SD, the frozen default);
        sd_ref defaults to the weighted SD of the reference score. The use sets the tolerance.
    report_sd : extra tolerances (SD units) whose pass flags are reported beside the verdict.
    prevalence_cut : when given, a prevalence table (share at or above the cut, in percentage
        points) is reported with the difference bands and the ratio band. It is reported beside the
        verdict and does not enter it.

    Simulated cell value: the mean of the cell's persona means; var(s_c) = sum over its personas of
    s2_p / n_p, over (personas in the cell)^2 (draw sampling only; personas are fixed by design).

    Returns
    -------
    dict(groups=DataFrame, verdict, tolerance, sd_ref, prevalence=DataFrame or None).

    Reading
    -------
    A pass (every group's 90% interval of the residual inside +/- delta) supports using the
    simulated population's means, overall and per group, as estimates of the reference's to within
    delta. A fail (an interval wholly outside the band) rules out that use at that tolerance for
    that group.
    """
    ref_score = ref_score or score
    cell_attributes = list(cell_attributes)
    group_attributes = cell_attributes if group_attributes is None else list(group_attributes)
    w = np.ones(len(ref)) if weight is None else ref[weight].to_numpy(float)
    yref = ref[ref_score].to_numpy(float)
    if sd_ref is None:
        mu = np.average(yref, weights=w)
        sd_ref = float(np.sqrt(np.average((yref - mu) ** 2, weights=w) * len(yref) / (len(yref) - 1)))
    if tolerance is None:
        tolerance = tolerance_sd * sd_ref
    psu_v = None if psu is None else ref[psu].to_numpy()
    st_v = None if stratum is None else ref[stratum].to_numpy()

    def run(cut):
        pm = _persona_cells(sim, persona, score, cell_attributes, cut)
        cells = pm.groupby(cell_attributes, observed=True, dropna=False).agg(
            s=("mean", "mean"), vsum=("v", "sum"), n_personas=("mean", "size")).reset_index()
        cells["v"] = cells.vsum / cells.n_personas ** 2
        key = pd.MultiIndex.from_frame(cells[cell_attributes])
        rkey = pd.MultiIndex.from_frame(ref[cell_attributes])
        ref_cell = key.get_indexer(rkey)
        groups = {"Overall": np.ones(len(cells), bool)}
        for a in group_attributes:
            for lv in pd.unique(cells[a]):
                groups[f"{a}={lv}"] = (cells[a] == lv).to_numpy()
        yy = yref if cut is None else (yref >= cut).astype(float) * 100.0
        s = cells.s.to_numpy(float) * (1.0 if cut is None else 100.0)
        v = cells.v.to_numpy(float) * (1.0 if cut is None else 1e4)
        return level3_from_cells(s, v, ref_cell, yy, w, groups, tolerance=tolerance if cut is None else prevalence_bands[0],
                                 psu=psu_v, stratum=st_v, alpha=alpha,
                                 ratio_band=None if cut is None else ratio_band)

    tab = run(None)
    for c in ("resid", "ci90_lo", "ci90_hi"):
        if c not in tab:
            tab[c] = np.nan
    for x in report_sd:
        tab[f"pass_{x}sd"] = [bool(read_tost(lo, hi, x * sd_ref) == "pass") if np.isfinite(lo) else np.nan
                              for lo, hi in zip(tab.ci90_lo, tab.ci90_hi)]
    tab["resid_sd_units"] = tab.resid / sd_ref
    prev = None
    if prevalence_cut is not None:
        prev = run(prevalence_cut)
        for c in ("ci90_lo", "ci90_hi"):
            if c not in prev:
                prev[c] = np.nan
        for b in prevalence_bands:
            prev[f"verdict_{b}pp"] = [read_tost(lo, hi, b) if np.isfinite(lo) else "not read"
                                      for lo, hi in zip(prev.ci90_lo, prev.ci90_hi)]
        prev = prev.drop(columns=["verdict", "tolerance"], errors="ignore")
    return dict(groups=tab, verdict=level3_model_verdict(list(tab.verdict)), tolerance=tolerance,
                sd_ref=sd_ref, prevalence=prev)
