"""Level 2: subgroup fidelity.

For each contrast (e.g. Women minus Men) the simulated gap g is the equal-weight mean, over strata
of the other attributes, of within-stratum persona-mean differences; the population gap gamma is
the reference contrast in the same strata with the same weights. rho = g / gamma is read with a
Fieller interval against the boundaries -0.25, 0.25, 0.75, 1.25 (reversed | missing | attenuated |
kept | steepened), Bonferroni over the model's family of contrasts, with two stops.

Ported from scripts/78c_l2_r3_verdicts.py (tcrit, sdf, r3: the frozen `verdict`),
scripts/78e_l2_external_r3.py (r3cov: the same rule with a covariance between g and gamma),
scripts/78b_l2_simulated_gaps.py (paired, mse: the simulated gap),
scripts/78a_l2_population_reference.py (standardised, Design.contrast: the population gap),
scripts/88_intermediate_panel.py (l2_model_verdict) and
paper_brm/external/scripts/l2_threeway.py (three_way).
"""
import itertools

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ._survey import taylor_contrast
from .thresholds import L2_ALPHA, L2_BOUNDS, L2_LABELS, L2_STOPS


# Source: scripts/78c_l2_r3_verdicts.py, tcrit
def _tcrit(a, df):
    df = float(df)
    return float(stats.t.ppf(1 - a, df if np.isfinite(df) and df > 0 else 1e9))


# Source: scripts/78c_l2_r3_verdicts.py, sdf
def _sdf(se_g, df_g, se_p, df_p, c):
    """Welch-Satterthwaite df for se_g^2 + c^2 se_p^2."""
    vg, vp = se_g ** 2, (c * se_p) ** 2
    if vp == 0:
        return df_g
    den = (vg ** 2 / df_g if np.isfinite(df_g) else 0) + (vp ** 2 / df_p if np.isfinite(df_p) else 0)
    return (vg + vp) ** 2 / den if den > 0 else np.inf


def _name_region(lo, hi, labels):
    # Source: scripts/78c_l2_r3_verdicts.py, r3, with the stop-2 bug fix of 3 Oct 2026
    # (scripts/_patch_stop2_2026-10-03.py): a reading that crosses more than one boundary and stays
    # below kept is "reversed to attenuated", which excludes kept, so stop 2 does not fire on it.
    if lo == hi:
        return labels[lo]
    if hi == lo + 1:
        return f"{labels[lo]} or {labels[hi]}"
    if hi < 3:
        return "reversed to attenuated"
    return "undetermined"


def _frozen(lab, p_lo, k, kmax):
    # Source: scripts/78c_l2_r3_verdicts.py, r3 (final_frozen). Stop 1: gamma's own interval covers
    # 0. Stop 2: k > kmax and the reading does not already exclude kept.
    if p_lo <= 0:
        return "no population gap"
    if k > kmax and (lab == "undetermined" or "kept" in lab.split(" or ")):
        return "reference too imprecise"
    return lab


# Source: scripts/78c_l2_r3_verdicts.py, r3 (df_mode "satterthwaite"; pre-freeze variants left out)
def _r3(g, se_g, df_g, p, se_p, df_p, a, bounds, labels):
    s = -1.0 if p < 0 else 1.0
    g, p = g * s, p * s
    tp = _tcrit(a, df_p)
    p_lo, p_hi = p - tp * se_p, p + tp * se_p

    def df_at(c):
        return _sdf(se_g, df_g, se_p, df_p, c)

    def stat(c):
        return (g - c * p) / np.sqrt(se_g ** 2 + (c * se_p) ** 2)

    above = sum(stat(c) > _tcrit(a, df_at(c)) for c in bounds)
    below = sum(stat(c) < -_tcrit(a, df_at(c)) for c in bounds)
    lo, hi = above, len(bounds) - below
    lab = _name_region(lo, hi, labels)
    grid = np.linspace(-20, 20, 8001)
    vg, vp = se_g ** 2, (grid * se_p) ** 2
    den = (vg ** 2 / df_g if np.isfinite(df_g) else 0) + (vp ** 2 / df_p if np.isfinite(df_p) else 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dfg = np.where(den > 0, (vg + vp) ** 2 / den, 1e9)
    tg = stats.t.ppf(1 - a, np.where(np.isfinite(dfg), dfg, 1e9))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.abs(stat(grid)) <= tg
    if acc.any():
        i0, i1 = np.argmax(acc), len(acc) - 1 - np.argmax(acc[::-1])

        def f(c):
            return abs(stat(c)) - _tcrit(a, df_at(c))

        ci_lo = grid[0] if i0 == 0 else optimize.brentq(f, grid[i0 - 1], grid[i0])
        ci_hi = grid[-1] if i1 == len(grid) - 1 else optimize.brentq(f, grid[i1], grid[i1 + 1])
        unbounded = bool(i0 == 0 or i1 == len(grid) - 1)
        split = bool(not acc[i0:i1 + 1].all())
    else:
        ci_lo = ci_hi = np.nan
        unbounded = split = False
    k = tp * se_p / p
    W = bounds[3] / bounds[2]
    kmax = (W - 1) / (W + 1)
    return dict(ratio=g / p, ci_lo=float(ci_lo), ci_hi=float(ci_hi), ci_unbounded=unbounded,
                ci_split=split, reading=lab,
                pop_ci_lo=s * p_lo if s > 0 else s * p_hi, pop_ci_hi=s * p_hi if s > 0 else s * p_lo,
                k_rel_halfwidth=k, k_max=kmax, no_population_gap=bool(p_lo <= 0),
                stop_population=bool(k > kmax), verdict=_frozen(lab, p_lo, k, kmax))


# Source: scripts/78e_l2_external_r3.py, r3cov (closed-form Fieller set; one df for g and gamma)
def _r3cov(g, se_g, p, se_p, cov, df, a, bounds, labels):
    s = -1.0 if p < 0 else 1.0
    g, p = g * s, p * s
    cov = cov * s * s
    t = float(stats.t.ppf(1 - a, df if np.isfinite(df) else 1e9))
    above = below = 0
    for c in bounds:
        z = (g - c * p) / np.sqrt(se_g ** 2 + c ** 2 * se_p ** 2 - 2 * c * cov)
        above += z > t
        below += z < -t
    lo, hi = above, len(bounds) - below
    lab = _name_region(lo, hi, labels)
    A = p ** 2 - t ** 2 * se_p ** 2
    Bq = g * p - t ** 2 * cov
    C = g ** 2 - t ** 2 * se_g ** 2
    disc = Bq ** 2 - A * C
    if A > 0 and disc >= 0:
        ci_lo, ci_hi = (Bq - np.sqrt(disc)) / A, (Bq + np.sqrt(disc)) / A
        unbounded = False
    else:
        ci_lo = ci_hi = np.nan
        unbounded = True
    p_lo, p_hi = p - t * se_p, p + t * se_p
    k = t * se_p / p
    W = bounds[3] / bounds[2]
    kmax = (W - 1) / (W + 1)
    return dict(ratio=g / p, ci_lo=float(ci_lo), ci_hi=float(ci_hi), ci_unbounded=unbounded,
                ci_split=False, reading=lab,
                pop_ci_lo=s * p_lo if s > 0 else s * p_hi, pop_ci_hi=s * p_hi if s > 0 else s * p_lo,
                k_rel_halfwidth=k, k_max=kmax, no_population_gap=bool(p_lo <= 0),
                stop_population=bool(k > kmax and not p_lo <= 0), verdict=_frozen(lab, p_lo, k, kmax))


# Source: paper_brm/external/scripts/l2_threeway.py, three_way (kept region from the bounds)
def three_way(verdict, ci_lo, ci_hi, *, bounds=L2_BOUNDS):
    """kept / not kept / unresolved / not read, from the frozen verdict and the ratio interval.

    'not read' when the contrast is stopped; otherwise the interval against the kept region
    [bounds[2], bounds[3]]: wholly inside -> kept, wholly outside -> not kept, crossing -> unresolved."""
    if verdict in L2_STOPS:
        return "not read"
    if not (np.isfinite(ci_lo) and np.isfinite(ci_hi)):
        return "unresolved"
    LO, HI = bounds[2], bounds[3]
    if ci_lo >= LO and ci_hi <= HI:
        return "kept"
    if ci_hi < LO or ci_lo > HI:
        return "not kept"
    return "unresolved"


def level2_contrast(g, se_g, gamma, se_gamma, *, df_g=np.inf, df_gamma=np.inf, cov=0.0,
                    family_size=1, alpha=L2_ALPHA, bounds=L2_BOUNDS, labels=L2_LABELS):
    """Read one contrast: simulated gap g against population gap gamma.

    Parameters
    ----------
    g, se_g, df_g : simulated gap, its SE and df (persona pairs - 1 for the paired design).
    gamma, se_gamma, df_gamma : population gap, its design-based SE and design df.
    cov : covariance of g and gamma when they come from the same sample (e.g. a bootstrap over
        the same respondents). With cov = 0 the interval is the numerical inversion of the boundary
        tests with Welch-Satterthwaite df (78c); with cov != 0, or an exact simulated gap
        (se_g = 0, where the inversion has no interval), it is the closed-form Fieller set at a
        single df, df_gamma if df_g == df_gamma, else the smaller of the two (78e, 93).
    family_size : number of contrasts in the model's family; each one-sided test is run at
        alpha / family_size (Bonferroni; 7 contrasts give 98.6% intervals).

    Returns
    -------
    dict: ratio, ci_lo, ci_hi, reading (region reading before the stops), verdict (frozen rule,
    including the stops "no population gap" and "reference too imprecise"), label (kept / not
    kept / unresolved / not read), k_rel_halfwidth and the stop flags.

    Reading
    -------
    "kept" supports the claim that the simulator reproduces the population's gap at its size
    (rho between 0.75 and 1.25). A verdict that excludes kept (reversed, missing, attenuated,
    steepened, or a pair of these) rules out reproduction of that gap. A stop says the reference
    cannot support a verdict: no population gap to reproduce, or a reference too imprecise to
    certify kept.
    """
    a = alpha / family_size
    bounds = np.asarray(bounds, float)
    # numpy floats: an exact zero population gap gives an infinite ratio and k (stop 1 then
    # reads it as "no population gap") instead of a ZeroDivisionError on Python floats
    g, se_g, gamma, se_gamma, cov = (np.float64(x) for x in (g, se_g, gamma, se_gamma, cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        if cov == 0 and se_g > 0:
            out = _r3(g, se_g, df_g, gamma, se_gamma, df_gamma, a, bounds, labels)
            out["df_rule"] = "satterthwaite"
        else:
            df = df_gamma if df_g == df_gamma else min(df_g, df_gamma)
            out = _r3cov(g, se_g, gamma, se_gamma, cov, df, a, bounds, labels)
            out["df_rule"] = "single df"
    out.update(g=g, se_g=se_g, df_g=df_g, gamma=gamma, se_gamma=se_gamma, df_gamma=df_gamma, cov=cov,
               family_size=family_size, alpha_one_sided=a, interval_level=1 - 2 * a)
    out["label"] = three_way(out["verdict"], out["ci_lo"], out["ci_hi"], bounds=bounds)
    out["certifiable"] = certifiable(gamma, se_gamma, df_gamma=df_gamma, family_size=family_size, alpha=alpha,
                                     bounds=bounds, labels=labels)
    return out


def certifiable(gamma, se_gamma, *, df_gamma=np.inf, family_size=1, alpha=L2_ALPHA, bounds=L2_BOUNDS,
                labels=L2_LABELS):
    """Step 0 (certifiability): can this reference certify the contrast at all?

    True when a noise-free faithful simulator (g = gamma exactly, se_g = 0) would be read 'kept' by
    the frozen rule at this family size. False when the reference stops it (no population gap, or
    a reference too imprecise) or leaves even that simulator's interval outside the kept region.
    No simulator, however good, can earn 'kept' on a contrast that is not certifiable, so a model
    family with no certifiable contrast cannot pass whatever its answers."""
    a = alpha / family_size
    bounds = np.asarray(bounds, float)
    gamma, se_gamma = np.float64(gamma), np.float64(se_gamma)
    if not (np.isfinite(gamma) and np.isfinite(se_gamma)) or gamma == 0:
        return False
    with np.errstate(divide="ignore", invalid="ignore"):
        r = _r3cov(gamma, np.float64(0.0), gamma, se_gamma, np.float64(0.0), df_gamma, a, bounds, labels)
    return bool(r["verdict"] == "kept")


# Source: scripts/88_intermediate_panel.py, l2_model_verdict
def level2_model_verdict(verdicts):
    """pass: every contrast that is not stopped is 'kept'; fail: any verdict excludes 'kept';
    otherwise unresolved (including a family in which every contrast is stopped)."""
    live = [v for v in verdicts if v not in L2_STOPS]
    if any(v != "undetermined" and "kept" not in v.split(" or ") for v in live):
        return "fail"
    if live and all(v == "kept" for v in live):
        return "pass"
    return "unresolved"


def level2_model_reading(verdicts, certifiable_flags):
    """The model verdict with step 0 made explicit: pass and fail as level2_model_verdict; an
    unresolved family splits into 'reference cannot certify' (no contrast in the family is
    certifiable, so no simulator could pass against this reference) and 'unresolved' (the reference
    could certify at least one contrast and the simulator's readings are too wide)."""
    v = level2_model_verdict(verdicts)
    if v == "unresolved" and not any(bool(c) for c in certifiable_flags):
        return "reference cannot certify"
    return v


def faithful_simulation(gamma, se_gamma, se_g, *, df_gamma=np.inf, df_g=np.inf, family_size=None, n_rep=1000,
                        seed=0, alpha=L2_ALPHA, bounds=L2_BOUNDS):
    """Planning: how a faithful simulator would be read against this reference at this design.

    gamma, se_gamma : the reference gaps and SEs of one family (arrays). se_g : the SE each simulated
    gap would have at the planned design (draws, personas). Each replicate draws g ~ N(gamma, se_g)
    independently of the reference and reads the family by the frozen rule.
    Returns dict(kept_share per contrast, label shares, family verdict shares, certifiable flags)."""
    gamma, se_gamma, se_g = (np.atleast_1d(np.asarray(x, float)) for x in (gamma, se_gamma, se_g))
    m = len(gamma) if family_size is None else family_size
    rng = np.random.default_rng(seed)
    labels = np.empty((n_rep, len(gamma)), dtype=object)
    fam = []
    for r in range(n_rep):
        g = gamma + rng.standard_normal(len(gamma)) * se_g
        outs = [level2_contrast(g[j], se_g[j], gamma[j], se_gamma[j], df_g=df_g, df_gamma=df_gamma,
                                family_size=m, alpha=alpha, bounds=bounds) for j in range(len(gamma))]
        labels[r] = [o["label"] for o in outs]
        fam.append(level2_model_verdict([o["verdict"] for o in outs]))
    lab_names = ("kept", "not kept", "unresolved", "not read")
    return dict(
        kept_share=(labels == "kept").mean(0),
        label_shares={k: float((labels == k).mean()) for k in lab_names},
        family_shares={k: float(np.mean([f == k for f in fam])) for k in ("pass", "fail", "unresolved")},
        certifiable=[certifiable(gamma[j], se_gamma[j], df_gamma=df_gamma, family_size=m, alpha=alpha,
                                 bounds=bounds) for j in range(len(gamma))])


# Source: scripts/78b_l2_simulated_gaps.py, paired / mse (generalised to several personas per
# stratum side, and to a design without other attributes)
def simulated_gap(sim, attribute, hi, lo, *, score="score", persona="persona", strata=()):
    """Simulated gap and its persona-pair SE.

    Persona means are formed over draws. With strata (the other persona attributes), each stratum
    that holds both levels contributes the mean of its `hi` persona means minus the mean of its
    `lo` persona means, and g is the equal-weight mean over strata with SE = SD of the stratum
    differences / sqrt(strata), df = strata - 1. Without strata, g is the difference of the two
    groups' persona means with the Welch SE and df.

    Returns dict(g, se, df, n_pairs, strata (DataFrame of the strata used and their differences)).
    """
    strata = list(strata)
    cols = [persona, attribute] + strata
    pm = sim.groupby(cols, observed=True, dropna=False)[score].mean().reset_index()
    if pm[persona].duplicated().any():
        raise ValueError("persona attributes must be constant within a persona")
    A = pm[pm[attribute] == hi]
    Bp = pm[pm[attribute] == lo]
    if not len(A) or not len(Bp):
        raise ValueError(f"no simulated personas with {attribute} = {hi!r} and {lo!r}")
    if strata:
        ga = A.groupby(strata, observed=True)[score].agg(["mean", "var", "size"])
        gb = Bp.groupby(strata, observed=True)[score].agg(["mean", "var", "size"])
        common = ga.index.intersection(gb.index)
        ga, gb = ga.loc[common], gb.loc[common]
        d = ga["mean"] - gb["mean"]
        n = len(d)
        g = float(d.mean()) if n else np.nan
        st = d.rename("diff").reset_index()
        if n and (ga["size"] == 1).all() and (gb["size"] == 1).all():
            # paired design (78b): one persona per stratum side; SE from the pair differences
            se = float(d.std(ddof=1) / np.sqrt(n)) if n >= 2 else np.nan
            return dict(g=g, se=se, df=float(n - 1), n_pairs=int(n), se_rule="pairs", strata=st)
        # Several personas per stratum side (a package generalisation, not in the scripts): the
        # variance of each stratum difference is s2_hi / n_hi + s2_lo / n_lo over persona means,
        # with a side of one persona taking the pooled within-side variance; df by
        # Welch-Satterthwaite over the components.
        sides = pd.concat([ga, gb])
        multi = sides["size"] >= 2
        pooled = float(((sides["var"] * (sides["size"] - 1))[multi].sum() / (sides["size"] - 1)[multi].sum())
                       if multi.any() else np.nan)
        comp = []
        for s in (ga, gb):
            v = np.where(s["size"] >= 2, s["var"].fillna(pooled), pooled) / s["size"]
            dfc = np.where(s["size"] >= 2, s["size"] - 1, (sides["size"] - 1)[multi].sum())
            comp.append((v / n ** 2, dfc))
        v_all = np.concatenate([c[0] for c in comp])
        df_all = np.concatenate([c[1] for c in comp]).astype(float)
        var = float(v_all.sum())
        df = float(var ** 2 / np.sum(v_all ** 2 / df_all)) if var > 0 else np.nan
        return dict(g=g, se=float(np.sqrt(var)), df=df, n_pairs=int(n), se_rule="stratified persona means",
                    strata=st)
    xa, xb = A[score].to_numpy(float), Bp[score].to_numpy(float)
    va, vb = xa.var(ddof=1) / len(xa), xb.var(ddof=1) / len(xb)
    df = (va + vb) ** 2 / (va ** 2 / (len(xa) - 1) + vb ** 2 / (len(xb) - 1))
    return dict(g=float(xa.mean() - xb.mean()), se=float(np.sqrt(va + vb)), df=float(df),
                n_pairs=0, strata=None)


# Source: scripts/78a_l2_population_reference.py, build_specs.standardised / marginal and
# Design.contrast (Taylor SE)
def population_gap(ref, attribute, hi, lo, *, score="score", strata=(), cells=None, weight=None,
                   psu=None, stratum=None):
    """Population gap in the reference, standardised to the given strata with equal weights.

    cells : DataFrame of the strata to standardise over (one row per stratum, columns = strata);
        default: every combination of the strata columns present in the reference. Without strata
        the gap is the marginal weighted mean difference.
    weight, psu, stratum : column names in `ref` (or None) for the survey design.

    Returns dict(gamma, se, df, n_cells, min_cell_n).
    """
    strata = list(strata)
    y = ref[score].to_numpy(float)
    w = np.ones(len(ref)) if weight is None else ref[weight].to_numpy(float)
    psu_v = None if psu is None else ref[psu].to_numpy()
    st_v = None if stratum is None else ref[stratum].to_numpy()
    av = ref[attribute].to_numpy()
    masks, coef = [], []
    if strata:
        if cells is None:
            cells = ref[strata].drop_duplicates()
        S = len(cells)
        for row in cells.itertuples(index=False):
            cond = np.ones(len(ref), bool)
            for c, v in zip(strata, row):
                cond &= (ref[c] == v).to_numpy()
            masks += [cond & (av == hi), cond & (av == lo)]
            coef += [1.0 / S, -1.0 / S]
    else:
        masks = [av == hi, av == lo]
        coef = [1.0, -1.0]
    r = taylor_contrast(y, w, masks, coef, psu=psu_v, stratum=st_v)
    return dict(gamma=r["estimate"], se=r["se"], df=float(r["df"]), n_cells=len(masks),
                min_cell_n=int(r["domain_n"].min()))


def _norm_contrasts(contrasts):
    out = []
    for c in contrasts:
        if isinstance(c, dict):
            attr, hi, lo = c["attribute"], c["hi"], c["lo"]
            name = c.get("name", f"{hi} minus {lo}")
        else:
            attr, hi, lo = c[:3]
            name = c[3] if len(c) > 3 else f"{hi} minus {lo}"
        out.append((name, attr, hi, lo))
    return out


def default_contrasts(sim, ref, attributes):
    """Every pair of levels of every attribute that both the personas and the reference carry,
    first-appearance order in the simulated data (hi = the later level)."""
    out = []
    for attr in attributes:
        levels = [v for v in pd.unique(sim[attr]) if v in set(ref[attr].unique())]
        for x, y in itertools.combinations(levels, 2):
            out.append((attr, y, x))
    return out


def level2(sim, ref, contrasts=None, *, attributes=None, score="score", persona="persona",
           ref_score=None, weight=None, psu=None, stratum=None, standardise=True, family_size=None,
           alpha=L2_ALPHA, bounds=L2_BOUNDS):
    """Level 2 for one model and framing.

    Parameters
    ----------
    sim : long DataFrame, one row per draw: persona, score and persona attribute columns.
    ref : reference DataFrame, one row per respondent: score, the same attribute columns with the
        same level labels, and optionally weight / psu / stratum columns.
    contrasts : list of (attribute, hi, lo) or (attribute, hi, lo, name) tuples or dicts with keys
        attribute, hi, lo, name. Default: every pair of levels of every attribute (prefer naming
        the contrasts: the family size sets the Bonferroni level).
    attributes : persona attributes the reference records. Default: columns shared by sim and ref
        other than persona, score and the design columns.
    standardise : True (frozen estimand) standardises both gaps over the strata of the other
        attributes; False gives the marginal gaps.
    family_size : default len(contrasts).

    Strata of the other attributes that are empty in the reference on either side are dropped from
    both gaps (n_strata_dropped records it).

    Returns
    -------
    dict(contrasts=DataFrame one row per contrast (with `certifiable`, step 0), verdict=pass / fail /
    unresolved, reading=verdict with 'reference cannot certify' split out of unresolved, n_certifiable,
    family_size).

    Reading
    -------
    A pass (every contrast that is not stopped is kept) supports using the simulated population for
    comparisons between the groups tested, at the size the population shows. A fail (any contrast
    whose verdict excludes kept) rules out reading the simulated gaps as the population's gaps.
    """
    ref_score = ref_score or score
    design_cols = {persona, score, ref_score, weight, psu, stratum} - {None}
    if attributes is None:
        attributes = [c for c in sim.columns if c in ref.columns and c not in design_cols]
    attributes = list(attributes)
    if contrasts is None:
        contrasts = default_contrasts(sim, ref, attributes)
    cons = _norm_contrasts(contrasts)
    if not cons:
        raise ValueError("no contrasts to read")
    m = len(cons) if family_size is None else int(family_size)
    rows = []
    for name, attr, hi, lo in cons:
        strata = [a for a in attributes if a != attr] if standardise else []
        sg = simulated_gap(sim, attr, hi, lo, score=score, persona=persona, strata=strata)
        cells, dropped = None, 0
        if strata:
            st = sg["strata"]
            keep = []
            for row in st[strata].itertuples(index=False):
                cond = np.ones(len(ref), bool)
                for c, v in zip(strata, row):
                    cond &= (ref[c] == v).to_numpy()
                keep.append(bool((cond & (ref[attr] == hi).to_numpy()).any()
                                 and (cond & (ref[attr] == lo).to_numpy()).any()))
            keep = np.array(keep, bool)
            dropped = int((~keep).sum())
            cells = st.loc[keep, strata]
            if dropped and len(cells):
                # re-estimate the simulated gap on the strata the reference can match
                kidx = pd.MultiIndex.from_frame(cells)
                sub = sim[pd.MultiIndex.from_frame(sim[strata]).isin(kidx)]
                sg = simulated_gap(sub, attr, hi, lo, score=score, persona=persona, strata=strata)
        base = dict(contrast=name, attribute=attr, hi=hi, lo=lo, estimand="standardised" if strata else "marginal",
                    n_pairs=sg["n_pairs"], n_strata_dropped=dropped)
        if not np.isfinite(sg["g"]) or not np.isfinite(sg["se"]) or (cells is not None and not len(cells)):
            rows.append(dict(base, verdict="not read", label="not read",
                             reason="too few persona pairs or reference strata to estimate the gap"))
            continue
        pg = population_gap(ref, attr, hi, lo, score=ref_score, strata=strata, cells=cells, weight=weight,
                            psu=psu, stratum=stratum)
        r = level2_contrast(sg["g"], sg["se"], pg["gamma"], pg["se"], df_g=sg["df"], df_gamma=pg["df"],
                            family_size=m, alpha=alpha, bounds=bounds)
        rows.append(dict(base, min_ref_cell_n=pg["min_cell_n"], **r))
    res = pd.DataFrame(rows)
    if "certifiable" not in res:
        res["certifiable"] = False
    res["certifiable"] = res["certifiable"].fillna(False).astype(bool)
    live = [v for v in res.verdict if v != "not read"]
    verdict = level2_model_verdict(live)
    reading = level2_model_reading(live, res["certifiable"])
    return dict(contrasts=res, verdict=verdict, reading=reading, n_certifiable=int(res["certifiable"].sum()),
                family_size=m, interval_level=1 - 2 * alpha / m)
