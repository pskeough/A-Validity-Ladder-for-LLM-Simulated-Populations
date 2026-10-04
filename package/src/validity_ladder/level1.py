"""Level 1: individual coherence.

Each simulated answer vector is scored with lz* (WLE theta, Snijders correction) under a graded
response model fitted to the reference, and placed in the reference distribution of lz* among
respondents with the same total (randomised PIT). The misfit ratio is the share below the
within-total 5th percentile over .05, the overfit ratio the share above the 95th over .05.

Ported from scripts/91_ladder_core.py (L1Ref, l1_eval, tau_from_subgroups, _resolved, TAUS,
TAU_FLOOR), scripts/79_l1_lib.py (verdict) and scripts/79c_l1_personfit.py (read_verdict).
"""
import numpy as np
import pandas as pd

from ._grm import GRM, TailRef, tail_probs, total_strata
from ._survey import design_codes, raowu_multipliers
from .thresholds import (INTERVAL_LEVEL, L1_MIN_GROUP_N, L1_MIN_STRATUM_N, L1_N_BOOT, L1_TAIL_Q,
                         L1_TAU_FLOOR, L1_TAUS)


class L1Reference:
    """The population reference for level 1: GRM, reference lz*, within-total strata and B
    bootstrap reweightings of the reference.

    Parameters
    ----------
    items : (n, J) integer array of reference answers scored 0..M-1 after reverse keying.
    weights : survey weights (None for an unweighted sample). The GRM is fitted with them and the
        within-total CDF is weighted by them.
    psu, stratum : sample design. With them the reference replicates are Rao-Wu bootstrap
        replicates over PSUs within strata; without them, an iid person bootstrap.
    n_categories : M. Default: the largest answer + 1.
    n_boot : number of reference replicates (paired one-to-one with the persona bootstrap).
    grm_params : optional (a, b) to skip the fit (b has shape (J, M - 1)).
    """

    def __init__(self, items, weights=None, psu=None, stratum=None, n_categories=None,
                 n_boot=L1_N_BOOT, seed=0, min_stratum_n=L1_MIN_STRATUM_N, grm_params=None):
        X = np.asarray(items)
        if X.ndim != 2:
            raise ValueError("items must be a 2-D array (respondents x items)")
        X = X.astype(int)
        n, J = X.shape
        M = int(n_categories) if n_categories is not None else int(X.max()) + 1
        if X.min() < 0 or X.max() > M - 1:
            raise ValueError(f"answers must be scored 0..{M - 1}")
        self.J, self.M, self.n = J, M, n
        self.grm = GRM(J, M)
        if grm_params is None:
            fit = self.grm.fit(X, weights)
            self.a, self.b, self.fit = fit["a"], fit["b"], fit
        else:
            self.a, self.b = (np.asarray(p, float) for p in grm_params)
            self.fit = None
        _, _, lzs = self.grm.score(self.a, self.b, X)
        self.lzstar = lzs
        self.total = X.sum(1)
        self.max_total = J * (M - 1)
        self.smap = total_strata(self.total, self.max_total, min_stratum_n)
        self.cond = TailRef(self.smap[self.total], lzs)
        self.w = np.ones(n) if weights is None else np.asarray(weights, float)
        self.B = int(n_boot)
        self.seed = seed
        if psu is not None or stratum is not None:
            self.psu_idx, s_of_psu = design_codes(n, psu, stratum)
            self.mult = raowu_multipliers(s_of_psu, self.B, np.random.default_rng([seed, 1]))
        else:
            self.psu_idx, self.mult = None, None

    def wrep(self, b):
        """Reference weights of replicate b (None: the full-sample weights)."""
        if b is None:
            return self.w
        if self.mult is not None:
            return self.mult[b, self.psu_idx] * self.w
        rng = np.random.default_rng([self.seed, 2, b])
        return np.bincount(rng.integers(0, self.n, self.n), minlength=self.n) * self.w

    def lzstar_of(self, items):
        X = np.asarray(items).astype(int)
        if X.ndim != 2 or X.shape[1] != self.J:
            raise ValueError(f"simulated items must have {self.J} columns")
        if X.min() < 0 or X.max() > self.M - 1:
            raise ValueError(f"simulated answers must be scored 0..{self.M - 1}")
        _, _, lzs = self.grm.score(self.a, self.b, X)
        return X.sum(1), lzs


# Source: scripts/79_l1_lib.py, verdict
def _interval_reading(lo, hi, tol):
    L, U = 1.0 / tol, tol
    if lo >= L and hi <= U:
        return "equivalent"
    if hi < L:
        return "below"
    if lo > U:
        return "above"
    if lo < L <= hi <= U:
        return "below or equivalent"
    if L <= lo <= U < hi:
        return "equivalent or above"
    return "undetermined"


# Source: scripts/79c_l1_personfit.py, read_verdict
def read_verdict(r, tau):
    """Each tail read against [1/tau, tau]; the overall reading names the pattern of the tails.

    Returns (misfit reading, overfit reading, overall): pass when both tails are equivalent;
    'fail: compressed (both tails in deficit)' when both are below; otherwise the single-tail
    reading; 'undetermined' when no tail is decided."""
    vm = _interval_reading(r["misfit_ratio_ci90_lo"], r["misfit_ratio_ci90_hi"], tau)
    vo = _interval_reading(r["overfit_ratio_ci90_lo"], r["overfit_ratio_ci90_hi"], tau)
    vm = vm.replace("below", "misfit deficit").replace("above", "excess misfit")
    vo = vo.replace("below", "overfit deficit").replace("above", "excess overfit")
    if vm == "equivalent" and vo == "equivalent":
        return vm, vo, "pass"
    dec = {x for x in [vm, vo] if " or " not in x and x not in ("equivalent", "undetermined")}
    if dec == {"misfit deficit", "overfit deficit"}:
        overall = "fail: compressed (both tails in deficit)"
    elif dec:
        overall = "fail: " + " and ".join(sorted(dec))
    else:
        overall = "undetermined"
    return vm, vo, overall


# Source: scripts/91_ladder_core.py, _resolved
def resolved_departure(lo, hi):
    """Fold departure from 1 that a 90% interval [lo, hi] of a ratio establishes: lo when the
    interval lies above 1, 1/hi when it lies below 1, and 1 when it covers 1."""
    return lo if lo > 1 else (1 / hi if hi < 1 else 1.0)


def _group_masks(groups, n, min_n):
    if isinstance(groups, dict) and all(np.asarray(v).dtype == bool for v in groups.values()):
        return {k: np.asarray(v, bool) for k, v in groups.items()}
    df = pd.DataFrame(groups)
    if len(df) != n:
        raise ValueError("reference subgroup attributes must have one row per reference respondent")
    out = {}
    for col in df.columns:
        for lv, cnt in df[col].value_counts().items():
            if cnt >= min_n:
                out[f"{col}={lv}"] = (df[col] == lv).to_numpy()
    return out


# Source: scripts/91_ladder_core.py, tau_from_subgroups (taus and floor as arguments; groups may
# be given as attribute columns)
def tau_from_subgroups(reference, groups, *, taus=L1_TAUS, tau_floor=L1_TAU_FLOOR,
                       min_group_n=L1_MIN_GROUP_N, n_boot=None, ci=INTERVAL_LEVEL):
    """Tolerance tau for level 1 from the reference's own subgroups.

    Each subgroup (every demographic attribute the reference records, levels with n >= 100) is
    scored against the pooled reference; its resolved departure is read on the 90% interval of each
    ratio. tau is the smallest value in `taus`, at least `tau_floor`, that contains the worst
    resolved departure (inf when none does).

    groups : dict name -> boolean mask over reference rows, or a DataFrame / dict of attribute
        columns (one row per reference respondent).
    Returns (tau, worst resolved departure, DataFrame of subgroup rows).
    """
    ref = reference
    B = ref.B if n_boot is None else n_boot
    q = L1_TAIL_Q
    a = (1 - ci) / 2
    masks = _group_masks(groups, ref.n, min_group_n)
    loc = ref.cond.locate(ref.smap[ref.total], ref.lzstar)

    def ratios(w):
        Fm, F = ref.cond.cdf(loc, w)
        cl, ch, _ = tail_probs(Fm, F, q)
        return {k: (np.average(cl[m], weights=w[m]) / q, np.average(ch[m], weights=w[m]) / q)
                for k, m in masks.items()}

    pt = ratios(ref.w)
    reps = [ratios(ref.wrep(b)) for b in range(B)]
    rows, worst = [], 1.0
    for name, m in masks.items():
        rm, ro = pt[name]
        bm = np.array([r[name][0] for r in reps])
        bo = np.array([r[name][1] for r in reps])
        mlo, mhi = np.quantile(bm, a), np.quantile(bm, 1 - a)
        olo, ohi = np.quantile(bo, a), np.quantile(bo, 1 - a)
        dev = max(resolved_departure(mlo, mhi), resolved_departure(olo, ohi))
        worst = max(worst, dev)
        rows.append(dict(group=name, n=int(m.sum()), misfit_ratio=rm, misfit_ci90_lo=mlo,
                         misfit_ci90_hi=mhi, overfit_ratio=ro, overfit_ci90_lo=olo,
                         overfit_ci90_hi=ohi, point_departure=max(rm, 1 / rm, ro, 1 / ro),
                         resolved_departure=dev))
    tau = next((t for t in sorted(taus) if worst <= t and t >= tau_floor), np.inf)
    return tau, worst, pd.DataFrame(rows)


def tau_from_intervals(misfit_lo, misfit_hi, overfit_lo, overfit_hi, *, taus=L1_TAUS,
                       tau_floor=L1_TAU_FLOOR):
    """tau from subgroup ratio intervals that were computed elsewhere (arrays, one per subgroup)."""
    dev = [max(resolved_departure(a, b), resolved_departure(c, d))
           for a, b, c, d in zip(misfit_lo, misfit_hi, overfit_lo, overfit_hi)]
    worst = max([1.0] + dev)
    return next((t for t in sorted(taus) if worst <= t and t >= tau_floor), np.inf), worst


def level1(sim_items, persona, reference, *, ref_weights=None, ref_psu=None, ref_stratum=None,
           n_categories=None, tau=None, ref_groups=None, taus=L1_TAUS, tau_floor=L1_TAU_FLOOR,
           n_boot=None, ci=INTERVAL_LEVEL, seed=0):
    """Level 1 for one model and framing: are the simulated answer vectors as coherent as people's
    at the same total?

    Parameters
    ----------
    sim_items : (n_draws, J) answers scored 0..M-1 after reverse keying.
    persona : persona label of every draw (the bootstrap resamples personas).
    reference : an L1Reference, or the (n, J) reference answers (then ref_weights, ref_psu,
        ref_stratum and n_categories build one; reuse an L1Reference across models to fit the GRM
        once).
    tau : tolerance for the ratios. Default: from `ref_groups` by the paper's rule
        (tau_from_subgroups); without ref_groups, the floor 1.5.
    ref_groups : reference subgroup attributes for tau (see tau_from_subgroups).
    n_boot : persona bootstrap replicates (default: the reference's B, paired one-to-one).

    Returns
    -------
    dict with misfit and overfit ratios and their 90% intervals, the reading of each tail, tau,
    `verdict_detail` (79c wording) and `verdict` (pass / fail / unresolved).

    Reading
    -------
    A pass (both ratios' intervals inside [1/tau, tau]) supports treating each simulated answer
    vector as an internally coherent response of the kind people give at that total. A fail rules
    out that reading: too many incoherent vectors (excess misfit), too few atypical vectors
    (misfit deficit), too many overly regular vectors (excess overfit), or vectors that avoid both
    ends of the human distribution ("compressed").
    """
    if not isinstance(reference, L1Reference):
        reference = L1Reference(reference, weights=ref_weights, psu=ref_psu, stratum=ref_stratum,
                                n_categories=n_categories, n_boot=L1_N_BOOT if n_boot is None else n_boot,
                                seed=seed)
    ref = reference
    B = ref.B if n_boot is None else int(n_boot)
    if B > ref.B:
        raise ValueError(f"n_boot ({B}) exceeds the reference's replicates ({ref.B})")
    q = L1_TAIL_Q
    a = (1 - ci) / 2
    total, lz = ref.lzstar_of(sim_items)
    tau_source, worst, tau_rows = "given", np.nan, None
    if tau is None:
        if ref_groups is not None:
            tau, worst, tau_rows = tau_from_subgroups(ref, ref_groups, taus=taus, tau_floor=tau_floor,
                                                      n_boot=B, ci=ci)
            tau_source = "reference subgroups"
        else:
            tau, tau_source = tau_floor, "floor (no reference subgroups given)"
    # Source: scripts/91_ladder_core.py, l1_eval
    _, pidx = np.unique(np.asarray(persona), return_inverse=True)
    loc = ref.cond.locate(ref.smap[np.asarray(total, int)], lz)
    Fm, F = ref.cond.cdf(loc, ref.w)
    cl, ch, mid = tail_probs(Fm, F, q)
    npers = pidx.max() + 1
    cnt = np.bincount(pidx, minlength=npers).astype(float)
    rng = np.random.default_rng([seed, 3])
    ridx = rng.integers(0, npers, size=(B, npers))
    bl, bh = np.empty(B), np.empty(B)
    for b in range(B):
        Fm_b, F_b = ref.cond.cdf(loc, ref.wrep(b))
        lb, hb, _ = tail_probs(Fm_b, F_b, q)
        sl = np.bincount(pidx, weights=lb, minlength=npers)
        sh = np.bincount(pidx, weights=hb, minlength=npers)
        den = cnt[ridx[b]].sum()
        bl[b], bh[b] = sl[ridx[b]].sum() / den, sh[ridx[b]].sum() / den
    r = dict(n_draws=len(total), n_personas=int(npers), mean_mid_pit=float(mid.mean()),
             misfit_ratio=cl.mean() / q, misfit_ratio_ci90_lo=np.quantile(bl, a) / q,
             misfit_ratio_ci90_hi=np.quantile(bl, 1 - a) / q, overfit_ratio=ch.mean() / q,
             overfit_ratio_ci90_lo=np.quantile(bh, a) / q, overfit_ratio_ci90_hi=np.quantile(bh, 1 - a) / q)
    if np.isfinite(tau):
        vm, vo, overall = read_verdict(r, tau)
    else:
        # no candidate tau contains the reference's own subgroup departures: no tolerance, no reading
        vm = vo = "not read"
        overall = "undetermined: no tau in the candidate set contains the reference subgroups' departures"
    r.update(misfit_reading=vm, overfit_reading=vo, verdict_detail=overall, tau=tau,
             tau_source=tau_source, worst_resolved_departure=worst, tau_subgroups=tau_rows,
             verdict="pass" if overall == "pass" else ("fail" if overall.startswith("fail") else "unresolved"))
    return r
