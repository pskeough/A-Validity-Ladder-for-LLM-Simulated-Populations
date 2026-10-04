"""Level 4: structural fidelity (R1, R2, R3; R4 is taken as an input).

All draws of one model and framing form the unit; persona is the resampling unit. Two-step
polychoric correlations and a one-factor ULS fit give the loadings.
  R1  general factor: every loading >= .30 and lambda1 / lambda2 >= 3, each threshold lowered to the
      reference's own value where the reference is below it.
  R2  pass: Tucker congruence with the reference loadings has lower 90% limit >= .95 AND the RMS
      loading difference has upper 90% limit <= .10; fail: congruence upper limit < .95 OR RMSD
      lower limit > .10; else unresolved. Persona bootstrap paired with the reference bootstrap.
  R3  diagnostic: one-factor fit to the pooled within-persona correlation matrix.
  R4  multi-group invariance (metric and scalar steps per attribute). Not implemented here; pass
      its verdict through `r4`.

Ported from scripts/91_ladder_core.py (Tables, _thresholds, _loglik_grid, poly_from,
one_factor_uls, eig_desc, r2_verdict, general_factor, general_factor_vs_ref, congruence, boot_mult,
within_R, L4Ref, l4_eval) and scripts/81_l4_lib.py (cluster_boot_mult, stratified branch).
"""
import numpy as np
from scipy import optimize, stats

from .thresholds import INTERVAL_LEVEL, L4_N_BOOT, R1_LOAD, R1_RATIO, R2_PHI, R2_RMSD


# Source: scripts/91_ladder_core.py, class Tables
class Tables:
    """Weighted pairwise M x M tables per cluster."""

    def __init__(self, X, w, cluster, M):
        X = np.asarray(X, int)
        w = np.asarray(w, float)
        self.M, self.J = M, X.shape[1]
        self.pairs = [(i, j) for i in range(self.J) for j in range(i + 1, self.J)]
        self.ids, cidx = np.unique(np.asarray(cluster), return_inverse=True)
        C = len(self.ids)
        self.C = C
        MM = M * M
        tab = np.zeros((C, len(self.pairs), MM))
        for k, (i, j) in enumerate(self.pairs):
            code = X[:, i] * M + X[:, j]
            tab[:, k, :] = np.bincount(cidx * MM + code, weights=w, minlength=C * MM).reshape(C, MM)
        self.tab = tab
        self.cidx = cidx

    def agg(self, mult=None):
        mult = np.ones(self.C) if mult is None else mult
        return np.tensordot(mult, self.tab, 1)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)


# Source: scripts/91_ladder_core.py, _thresholds
def _thresholds(tabs):
    n = tabs.sum(axis=(1, 2))
    cr = np.cumsum(tabs.sum(2), 1)[:, :-1] / n[:, None]
    cc = np.cumsum(tabs.sum(1), 1)[:, :-1] / n[:, None]
    ta = stats.norm.ppf(np.clip(cr, 1e-8, 1 - 1e-8))
    tb = stats.norm.ppf(np.clip(cc, 1e-8, 1 - 1e-8))
    pad = np.full((len(tabs), 1), 8.0)
    return np.hstack([-pad, ta, pad]), np.hstack([-pad, tb, pad])


# Source: scripts/91_ladder_core.py, _loglik_grid
def _loglik_grid(tabs, A, B, rg):
    h = A[:, None, :, None]
    k = B[:, None, None, :]
    base = stats.norm.cdf(h) * stats.norm.cdf(k)
    K, G = rg.shape
    M1 = A.shape[1]
    out = np.zeros((K, G, M1, M1))
    for x, wt in zip(_GL_X, _GL_W):
        t = (x * rg / 2 + rg / 2)[:, :, None, None]
        q = 1 - t ** 2
        out += (wt * rg / 2)[:, :, None, None] * np.exp(-(h ** 2 - 2 * t * h * k + k ** 2) / (2 * q)) / np.sqrt(q)
    Pm = base + out / (2 * np.pi)
    cell = Pm[:, :, 1:, 1:] - Pm[:, :, :-1, 1:] - Pm[:, :, 1:, :-1] + Pm[:, :, :-1, :-1]
    cell = np.clip(cell, 1e-12, 1.0)
    return (tabs[:, None, :, :] * np.log(cell)).sum(axis=(2, 3))


# Source: scripts/91_ladder_core.py, poly_from
def poly_from(tab, J, M, chunk=60):
    """All J(J-1)/2 polychorics: coarse grid, fine grid, parabolic step."""
    pairs = [(i, j) for i in range(J) for j in range(i + 1, J)]
    tabs_all = tab.reshape(len(pairs), M, M)
    r = np.empty(len(pairs))
    for s0 in range(0, len(pairs), chunk):
        tabs = tabs_all[s0:s0 + chunk]
        K = len(tabs)
        A, B = _thresholds(tabs)
        g1 = np.tile(np.linspace(-0.98, 0.98, 50), (K, 1))
        ll = _loglik_grid(tabs, A, B, g1)
        r0 = g1[np.arange(K), ll.argmax(1)]
        g2 = np.clip(r0[:, None] + np.linspace(-0.04, 0.04, 41)[None, :], -0.995, 0.995)
        ll2 = _loglik_grid(tabs, A, B, g2)
        j = np.clip(ll2.argmax(1), 1, 39)
        idx = np.arange(K)
        y0, y1, y2 = ll2[idx, j - 1], ll2[idx, j], ll2[idx, j + 1]
        den = y0 - 2 * y1 + y2
        step = g2[idx, 1] - g2[idx, 0]
        off = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0)
        r[s0:s0 + K] = g2[idx, j] + np.clip(off, -1, 1) * step
    R = np.eye(J)
    for kk, (i, jj) in enumerate(pairs):
        R[i, jj] = R[jj, i] = r[kk]
    return R


def polychoric_matrix(items, weights=None, n_categories=None):
    """Two-step polychoric correlation matrix of (n, J) ordinal answers."""
    X = np.asarray(items, int)
    M = int(n_categories) if n_categories is not None else int(X.max()) + 1
    T = Tables(X, np.ones(len(X)) if weights is None else weights, np.zeros(len(X), int), M)
    return poly_from(T.agg(), X.shape[1], M)


# Source: scripts/91_ladder_core.py, one_factor_uls
def one_factor_uls(R):
    """One-factor unweighted least squares loadings (sign: positive sum)."""
    P = R.shape[0]
    iu = np.triu_indices(P, 1)
    r = R[iu]

    def f(lam):
        e = r - np.outer(lam, lam)[iu]
        E = np.zeros((P, P))
        E[iu] = e
        E = E + E.T
        return float((e ** 2).sum()), -2 * E @ lam

    vals, vecs = np.linalg.eigh(R)
    start = vecs[:, -1] * np.sqrt(max(vals[-1] - 1, 0.1))
    best = None
    for s0 in (start, np.full(P, 0.6), -start):
        res = optimize.minimize(f, s0, jac=True, method="L-BFGS-B", bounds=[(-0.999, 0.999)] * P)
        if best is None or res.fun < best.fun:
            best = res
    lam = best.x
    if lam.sum() < 0:
        lam = -lam
    return lam


# Source: scripts/91_ladder_core.py, eig_desc
def eig_desc(R):
    return np.sort(np.linalg.eigvalsh(R))[::-1]


# Source: scripts/91_ladder_core.py, congruence
def congruence(a, b):
    """Tucker's congruence coefficient."""
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


# Source: scripts/91_ladder_core.py, r2_verdict (thresholds as arguments)
def r2_verdict(phi_lo, phi_hi, rmsd_lo, rmsd_hi, *, r2_phi=R2_PHI, r2_rmsd=R2_RMSD):
    """pass: congruence lower limit >= r2_phi and RMSD upper limit <= r2_rmsd; fail: congruence
    upper limit < r2_phi or RMSD lower limit > r2_rmsd; otherwise unresolved."""
    if phi_lo >= r2_phi and rmsd_hi <= r2_rmsd:
        return "pass"
    if phi_hi < r2_phi or rmsd_lo > r2_rmsd:
        return "fail"
    return "unresolved"


# Source: scripts/91_ladder_core.py, general_factor
def general_factor(lam, ev, *, min_load=R1_LOAD, min_ratio=R1_RATIO):
    """Absolute R1: every loading >= min_load and lambda1 / lambda2 >= min_ratio."""
    return bool((np.asarray(lam) >= min_load).all() and ev[0] / ev[1] >= min_ratio)


# Source: scripts/91_ladder_core.py, general_factor_vs_ref (thresholds as arguments)
def general_factor_vs_ref(lam, ev, ref_lam, ref_ev, *, min_load=R1_LOAD, min_ratio=R1_RATIO):
    """Frozen R1: the absolute thresholds, lowered to the reference's own value where the reference
    falls below them (item by item for the loadings)."""
    floor = np.minimum(min_load, np.asarray(ref_lam))
    return bool((np.asarray(lam) >= floor).all() and ev[0] / ev[1] >= min(min_ratio, ref_ev[0] / ref_ev[1]))


# Source: scripts/81_l4_lib.py, cluster_boot_mult
def cluster_boot_mult(C, rng, strata=None):
    """One cluster bootstrap replicate: iid clusters, or Rao-Wu within strata when strata given."""
    if strata is None:
        return np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
    mult = np.zeros(C)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        nh = len(idx)
        if nh < 2:
            mult[idx] += 1.0
            continue
        pick = rng.choice(idx, nh - 1, replace=True)
        np.add.at(mult, pick, nh / (nh - 1))
    return mult


# Source: scripts/91_ladder_core.py, within_R
def within_R(X, w, cell):
    """Pooled within-cell correlation matrix."""
    X = np.asarray(X, float)
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    _, ci = np.unique(np.asarray(cell), return_inverse=True)
    S0 = np.bincount(ci, weights=w)
    Mn = np.stack([np.bincount(ci, weights=w * X[:, i]) for i in range(X.shape[1])], 1) / S0[:, None]
    D = X - Mn[ci]
    Sw = (D * w[:, None]).T @ D / w.sum()
    d = np.sqrt(np.clip(np.diag(Sw), 1e-12, None))
    return Sw / np.outer(d, d)


# Source: scripts/91_ladder_core.py, class L4Ref (stratified clusters added from 81_l4_lib)
class L4Reference:
    """Reference loadings, point estimate and B bootstrap replicates.

    items : (n, J) reference answers scored 0..M-1. weights : survey weights or None.
    cluster : resampling unit per respondent (PSU for a survey; default each respondent).
    stratum : stratum per respondent; with it the replicates are Rao-Wu within strata.
    cell : optional demographic cell per respondent for the reference's R3 (within-cell factor).
    """

    def __init__(self, items, weights=None, cluster=None, stratum=None, n_categories=None,
                 n_boot=L4_N_BOOT, seed=0, cell=None):
        X = np.asarray(items, int)
        n, J = X.shape
        M = int(n_categories) if n_categories is not None else int(X.max()) + 1
        self.J, self.M = J, M
        cl = np.arange(n) if cluster is None else np.asarray(cluster)
        w = np.ones(n) if weights is None else np.asarray(weights, float)
        T = Tables(X, w, cl, M)
        strata_c = None
        if stratum is not None:
            st = np.asarray(stratum)
            first = np.zeros(T.C, int)
            first[T.cidx[::-1]] = np.arange(n)[::-1]
            strata_c = st[first]
        R = poly_from(T.agg(), J, M)
        self.R = R
        self.lam = one_factor_uls(R)
        self.ev = eig_desc(R)
        self.gf = general_factor(self.lam, self.ev)
        rng = np.random.default_rng([seed, 4])
        self.lam_b = np.array([one_factor_uls(poly_from(T.agg(cluster_boot_mult(T.C, rng, strata_c)), J, M))
                               for _ in range(n_boot)])
        self.within_ev_ratio = self.within_load_min = self.R3 = None
        if cell is not None:
            Rw = within_R(X, w, cell)
            lw, evw = one_factor_uls(Rw), eig_desc(Rw)
            self.within_ev_ratio, self.within_load_min, self.R3 = evw[0] / evw[1], float(lw.min()), general_factor(lw, evw)


def level4_verdict(R1, R2, r4=None):
    """Level-4 verdict for one model and framing.

    fail when R1 fails (later rules are then description), R2 fails or R4 fails; pass when R1 and
    R2 pass and R4 passes (or is not applicable because the personas carry no attribute the
    population records); otherwise unresolved. r4=None (not run) cannot give a pass."""
    if not R1 or R2 == "fail" or r4 == "fail":
        return "fail"
    if R2 == "pass" and r4 in ("pass", "not applicable"):
        return "pass"
    return "unresolved"


def level4(sim_items, persona, reference, *, ref_weights=None, ref_cluster=None, ref_stratum=None,
           n_categories=None, n_boot=None, seed=0, r4=None, r1_load=R1_LOAD, r1_ratio=R1_RATIO,
           r2_phi=R2_PHI, r2_rmsd=R2_RMSD, ci=INTERVAL_LEVEL):
    """Level 4 for one model and framing: do the simulated answers have the population's factor
    structure?

    Parameters
    ----------
    sim_items : (n_draws, J) answers scored 0..M-1; persona : persona of every draw.
    reference : an L4Reference, or the (n, J) reference answers (ref_weights, ref_cluster,
        ref_stratum then build one; reuse an L4Reference across models).
    n_boot : persona bootstrap replicates, paired one-to-one with the reference's (default: all).
    r4 : the R4 invariance verdict computed elsewhere: "pass", "fail", "unresolved", or
        "not applicable" when the personas carry no attribute the population records. None means
        R4 was not run, which leaves a level-4 pass unavailable.

    Returns
    -------
    dict: loadings, ev_ratio, R1 (bool), phi (congruence) and RMSD with 90% limits, R2 (pass /
    fail / unresolved), R3 diagnostics, r4, verdict.

    Reading
    -------
    A pass supports reading the simulated total as measuring one construct with the population's
    loading pattern and loading size (and, with R4, the same measurement across groups). A fail
    rules out that reading: no general factor (R1), loadings of a different shape or size (R2), or
    a measurement that differs across groups where the population's does not (R4).
    """
    if r4 not in (None, "pass", "fail", "unresolved", "not applicable"):
        raise ValueError('r4 must be None, "pass", "fail", "unresolved" or "not applicable"')
    if not isinstance(reference, L4Reference):
        reference = L4Reference(reference, weights=ref_weights, cluster=ref_cluster, stratum=ref_stratum,
                                n_categories=n_categories, n_boot=L4_N_BOOT if n_boot is None else n_boot,
                                seed=seed)
    ref = reference
    B = ref.lam_b.shape[0] if n_boot is None else int(n_boot)
    if B > ref.lam_b.shape[0]:
        raise ValueError("n_boot exceeds the reference's replicates")
    a = (1 - ci) / 2
    J, M = ref.J, ref.M
    X = np.asarray(sim_items, int)
    if X.shape[1] != J or X.min() < 0 or X.max() > M - 1:
        raise ValueError(f"simulated items must have {J} columns scored 0..{M - 1}")
    persona = np.asarray(persona)
    rng = np.random.default_rng([seed, 5])
    T = Tables(X, np.ones(len(X)), persona, M)
    R = poly_from(T.agg(), J, M)
    lam = one_factor_uls(R)
    ev = eig_desc(R)
    lbs = [one_factor_uls(poly_from(T.agg(cluster_boot_mult(T.C, rng)), J, M)) for _ in range(B)]
    phis = np.array([congruence(lb, ref.lam_b[b]) for b, lb in enumerate(lbs)])
    rmsds = np.array([np.sqrt(np.mean((lb - ref.lam_b[b]) ** 2)) for b, lb in enumerate(lbs)])
    phi_lo, phi_hi = float(np.quantile(phis, a)), float(np.quantile(phis, 1 - a))
    rmsd_lo, rmsd_hi = float(np.quantile(rmsds, a)), float(np.quantile(rmsds, 1 - a))
    Rw = within_R(X, None, persona)
    lw, evw = one_factor_uls(Rw), eig_desc(Rw)
    R1 = general_factor_vs_ref(lam, ev, ref.lam, ref.ev, min_load=r1_load, min_ratio=r1_ratio)
    R2 = r2_verdict(phi_lo, phi_hi, rmsd_lo, rmsd_hi, r2_phi=r2_phi, r2_rmsd=r2_rmsd)
    return dict(loadings=lam, ref_loadings=ref.lam, ev_ratio=float(ev[0] / ev[1]),
                ref_ev_ratio=float(ref.ev[0] / ref.ev[1]), load_min=float(lam.min()), R1=R1,
                R1_absolute=general_factor(lam, ev, min_load=r1_load, min_ratio=r1_ratio),
                phi=congruence(lam, ref.lam), phi_lo90=phi_lo, phi_hi90=phi_hi,
                loading_rmsd=float(np.sqrt(np.mean((lam - ref.lam) ** 2))), loading_rmsd_lo90=rmsd_lo,
                loading_rmsd_hi90=rmsd_hi, R2=R2, within_ev_ratio=float(evw[0] / evw[1]),
                within_load_min=float(lw.min()), R3=general_factor(lw, evw, min_load=r1_load, min_ratio=r1_ratio),
                r4=r4, verdict=level4_verdict(R1, R2, r4))
