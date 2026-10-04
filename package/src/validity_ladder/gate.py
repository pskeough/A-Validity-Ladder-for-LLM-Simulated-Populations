"""Gate: precision of a persona mean (one-facet G-study within one model and framing).

Ported from scripts/82_gate_lib.py (one_facet, pos, phi_k, k_for_phi, k_for_se) and
scripts/91_ladder_core.py (gate, gate_tolerances). The bootstrap follows
paper_brm/external/scripts/bisbee_ladder.py (gate_rows): personas resampled with replacement.
"""
import numpy as np

from .thresholds import (GATE_N_BOOT, GATE_PHI_MIN, GATE_PHI_REC, GATE_SE_MIN_SD, GATE_SE_REC_SD,
                         INTERVAL_LEVEL)


# Source: scripts/82_gate_lib.py, pos
def pos(x):
    """Variance component truncated at zero."""
    return max(float(x), 0.0)


# Source: scripts/82_gate_lib.py, one_facet
def one_facet(mean, n, ss):
    """Persona (p) with draws (r) nested: unbalanced one-way ANOVA, method of moments.

    mean, n, ss: per-persona mean, number of draws and within-persona sum of squares. Personas with
    n = 0 are skipped. sigma2_p = (MS_p - MS_e) / n0 with n0 = (N - sum n_i^2 / N) / (a - 1).
    Negative estimates are returned raw; coefficients truncate them at zero.
    """
    mean, n, ss = np.asarray(mean, float), np.asarray(n, float), np.asarray(ss, float)
    ok = n > 0
    mean, n, ss = mean[ok], n[ok], ss[ok]
    a, N = len(n), n.sum()
    if a < 2:
        raise ValueError("the gate needs at least two personas")
    grand = (n * mean).sum() / N
    ms_p = (n * (mean - grand) ** 2).sum() / (a - 1)
    df_e = (n - 1).clip(min=0).sum()
    if df_e <= 0:
        raise ValueError("the gate needs repeated draws of at least one persona")
    ms_e = ss.sum() / df_e
    n0 = (N - (n ** 2).sum() / N) / (a - 1)
    return dict(p=(ms_p - ms_e) / n0, e=ms_e, n_cells=int(a), n_draws=int(N), n0=float(n0),
                ms_p=ms_p, ms_e=ms_e, grand=grand)


# Source: scripts/82_gate_lib.py, phi_k
def phi_k(s2_p, s2_r, k):
    """Dependability of a k-draw persona mean: s2_p / (s2_p + s2_r / k), components truncated at 0."""
    sp, se = pos(s2_p), pos(s2_r)
    if sp + se == 0:
        return np.nan
    return sp / (sp + se / k)


# Source: scripts/82_gate_lib.py, k_for_phi
def k_for_phi(s2_p, s2_r, target):
    """Smallest integer k with phi(k) >= target; inf when the persona component is zero."""
    sp, se = pos(s2_p), pos(s2_r)
    if se == 0:
        return 1.0
    if sp == 0:
        return np.inf
    return float(max(1, int(np.ceil(target / (1 - target) * se / sp - 1e-12))))


# Source: scripts/82_gate_lib.py, k_for_se
def k_for_se(s2_r, tol):
    """Smallest integer k with sqrt(s2_r / k) <= tol: ceil(s2_r / tol^2)."""
    se = pos(s2_r)
    return float(max(1, int(np.ceil(se / tol ** 2 - 1e-12))))


def persona_summaries(scores, persona):
    """Per-persona mean, draw count and within-persona sum of squares."""
    persona = np.asarray(persona)
    scores = np.asarray(scores, float)
    if len(scores) != len(persona):
        raise ValueError("scores and persona must have the same length")
    keep = np.isfinite(scores)
    scores, persona = scores[keep], persona[keep]
    _, inv = np.unique(persona, return_inverse=True)
    n = np.bincount(inv).astype(float)
    s1 = np.bincount(inv, weights=scores)
    mean = s1 / n
    ss = np.bincount(inv, weights=(scores - mean[inv]) ** 2)
    return mean, n, ss


def gate(scores, persona, sd_ref, *, k=None, se_min_sd=GATE_SE_MIN_SD, se_rec_sd=GATE_SE_REC_SD,
         phi_min=GATE_PHI_MIN, phi_rec=GATE_PHI_REC, n_boot=GATE_N_BOOT, ci=INTERVAL_LEVEL, seed=0):
    """Gate for one model and framing: is a k-draw persona mean precise enough to read?

    Parameters
    ----------
    scores : array of per-draw instrument totals (one entry per draw).
    persona : array of persona labels, same length.
    sd_ref : standard deviation of the total in the population reference (3.935 for the PHQ-8 in
        NHANES 2005-2018). Tolerances are stated in these units.
    k : number of draws averaged into a persona mean. Default: the median draws per persona.
    se_min_sd, se_rec_sd : SE tolerances in reference SD units (minimum 0.25, recommended 0.125).
    phi_min, phi_rec : separation thresholds for uses that compare or rank individual personas.
    n_boot : percentile bootstrap over personas for the intervals (0 skips it).

    Returns
    -------
    dict with s2_p and s2_r (the persona and draw variance components), se_k = sqrt(s2_r / k),
    phi_k, the tolerances in score points, k_min / k_rec (draws needed for each rule),
    k_sep_min / k_sep_rec (draws needed for phi >= .80 / .90), pass flags and `verdict`.

    Reading
    -------
    A pass supports reading a k-draw persona mean as precise to within the tolerance: the draw-to-
    draw noise left in the mean is at most 0.25 (minimum) or 0.125 (recommended) reference SD. It
    does not support any claim about validity; the gate reads precision only. phi(k) is reported
    beside the verdict and is required (>= .80, recommended .90) only for uses that compare or rank
    individual personas. A fail rules out reading single persona means at this k; `k_min` gives the
    number of draws that would meet the rule.
    """
    mean, n, ss = persona_summaries(scores, persona)
    if k is None:
        k = float(np.median(n))
    c = one_facet(mean, n, ss)
    t_min, t_rec = se_min_sd * sd_ref, se_rec_sd * sd_ref
    se_k = float(np.sqrt(pos(c["e"]) / k))
    phi = phi_k(c["p"], c["e"], k)
    out = dict(n_personas=c["n_cells"], n_draws=c["n_draws"], k=k, grand_mean=float(c["grand"]),
               s2_p=float(c["p"]), s2_r=float(c["e"]), within_sd=float(np.sqrt(pos(c["e"]))),
               se_k=se_k, phi_k=phi, phi_1=phi_k(c["p"], c["e"], 1), sd_ref=float(sd_ref),
               tol_min=t_min, tol_rec=t_rec, k_min=k_for_se(c["e"], t_min), k_rec=k_for_se(c["e"], t_rec),
               k_sep_min=k_for_phi(c["p"], c["e"], phi_min), k_sep_rec=k_for_phi(c["p"], c["e"], phi_rec),
               pass_min=bool(se_k <= t_min), pass_rec=bool(se_k <= t_rec),
               pass_single=bool(np.sqrt(pos(c["e"])) <= t_min),
               sep_min=bool(phi >= phi_min), sep_rec=bool(phi >= phi_rec))
    if n_boot:
        rng = np.random.default_rng(seed)
        a = (1 - ci) / 2
        bs = np.empty((n_boot, 3))
        for b in range(n_boot):
            i = rng.integers(0, len(n), len(n))
            try:
                cb = one_facet(mean[i], n[i], ss[i])
                bs[b] = (np.sqrt(pos(cb["e"]) / k), phi_k(cb["p"], cb["e"], k), k_for_se(cb["e"], t_min))
            except ValueError:
                bs[b] = np.nan
        for j, name in enumerate(("se_k", "phi_k", "k_min")):
            out[f"{name}_lo"] = float(np.nanquantile(bs[:, j], a))
            out[f"{name}_hi"] = float(np.nanquantile(bs[:, j], 1 - a))
    out["verdict"] = "pass" if out["pass_min"] else "fail"
    out["rule_met"] = "recommended" if out["pass_rec"] else ("minimum" if out["pass_min"] else "none")
    return out
