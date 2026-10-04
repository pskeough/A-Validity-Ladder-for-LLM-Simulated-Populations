"""Design-based variance for the population reference.

A reference with a complex sample design passes `psu` and `stratum` (one value per respondent;
PSU labels need only be unique within a stratum). Without them every respondent is its own PSU in
a single stratum, which turns each estimator below into its iid counterpart (jackknife, Taylor and
person bootstrap over respondents).
"""
import numpy as np
import pandas as pd


def design_codes(n, psu=None, stratum=None):
    """PSU index of every row and stratum index of every PSU."""
    if psu is None and stratum is None:
        return np.arange(n), np.zeros(n, int)
    psu = np.zeros(n, int) if psu is None else np.asarray(psu)
    stratum = np.zeros(n, int) if stratum is None else np.asarray(stratum)
    key = pd.Series(stratum).astype(str) + "\x1f" + pd.Series(psu).astype(str)
    codes, uniq = pd.factorize(key)
    pstrat = [u.split("\x1f")[0] for u in uniq]
    s_of_psu, _ = pd.factorize(pd.Series(pstrat))
    return np.asarray(codes), np.asarray(s_of_psu)


# Source: scripts/79_l1_lib.py, raowu_mult (strata passed explicitly instead of psu_code // 10)
def raowu_multipliers(s_of_psu, B, rng):
    """Rao-Wu rescaling bootstrap: in each stratum draw n_h - 1 PSUs with replacement from n_h and
    scale by n_h / (n_h - 1). Returns (B, n_psu) multipliers. Strata with one PSU keep it (x1)."""
    s_of_psu = np.asarray(s_of_psu)
    mult = np.zeros((B, len(s_of_psu)))
    for h in np.unique(s_of_psu):
        members = np.where(s_of_psu == h)[0]
        nh = len(members)
        if nh < 2:
            mult[:, members] = 1.0
            continue
        draws = rng.integers(0, nh, size=(B, nh - 1))
        for jpos, m in enumerate(members):
            mult[:, m] = (draws == jpos).sum(axis=1) * nh / (nh - 1)
    return mult


# Source: scripts/80_l3_lib.py, class Design (constructor takes codes instead of a frame)
class JKnDesign:
    """PSU structure and the stratified delete-one-PSU jackknife (JKn).

    Replicate (h, j) drops PSU j of stratum h and multiplies the weights of the stratum's other PSUs
    by n_h / (n_h - 1); v = sum_h (n_h - 1) / n_h * sum_j (theta_hj - theta)^2. Strata with one PSU
    are skipped. Design df = PSUs minus strata."""

    def __init__(self, n, psu=None, stratum=None):
        self.codes, self.s_of_psu = design_codes(n, psu, stratum)
        self.K = len(self.s_of_psu)
        self.n_h = np.bincount(self.s_of_psu)
        self.dfree = int(self.K - len(self.n_h))
        self.rep_psu = np.where(self.n_h[self.s_of_psu] >= 2)[0]
        self.rep_h = self.s_of_psu[self.rep_psu]
        self.rep_nh = self.n_h[self.rep_h]

    def psu_totals(self, X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        T = np.zeros((self.K, X.shape[1]))
        np.add.at(T, self.codes, X)
        return T

    def replicate(self, T):
        """PSU totals (K, m) -> full totals (m,) and replicate totals (R, m)."""
        full = T.sum(0)
        S = np.zeros((len(self.n_h), T.shape[1]))
        np.add.at(S, self.s_of_psu, T)
        f = (self.rep_nh / (self.rep_nh - 1.0))[:, None]
        reps = full[None, :] - S[self.rep_h] + f * (S[self.rep_h] - T[self.rep_psu])
        return full, reps

    def jk_var(self, theta, theta_reps):
        c = (self.rep_nh - 1.0) / self.rep_nh
        dev = np.asarray(theta_reps) - theta
        return np.nansum(c.reshape((-1,) + (1,) * (dev.ndim - 1)) * dev ** 2, axis=0)


# Source: scripts/78a_l2_population_reference.py, Design.contrast (Taylor part; the Rao-Wu check
# is left out)
def taylor_contrast(y, w, masks, coef, psu=None, stratum=None):
    """Linear contrast sum_j coef_j * mean_w(y | domain j) with its Taylor-linearised SE.

    Each domain mean is linearised as a ratio, the linear combination is taken, and the
    with-replacement between-PSU variance is computed within strata, so the covariance between
    domains that share PSUs is carried. df = PSUs minus strata among those the contrast touches.
    """
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    codes, s_of_psu = design_codes(len(y), psu, stratum)
    P = len(s_of_psu)
    D = len(masks)
    A = np.zeros((P, D))
    Bw = np.zeros((P, D))
    N = np.zeros((P, D))
    for j, mk in enumerate(masks):
        mk = np.asarray(mk, bool)
        A[:, j] = np.bincount(codes[mk], weights=(w * y)[mk], minlength=P)
        Bw[:, j] = np.bincount(codes[mk], weights=w[mk], minlength=P)
        N[:, j] = np.bincount(codes[mk], minlength=P)
    Wt = Bw.sum(0)
    if np.any(Wt <= 0):
        raise ValueError("a reference domain of the contrast is empty")
    R = A.sum(0) / Wt
    a = np.asarray(coef, float)
    est = float(R @ a)
    z = ((A - R * Bw) / Wt) @ a
    touched = N.sum(1) > 0
    var, n_psu, strata_touched = 0.0, 0, 0
    for h in np.unique(s_of_psu):
        s = s_of_psu == h
        zh = z[s]
        nh = len(zh)
        if nh > 1:
            var += nh / (nh - 1) * ((zh - zh.mean()) ** 2).sum()
        k = int(touched[s].sum())
        if k:
            n_psu += k
            strata_touched += 1
    return dict(estimate=est, se=float(np.sqrt(var)), df=int(n_psu - strata_touched),
                domain_means=R, domain_n=N.sum(0).astype(int))
