"""Graded response model, WLE theta and the lz / lz* person-fit statistics for J items with M
ordered categories scored 0..M-1, plus the within-total reference CDF used by level 1.

Ported without change of method from scripts/91_ladder_core.py (GRM, total_strata) and
scripts/79_l1_lib.py (TailRef, tail_probs).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# Source: scripts/91_ladder_core.py, module constants (as 79_l1_lib.py)
QN = 81
QT = np.linspace(-6, 6, QN)
QW = np.exp(-0.5 * QT ** 2)
QW = QW / QW.sum()
GRID = np.linspace(-10, 10, 2001)


# Source: scripts/91_ladder_core.py, class GRM
class GRM:
    """Graded response model for J items with M categories (marginal ML, 81-point quadrature)."""

    def __init__(self, J, M):
        self.J, self.M = int(J), int(M)

    # parameters per item: log a, b1, log(b2 - b1), ..., log(b_{M-1} - b_{M-2})
    def unpack(self, par):
        p = par.reshape(self.J, self.M)
        a = np.exp(p[:, 0])
        b = np.cumsum(np.column_stack([p[:, 1], np.exp(p[:, 2:])]), axis=1)
        return a, b

    def pack(self, a, b):
        return np.column_stack([np.log(a), b[:, 0], np.log(np.diff(b, axis=1))]).ravel()

    def cat_probs(self, a, b, theta):
        """P, P', P'' with shape (len(theta), J, M)."""
        th = np.asarray(theta, float)[:, None, None]
        s = expit(a[None, :, None] * (th - b[None, :, :]))
        ds = a[None, :, None] * s * (1 - s)
        d2s = a[None, :, None] ** 2 * s * (1 - s) * (1 - 2 * s)
        one = np.ones(s.shape[:2] + (1,))
        zero = np.zeros_like(one)
        S = np.concatenate([one, s, zero], axis=2)
        D = np.concatenate([zero, ds, zero], axis=2)
        D2 = np.concatenate([zero, d2s, zero], axis=2)
        M = self.M
        return (np.clip(S[:, :, :M] - S[:, :, 1:], 1e-300, 1.0), D[:, :, :M] - D[:, :, 1:],
                D2[:, :, :M] - D2[:, :, 1:])

    def patterns(self, X, w=None):
        X = np.asarray(X, np.int64)
        key = (X * (self.M ** np.arange(self.J, dtype=np.int64))).sum(axis=1)
        uk, inv = np.unique(key, return_inverse=True)
        w = np.ones(len(X)) if w is None else np.asarray(w, float)
        f = np.bincount(inv, weights=w)
        U = np.stack([(uk // self.M ** j) % self.M for j in range(self.J)], axis=1)
        return U, f, inv

    def _negll_grad(self, par, U, f):
        J, M = self.J, self.M
        a, b = self.unpack(par)
        s = expit(a[None, :, None] * (QT[:, None, None] - b[None, :, :]))     # (Q, J, M-1)
        S = np.concatenate([np.ones((QN, J, 1)), s, np.zeros((QN, J, 1))], axis=2)
        P = np.clip(S[:, :, :M] - S[:, :, 1:], 1e-300, 1.0)
        logP = np.log(P)
        LL = np.zeros((len(U), QN))
        for j in range(J):
            LL += logP[:, j, :][:, U[:, j]].T
        mx = LL.max(axis=1, keepdims=True)
        L = np.exp(LL - mx) * QW[None, :]
        marg = L.sum(axis=1)
        ll = float((f * (np.log(marg) + mx[:, 0])).sum())
        post = L / marg[:, None] * f[:, None]
        grad = np.zeros((J, M))
        for j in range(J):
            r = np.zeros((QN, M))
            for k in range(M):
                r[:, k] = post[U[:, j] == k].sum(axis=0)
            rp = r / P[:, j, :]
            gS = rp[:, 1:M] - rp[:, 0:M - 1]
            sk = s[:, j, :]
            dsk = sk * (1 - sk)
            g_a = (gS * dsk * (QT[:, None] - b[j][None, :])).sum()
            g_b = -(gS * dsk).sum(axis=0) * a[j]
            e = np.diff(b[j])
            tail = np.cumsum(g_b[::-1])[::-1]
            grad[j] = np.concatenate([[a[j] * g_a, g_b.sum()], e * tail[1:]])
        return -ll, -grad.ravel()

    def fit(self, X, w=None):
        """Marginal ML fit; w are survey weights (rescaled to sum to n)."""
        X = np.asarray(X, int)
        if w is not None:
            w = np.asarray(w, float) * len(X) / np.sum(w)
        U, f, _ = self.patterns(X, w)
        a0 = np.full(self.J, 1.5)
        b0 = np.tile(np.linspace(-1.5, 1.5, self.M - 1) if self.M > 4
                     else np.array([0.5, 1.5, 2.5])[:self.M - 1], (self.J, 1))
        res = minimize(self._negll_grad, self.pack(a0, b0), args=(U, f), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=5000, gtol=1e-7, ftol=1e-13))
        a, b = self.unpack(res.x)
        return dict(a=a, b=b, loglik=-res.fun, converged=bool(res.success), nit=res.nit,
                    n_patterns=len(U))

    def _wle_score(self, a, b, U, theta):
        P, P1, P2 = self.cat_probs(a, b, theta)
        idx = np.arange(len(U))[:, None]
        jj = np.arange(self.J)[None, :]
        r = (P1 / P)[idx, jj, U].sum(axis=1)
        I = (P1 ** 2 / P).sum(axis=(1, 2))
        Jt = (P1 * P2 / P).sum(axis=(1, 2))
        return r + Jt / (2 * I)

    def wle(self, a, b, U):
        U = np.asarray(U, int)
        P, P1, P2 = self.cat_probs(a, b, GRID)
        R = P1 / P
        I = (P1 ** 2 / P).sum(axis=(1, 2))
        Jt = (P1 * P2 / P).sum(axis=(1, 2))
        corr = Jt / (2 * I)
        theta = np.empty(len(U))
        for s0 in range(0, len(U), 4000):
            u = U[s0:s0 + 4000]
            S = corr[None, :].repeat(len(u), 0)
            for j in range(self.J):
                S += R[:, j, :][:, u[:, j]].T
            sgn = S > 0
            chg = sgn[:, :-1] & ~sgn[:, 1:]
            has = chg.any(axis=1)
            k = np.where(has, chg.shape[1] - 1 - np.argmax(chg[:, ::-1], axis=1), 0)
            lo, hi = GRID[k], GRID[k + 1]
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                sm = self._wle_score(a, b, u, mid)
                lo = np.where(sm > 0, mid, lo)
                hi = np.where(sm > 0, hi, mid)
            t = 0.5 * (lo + hi)
            theta[s0:s0 + 4000] = np.where(has, t, np.where(sgn[:, -1], GRID[-1], GRID[0]))
        return theta

    def personfit(self, a, b, U, theta):
        U = np.asarray(U, int)
        P, P1, P2 = self.cat_probs(a, b, theta)
        logP = np.log(P)
        idx = np.arange(len(U))[:, None]
        jj = np.arange(self.J)[None, :]
        obs = logP[idx, jj, U]
        Ew = (P * logP).sum(axis=2)
        Vw = (P * logP ** 2).sum(axis=2) - Ew ** 2
        num = (obs - Ew).sum(axis=1)
        lz = num / np.sqrt(Vw.sum(axis=1))
        r = P1 / P
        I = (P1 * r).sum(axis=(1, 2))
        Jt = (P1 * P2 / P).sum(axis=(1, 2))
        r0 = Jt / (2 * I)
        cn = (P1 * logP).sum(axis=(1, 2)) / I
        wt = logP - cn[:, None, None] * r
        Ewt = (P * wt).sum(axis=2)
        Vwt = (P * wt ** 2).sum(axis=2) - Ewt ** 2
        lzs = (num + cn * r0) / np.sqrt(Vwt.sum(axis=1))
        return lz, lzs

    def score(self, a, b, X):
        """theta (WLE), lz and lz* (Snijders correction) for every row of X."""
        U, _, inv = self.patterns(X)
        th = self.wle(a, b, U)
        lz, lzs = self.personfit(a, b, U, th)
        return th[inv], lz[inv], lzs[inv]


# Source: scripts/91_ladder_core.py, total_strata
def total_strata(ref_total, max_total, min_n=100):
    """Exact totals, merged downward from the top until each stratum holds min_n unweighted
    reference respondents. Returns the stratum label (lowest total in the stratum) of every total."""
    ref_total = np.asarray(ref_total, int)
    if len(ref_total) < min_n:
        raise ValueError(f"the reference has fewer than {min_n} respondents")
    cnt = np.bincount(ref_total, minlength=max_total + 1)
    lab = np.arange(max_total + 1)
    groups, cur, cc = [], [], 0
    for t in range(max_total, -1, -1):
        cur.append(t)
        cc += cnt[t]
        if cc >= min_n:
            groups.append(cur)
            cur, cc = [], 0
    if cur:
        groups[-1].extend(cur)
    for g in groups:
        lab[g] = min(g)
    return lab


# Source: scripts/79_l1_lib.py, class TailRef
class TailRef:
    """Reference CDF of lz* within strata. For a query (stratum, value) returns F(value-) and
    F(value) under any set of reference weights, so bootstrap replicates reuse the sort."""

    def __init__(self, s_ref, v_ref):
        self.s = np.asarray(s_ref)
        self.v = np.round(np.asarray(v_ref, float), 10)
        self.order = np.lexsort((self.v, self.s))
        self.ss = self.s[self.order]
        self.vs = self.v[self.order]
        self.strata = np.unique(self.ss)
        self.start = {h: np.searchsorted(self.ss, h, "left") for h in self.strata}
        self.stop = {h: np.searchsorted(self.ss, h, "right") for h in self.strata}

    def locate(self, s_q, v_q):
        s_q = np.asarray(s_q)
        v_q = np.round(np.asarray(v_q, float), 10)
        lo = np.empty(len(s_q), int)
        hi = np.empty(len(s_q), int)
        st = np.empty(len(s_q), int)
        en = np.empty(len(s_q), int)
        for h in np.unique(s_q):
            m = s_q == h
            a, e = self.start[h], self.stop[h]
            seg = self.vs[a:e]
            lo[m] = a + np.searchsorted(seg, v_q[m], "left")
            hi[m] = a + np.searchsorted(seg, v_q[m], "right")
            st[m] = a
            en[m] = e
        return lo, hi, st, en

    def cdf(self, loc, w_ref):
        lo, hi, st, en = loc
        cw = np.concatenate([[0.0], np.cumsum(np.asarray(w_ref, float)[self.order])])
        tot = cw[en] - cw[st]
        return (cw[lo] - cw[st]) / tot, (cw[hi] - cw[st]) / tot


# Source: scripts/79_l1_lib.py, tail_probs
def tail_probs(Fm, F, q=0.05):
    """Expected indicator that the randomised PIT falls below q (misfit) and above 1 - q (overfit),
    plus the mid-PIT."""
    mass = np.maximum(F - Fm, 1e-15)
    low = np.clip((q - Fm) / mass, 0, 1)
    high = np.clip((F - (1 - q)) / mass, 0, 1)
    return low, high, 0.5 * (Fm + F)
