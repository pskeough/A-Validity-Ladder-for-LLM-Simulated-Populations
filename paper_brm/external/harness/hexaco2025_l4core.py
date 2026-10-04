"""Facet-level level 4 for the HEXACO 2025 adapter (see hexaco2025_lib.py for the decisions).

The package's level4() takes item answers and builds polychoric matrices. The reference here is a
different instrument, so the only shared indicators are facet scores. This module repeats the body of
validity_ladder.level4.level4() with Pearson matrices of facet scores in place of polychorics, and uses
the package functions for every rule: one_factor_uls, eig_desc, congruence, general_factor_vs_ref,
r2_verdict, level4_verdict. Persona (agent) bootstrap B replicates are paired one-to-one with the
reference's B replicates, 90% limits are the 5th and 95th percentiles, as in level4().
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hexaco2025_lib as L
from validity_ladder.level4 import (congruence, eig_desc, general_factor, general_factor_vs_ref, level4_verdict,
                                    one_factor_uls, r2_verdict)
from validity_ladder.thresholds import R1_LOAD, R1_RATIO, R2_PHI, R2_RMSD

A90 = 0.05


def corr(F):
    return np.corrcoef(np.asarray(F, float), rowvar=False)


def fit(F):
    R = corr(F)
    return one_factor_uls(R), eig_desc(R)


class Reference:
    """Human reference: keyed item matrices per IPIP facet (n x 10), reference loadings and B bootstrap
    replicates for every domain set. mode 'full10': facet = mean of ten items; 'sub4': a fresh random four
    of the ten items per facet in every replicate (point estimate = mean over 200 draws)."""

    def __init__(self, keyed_items, mode="full10", n_boot=1000, seed=L.SEED, label=""):
        self.mode, self.B, self.label = mode, n_boot, label
        self.items = keyed_items  # dict prefix -> (n, 10) array
        self.n = next(iter(keyed_items.values())).shape[0]
        self.sets = L.DOMAIN_SETS
        self.cols = {s: [L.IPIP_OF[f] for f in fac] for s, (_, fac, _) in self.sets.items()}
        rng = np.random.default_rng([seed, 4])
        self.lam, self.ev = {}, {}
        if mode == "full10":
            F = self._scores(None)
            for s in self.sets:
                self.lam[s], self.ev[s] = fit(F[self.cols[s]])
        else:
            lams = {s: [] for s in self.sets}
            evs = {s: [] for s in self.sets}
            for _ in range(200):
                F = self._scores(rng)
                for s in self.sets:
                    l, e = fit(F[self.cols[s]])
                    lams[s].append(l)
                    evs[s].append(e)
            for s in self.sets:
                self.lam[s], self.ev[s] = np.mean(lams[s], 0), np.mean(evs[s], 0)
        self.lam_b = {s: np.empty((n_boot, len(self.cols[s]))) for s in self.sets}
        for b in range(n_boot):
            F = self._scores(rng if mode == "sub4" else None)
            idx = rng.integers(0, self.n, self.n)
            Fb = F.iloc[idx]
            for s in self.sets:
                self.lam_b[s][b] = one_factor_uls(corr(Fb[self.cols[s]]))

    def _scores(self, rng):
        if self.mode == "full10":
            if not hasattr(self, "_F"):
                self._F = pd.DataFrame({p: m.mean(1) for p, m in self.items.items()})
            return self._F
        out = {}
        for p, m in self.items.items():
            sel = rng.choice(m.shape[1], 4, replace=False)
            out[p] = m[:, sel].mean(1)
        return pd.DataFrame(out)


def read_set(F_sim, ref, setname, n_boot=None, seed=L.SEED, r4="not applicable"):
    """Level 4 of one simulated population on one domain set. F_sim: DataFrame of facet scores by PI-R
    facet name, one row per persona. Returns a dict (one CSV row)."""
    dom, fac, note = ref.sets[setname]
    X = F_sim[fac].to_numpy(float)
    n = len(X)
    B = ref.B if n_boot is None else int(n_boot)
    lam, ev = fit(X)
    rlam, rev = ref.lam[setname], ref.ev[setname]
    rng = np.random.default_rng([seed, 5])
    phis, rmsds = np.empty(B), np.empty(B)
    for b in range(B):
        lb = one_factor_uls(corr(X[rng.integers(0, n, n)]))
        phis[b] = congruence(lb, ref.lam_b[setname][b])
        rmsds[b] = np.sqrt(np.mean((lb - ref.lam_b[setname][b]) ** 2))
    plo, phi_hi = float(np.quantile(phis, A90)), float(np.quantile(phis, 1 - A90))
    rlo, rhi = float(np.quantile(rmsds, A90)), float(np.quantile(rmsds, 1 - A90))
    R1 = general_factor_vs_ref(lam, ev, rlam, rev)
    R2 = r2_verdict(plo, phi_hi, rlo, rhi)
    return dict(domain=setname, facets="; ".join(fac), n_indicators=len(fac), n_personas=n, n_boot=B,
                R1=bool(R1), R1_absolute=bool(general_factor(lam, ev)), load_min=float(lam.min()),
                ref_load_min=float(rlam.min()), ev_ratio=float(ev[0] / ev[1]), ref_ev_ratio=float(rev[0] / rev[1]),
                congruence=congruence(lam, rlam), congruence_lo90=plo, congruence_hi90=phi_hi,
                rmsd=float(np.sqrt(np.mean((lam - rlam) ** 2))), rmsd_lo90=rlo, rmsd_hi90=rhi,
                R2=R2, r4=r4, level4=level4_verdict(R1, R2, r4),
                loadings=";".join(f"{v:.3f}" for v in lam), ref_loadings=";".join(f"{v:.3f}" for v in rlam),
                note=note)


def human_items(variant, keying, hum):
    """Keyed item matrices (dict IPIP prefix -> (n, 10)) for a reference variant, plus n."""
    items = list(keying.item)
    x = hum[items].to_numpy(float)
    ok = ~(x == 0).any(1)  # 0 = unanswered (52 respondents)
    if variant != "S2_nofilter":
        ok &= ((hum.V1 >= 4) & (hum.V2 >= 4)).to_numpy()
    if variant == "S3_GB":
        ok &= (hum.country == "GB").to_numpy()
    sub = hum.loc[ok]
    out = {}
    for pref, g in keying.groupby("facet"):
        cols = []
        for it, sg in zip(g.item, g.sign):
            v = sub[it].to_numpy(float)
            cols.append(v if sg > 0 else 8.0 - v)
        out[pref] = np.column_stack(cols)
    return out, int(ok.sum())
