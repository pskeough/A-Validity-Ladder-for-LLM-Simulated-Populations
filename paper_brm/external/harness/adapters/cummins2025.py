"""Cummins (2025), Study 1: 252 LLM configurations simulating 85 AIID participants (person-matched).

Level 2 (gaps mode): per configuration (family), per outcome (bjw_score, gut_diff) and per contrast
(adapters/_cummins.py: political identity, sex, education, age), g = simulated marginal gap and
gamma = human gap. Simulated participant i is the human participant_id i, so the two gaps share
sampling units: se_g, se_gamma and cov come from a bootstrap over the 85 participant_ids (B = 2000,
the same resamples for every configuration, seed 20261003), df = inf. gamma is always the gap in all
85 humans; g uses the participants whose output in that configuration is complete (complete_data == 1),
and in a bootstrap sample a missing simulated person drops out of g only. family_size = 8.

Level 3: per configuration and outcome, the mean of the paired differences (simulated minus human,
complete participants only) against a tolerance of 0.5 human SD (SD of all 85 humans, ddof 1). The SE
is the paired SE sd(d)/sqrt(n) with df n - 1, passed to level3_tost as the reference variance with
simulation variance 0 (one draw per person, so draw noise is inside d). Written to
results/cummins2025/l3.csv.
"""
import os

import numpy as np
import pandas as pd

import validity_ladder as vl
from adapters import _cummins as C

NAME = "cummins2025"
KIND = "language model (252 analytic-flexibility configurations)"
SOURCE = ("Cummins, J. (2025), The threat of analytic flexibility in using LLMs to simulate human data, "
          "arXiv 2509.13397; OSF https://osf.io/skbqm/ (study 1/data/processed/)")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", NAME)


def load():
    gt, masks, med = C.load_human()
    sim, comp = C.load_sim()
    H = gt[C.OUTCOMES].to_numpy(float)
    n = len(gt)
    rng = np.random.default_rng(C.SEED_MAIN)
    counts = C.resample_counts(rng, n)
    allok = np.ones(n, bool)
    sd_h = H.std(axis=0, ddof=1)
    comp = comp.set_index(C.CFG)

    # human side is the same for every configuration
    gam, gam_boot = {}, {}
    for j, o in enumerate(C.OUTCOMES):
        for c in C.CONTRASTS:
            hi, lo = masks[c]
            gam[o, c] = C.mean_gap(H[:, j], allok, hi, lo)
            gam_boot[o, c] = C.boot_gap(H[:, j], allok, hi, lo, counts)

    rows, l3 = [], []
    for key, S in sim.items():
        fam = " | ".join(str(k) for k in key)
        cr = float(comp.loc[key, "completion_rate"])
        meta = dict(zip(C.CFG, key))
        for j, o in enumerate(C.OUTCOMES):
            x = S[:, j]
            ok = ~np.isnan(x)
            # level 3
            if ok.sum() >= 2:
                d = (x - H[:, j])[ok]
                tol = 0.5 * sd_h[j]
                t = vl.level3_tost(float(d.mean()), 0.0, float(d.var(ddof=1) / len(d)), len(d) - 1, tol)
                l3.append(dict(family=fam, outcome=o, resid=float(d.mean()), ci_lo=t["ci90_lo"], ci_hi=t["ci90_hi"],
                               verdict=t["verdict"], n_pairs=int(ok.sum()), se=t["se"], df=t["df"],
                               tolerance=tol, resid_sd_units=float(d.mean() / sd_h[j]), **meta,
                               completion_rate=cr))
            if ok.sum() < 4:
                continue
            const = np.ptp(x[ok]) == 0
            for c in C.CONTRASTS:
                hi, lo = masks[c]
                g = C.mean_gap(x, ok, hi, lo)
                if not np.isfinite(g):
                    continue
                if const:
                    g, se_g, cov = 0.0, 0.0, 0.0
                    gb = np.zeros(len(counts))
                else:
                    gb = C.boot_gap(x, ok, hi, lo, counts)
                    v = np.isfinite(gb) & np.isfinite(gam_boot[o, c])
                    se_g = float(np.std(gb[v], ddof=1))
                    cov = float(np.cov(gb[v], gam_boot[o, c][v], ddof=1)[0, 1])
                pb = gam_boot[o, c]
                rows.append(dict(family=fam, contrast=f"{o} | {c}", g=g, se_g=se_g, gamma=gam[o, c],
                                 se_gamma=float(np.nanstd(pb, ddof=1)), cov=cov, family_size=8,
                                 outcome=o, attribute=c, informed=c in C.INFORMED[meta["demographics_used"]],
                                 n_sim=int(ok.sum()), n_hi=int((hi & ok).sum()), n_lo=int((lo & ok).sum()),
                                 completion_rate=cr, authors_excluded=cr < 0.5, **meta))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(l3).to_csv(os.path.join(OUT, "l3.csv"), index=False)
    return dict(mode="gaps", gaps=pd.DataFrame(rows), inputs=[C.FULL, C.GT, C.COMPLETION],
                notes=f"B={C.B}, seed={C.SEED_MAIN}, age median cut {med}; contrasts fixed before results; "
                      "gamma on all 85 humans; marginal gaps; df inf")
