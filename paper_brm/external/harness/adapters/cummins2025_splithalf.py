"""Ideal-simulator control for cummins2025: the 85 humans split at random into halves of 42 and 43,
half A read as the simulator and half B as the reference, 200 times (seeds 0 to 199). The halves are
independent, so cov = 0; se_g and se_gamma are the SD of 2000 bootstrap resamples within each half.
Contrasts, outcomes, cuts (age 27 from the full sample) and family_size = 8 are those of cummins2025.
A contrast with an empty side in either half is not read (counted in the manifest notes). A reference
gap of exactly 0 (possible with integer scores) makes the package divide by zero; it is entered as
1e-12 (gamma_raw keeps the 0), which the package reads as "no population gap", the correct stop.
Level 3 is run the same way (mean of A minus mean of B, independent SEs, tolerance 0.5 SD of all 85
humans) and written to results/cummins2025_splithalf/l3.csv.
"""
import os

import numpy as np
import pandas as pd

import validity_ladder as vl
from adapters import _cummins as C

NAME = "cummins2025_splithalf"
KIND = "human split-half (ideal-simulator control)"
SOURCE = "Cummins (2025) Study 1 human participants (85), randomly split 42/43, 200 seeds; see cummins2025"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", NAME)
N_SPLITS = 200


def load():
    gt, masks, med = C.load_human()
    H = gt[C.OUTCOMES].to_numpy(float)
    n = len(gt)
    sd_h = H.std(axis=0, ddof=1)
    rows, l3, skipped = [], [], 0
    for seed in range(N_SPLITS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        A, Bi = perm[:42], perm[42:]
        cA = C.resample_counts(rng, len(A))
        cB = C.resample_counts(rng, len(Bi))
        fam = f"human_split_half | seed={seed:03d}"
        for j, o in enumerate(C.OUTCOMES):
            xa, xb = H[A, j], H[Bi, j]
            tol = 0.5 * sd_h[j]
            va, vb = xa.var(ddof=1) / len(xa), xb.var(ddof=1) / len(xb)
            t = vl.level3_tost(float(xa.mean() - xb.mean()), float(va), float(vb), len(xb) - 1, tol)
            l3.append(dict(family=fam, outcome=o, resid=float(xa.mean() - xb.mean()), ci_lo=t["ci90_lo"],
                           ci_hi=t["ci90_hi"], verdict=t["verdict"], se=t["se"], df=t["df"], tolerance=tol,
                           resid_sd_units=float((xa.mean() - xb.mean()) / sd_h[j]), seed=seed))
            for c in C.CONTRASTS:
                hi, lo = masks[c]
                okA, okB = np.ones(len(A), bool), np.ones(len(Bi), bool)
                g = C.mean_gap(xa, okA, hi[A], lo[A])
                p = C.mean_gap(xb, okB, hi[Bi], lo[Bi])
                if not (np.isfinite(g) and np.isfinite(p)):
                    skipped += 1
                    continue
                gb = C.boot_gap(xa, okA, hi[A], lo[A], cA)
                pb = C.boot_gap(xb, okB, hi[Bi], lo[Bi], cB)
                rows.append(dict(family=fam, contrast=f"{o} | {c}", g=g, se_g=float(np.nanstd(gb, ddof=1)),
                                 gamma=p if p != 0 else 1e-12, gamma_raw=p,
                                 se_gamma=float(np.nanstd(pb, ddof=1)), cov=0.0, family_size=8,
                                 outcome=o, attribute=c, seed=seed, n_hi_A=int(hi[A].sum()), n_lo_A=int(lo[A].sum()),
                                 n_hi_B=int(hi[Bi].sum()), n_lo_B=int(lo[Bi].sum())))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(l3).to_csv(os.path.join(OUT, "l3.csv"), index=False)
    return dict(mode="gaps", gaps=pd.DataFrame(rows), inputs=[C.FULL, C.GT],
                notes=f"{N_SPLITS} splits 42/43, B={C.B} within each half, contrasts not read for an empty side: {skipped}")
