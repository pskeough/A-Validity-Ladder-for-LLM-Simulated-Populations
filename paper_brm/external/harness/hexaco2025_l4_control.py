"""Split-half control for level 4 of the HEXACO 2025 adapter.

The filtered Open Psychometrics respondents are split at random into halves A and B (seed 20261003).
Half B is the reference. Half A is treated as the simulated population (one draw per respondent, the
respondent is the persona) and read per domain exactly as the agents are. This shows what a faithful
population scores.

Modes (sim side; the reference is always half B with facet = mean of ten keyed items):
  c10_full   half A, ten-item facets, all of A
  c10_n310   a random 310 of A (the agents' n), ten-item facets
  c4_full    half A, four-item facets (one fixed random four of the ten per facet), all of A
  c4_n310    a random 310 of A, four-item facets  <- the agents' situation (n = 310, four-item facets)
The single-split rows (B = 1000) go to l4_control.csv. Then R random 310-subsamples of A with fresh
four-item draws (mode c4_n310_rep and c10_n310_rep, B = 500 each) go to l4_control_repeats.csv, with the
share of pass / unresolved / fail per domain in l4_control_summary.csv.

Usage: python hexaco2025_l4_control.py [R]    (R default 60; 0 skips the repeats)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hexaco2025_lib as L
import hexaco2025_l4core as C

R = int(sys.argv[1]) if len(sys.argv) > 1 else 60
os.makedirs(L.OUT, exist_ok=True)

keying = L.ipip_keying()
hum = L.load_human_raw()
items, n_all = C.human_items("primary", keying, hum)
rng = np.random.default_rng([L.SEED, 7])
perm = rng.permutation(n_all)
A_idx, B_idx = perm[: n_all // 2], perm[n_all // 2:]
print(f"filtered n = {n_all}; half A {len(A_idx)}, half B {len(B_idx)}", flush=True)
items_B = {p: m[B_idx] for p, m in items.items()}
items_A = {p: m[A_idx] for p, m in items.items()}
ipip_to_pir = {v: k for k, v in L.IPIP_OF.items()}


def facets_from(it, idx, four, r):
    """DataFrame of PI-R-named facet scores for the respondents idx of the item dict it."""
    cols = {}
    for p, m in it.items():
        mm = m[idx]
        if four:
            sel = r.choice(m.shape[1], 4, replace=False)
            mm = mm[:, sel]
        cols[ipip_to_pir[p]] = mm.mean(1)
    return pd.DataFrame(cols)


def run(ref, tag, F, rows, extra=None):
    for s in ref.sets:
        out = C.read_set(F, ref, s)
        rows.append(dict(mode=tag, ref_n=len(B_idx), **(extra or {}), **out))


t0 = time.time()
ref = C.Reference(items_B, mode="full10", n_boot=1000, label="halfB")
print(f"half-B reference built ({time.time() - t0:.0f}s)", flush=True)
rows = []
r4 = np.random.default_rng([L.SEED, 8])
fixed4 = np.random.default_rng([L.SEED, 9])
sub310 = np.random.default_rng([L.SEED, 10]).choice(len(A_idx), 310, replace=False)
F_A10 = facets_from(items_A, np.arange(len(A_idx)), False, None)
F_A4 = facets_from(items_A, np.arange(len(A_idx)), True, fixed4)  # one fixed four-item subset per facet
for tag, F in [("c10_full", F_A10), ("c10_n310", F_A10.iloc[sub310]), ("c4_full", F_A4), ("c4_n310", F_A4.iloc[sub310])]:
    run(ref, tag, F, rows)
    print(f"{tag} done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(L.OUT, "l4_control.csv"), index=False)

if R > 0:
    ref5 = C.Reference(items_B, mode="full10", n_boot=500, label="halfB_500")
    reps = []
    rr = np.random.default_rng([L.SEED, 11])
    for k in range(R):
        sub = rr.choice(len(A_idx), 310, replace=False)
        F10 = facets_from(items_A, sub, False, None)
        F4 = facets_from(items_A, sub, True, rr)
        run(ref5, "c10_n310_rep", F10, reps, dict(rep=k))
        run(ref5, "c4_n310_rep", F4, reps, dict(rep=k))
        if k % 5 == 4:
            print(f"rep {k + 1}/{R} ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(reps).to_csv(os.path.join(L.OUT, "l4_control_repeats.csv"), index=False)
    d = pd.DataFrame(reps)
    d.to_csv(os.path.join(L.OUT, "l4_control_repeats.csv"), index=False)
    summ = []
    for (mode, dom), g in d.groupby(["mode", "domain"], sort=False):
        summ.append(dict(mode=mode, domain=dom, reps=len(g),
                         R1_pass=float(g.R1.mean()),
                         R2_pass=float((g.R2 == "pass").mean()), R2_unresolved=float((g.R2 == "unresolved").mean()),
                         R2_fail=float((g.R2 == "fail").mean()),
                         level4_pass=float((g.level4 == "pass").mean()), level4_unresolved=float((g.level4 == "unresolved").mean()),
                         level4_fail=float((g.level4 == "fail").mean()),
                         congruence_median=float(g.congruence.median()), rmsd_median=float(g.rmsd.median())))
    pd.DataFrame(summ).to_csv(os.path.join(L.OUT, "l4_control_summary.csv"), index=False)
print("done", flush=True)
