"""Level 4 of the Validity Ladder for the Mercer et al. (2025) HEXACO agents against the Open Psychometrics
human reference, per model and domain (facet-level one-factor models; see hexaco2025_lib.py).

Usage: python hexaco2025_l4.py [B] [variants...]       B default 1000; variants default all four.
Writes results/hexaco2025/l4.csv (all variants; variant 'primary' is the reading) and
l4_item_agents.csv (agents-only item-level description: polychorics, one-factor ULS, absolute R1).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hexaco2025_lib as L
import hexaco2025_l4core as C
from validity_ladder.level4 import eig_desc, general_factor, one_factor_uls, polychoric_matrix

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
VARIANTS = sys.argv[2:] or ["primary", "S1_sub4ref", "S2_nofilter", "S3_GB"]
OUT_L4 = os.path.join(L.OUT, "l4.csv" if B == 1000 else f"l4_B{B}.csv")
os.makedirs(L.OUT, exist_ok=True)

keying = L.ipip_keying()
hum = L.load_human_raw()
agents = {m: L.facet_scores_agents(L.load_agents(m)) for m in L.MODELS}
for m, F in agents.items():
    assert not F.isna().any().any(), m

rows = []
for v in VARIANTS:
    t0 = time.time()
    mode = "sub4" if v == "S1_sub4ref" else "full10"
    items, n_ref = C.human_items("primary" if v == "S1_sub4ref" else v, keying, hum)
    ref = C.Reference(items, mode=mode, n_boot=B, label=v)
    print(f"[{v}] reference n = {n_ref}, built in {time.time() - t0:.0f}s", flush=True)
    for m, F in agents.items():
        for s in ref.sets:
            r = C.read_set(F, ref, s)
            rows.append(dict(model=m, variant=v, ref_n=n_ref, **r))
        print(f"[{v}] {m} done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT_L4, index=False)

# ---- item-level, agents only (description: no reference at item level)
if VARIANTS and VARIANTS[0] == "primary" and B == 1000:
    keys = L.pir_item_keys()
    rec = []
    for m in L.MODELS:
        k = L.keyed(L.load_agents(m))
        for d in L.DOMAINS:
            cols = sorted(i for f, lst in keys.items() if L.PIR_KEY_DOMAIN[f] == d for i, _ in lst)
            X = (k[cols].to_numpy(float) - 1).astype(int)  # 0..4
            R = polychoric_matrix(X, n_categories=5)
            lam = one_factor_uls(R)
            ev = eig_desc(R)
            rec.append(dict(model=m, domain=d, n_items=len(cols), n_agents=len(X), load_min=float(lam.min()),
                            load_median=float(np.median(lam)), n_loadings_below_030=int((lam < 0.30).sum()),
                            ev1=float(ev[0]), ev2=float(ev[1]), ev_ratio=float(ev[0] / ev[1]),
                            R1_absolute=bool(general_factor(lam, ev)),
                            loadings=";".join(f"{v:.2f}" for v in lam), items_1based=";".join(str(c + 1) for c in cols)))
    pd.DataFrame(rec).to_csv(os.path.join(L.OUT, "l4_item_agents.csv"), index=False)
    print("item-level agents done")
print("done")
