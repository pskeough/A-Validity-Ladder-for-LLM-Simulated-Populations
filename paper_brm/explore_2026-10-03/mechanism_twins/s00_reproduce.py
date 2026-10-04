"""Step 0. Reproduce the existing Republican - Democrat gaps on the 8 certifiable items for every
specification from the raw release, and compare with results/twin2k/level2_contrasts.csv
(config == headline). Prints and writes 00_reproduction.csv and 00_gaps_all_specs.csv.
"""
import os

import numpy as np
import pandas as pd

from common import *

sims, h4, h13, demo = load_all()
R = demo.party == "Republican"
D = demo.party == "Democrat"

rows = []
for k, S in sims.items():
    for it in ITEMS:
        d = pd.DataFrame(dict(s=S[it], y=h4.loc[S.index, it]), index=S.index).dropna()
        a = d.index.intersection(demo.index[R])
        b = d.index.intersection(demo.index[D])
        g = d.s[a].mean() - d.s[b].mean()
        p = d.y[a].mean() - d.y[b].mean()
        rows.append(dict(run=k, outcome=it, n_a=len(a), n_b=len(b), g=g, p=p, ratio=g / p))
mine = pd.DataFrame(rows)
mine.to_csv(os.path.join(OUT, "00_gaps_all_specs.csv"), index=False)

ref = pd.read_csv(os.path.join(RES, "level2_contrasts.csv"))
ref = ref[(ref.config == "headline") & (ref.contrast == "Republican - Democrat") & ref.outcome.isin(ITEMS)]
ref = ref[["run", "outcome", "n_a", "n_b", "g", "p", "ratio"]]
m = mine.merge(ref, on=["run", "outcome"], suffixes=("", "_ref"))
assert len(m) == len(mine) == 13 * 8 + 8 - 8 + 0 or len(m) == len(mine), (len(m), len(mine))
for c in ("g", "p", "ratio"):
    m[c + "_diff"] = (m[c] - m[c + "_ref"]).abs()
m["n_equal"] = (m.n_a == m.n_a_ref) & (m.n_b == m.n_b_ref)
m.to_csv(os.path.join(OUT, "00_reproduction.csv"), index=False)

print("rows compared:", len(m), "of", len(mine))
print("max |diff| g, p, ratio:", m.g_diff.max(), m.p_diff.max(), m.ratio_diff.max())
print("all n equal:", bool(m.n_equal.all()))
d = m[m.run == HEADLINE].sort_values("outcome")
print(d[["outcome", "n_a", "n_b", "g", "p", "ratio", "ratio_ref"]].to_string(index=False))
dd = m[m.run == HEADLINE].ratio
print(f"default twins ratio range: {dd.min():.2f} to {dd.max():.2f} (existing: 1.45 to 2.30)")
cs = m[m.run == CEILING].ratio
print(f"ceiling ratio range: {cs.min():.2f} to {cs.max():.2f} (existing: 0.93 to 1.04)")
