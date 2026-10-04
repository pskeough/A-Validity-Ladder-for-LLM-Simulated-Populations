"""Where step 0 flips: the relative half-width k = t*se/gamma at which a noise-free faithful simulator
stops being read kept, against the level-2 stop at k_max = 1/4. Writes out/k_threshold.csv (4 Oct 2026)."""
import csv
import os
from scipy.stats import norm
from validity_ladder.level2 import certifiable
os.makedirs("out", exist_ok=True)
rows = []
for m in (1, 2, 5, 9, 20, 40):
    z = norm.ppf(1 - 0.05 / m / 2)
    lo, hi = 0.1, 0.3
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if certifiable(1.0, mid / z, family_size=m) else (lo, mid)
    rows.append(dict(family_size=m, k_certifiable_max=round(lo, 4), k_stop=0.25))
with open("out/k_threshold.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(rows)
