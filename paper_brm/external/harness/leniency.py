"""Is the level-2 gate too hard? Re-read every stored contrast of every dataset under easier rules and
count family verdicts for the language models and for the faithful controls.

Rules: frozen (kept 0.75-1.25, Bonferroni over the family); wide band (kept 0.5-1.5); no multiplicity
correction (family_size 1); both. Stops (no population gap; k > band half-width share) follow the
package: with a wider band, k_max rises with it (the band's half-width relative to 1).
Output: results/LENIENCY.csv
"""
import glob
import os

import numpy as np
import pandas as pd

import validity_ladder as vl
from validity_ladder.thresholds import L2_BOUNDS

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = {
    "frozen": (L2_BOUNDS, True),
    "wide band 0.5-1.5": ((-0.25, 0.25, 0.5, 1.5), True),
    "no multiplicity": (L2_BOUNDS, False),
    "wide band, no multiplicity": ((-0.25, 0.25, 0.5, 1.5), False),
}
FOLDERS = ["val_opinionqa", "val_argyle", "val_twin2k", "huang2025", "malmberg2026", "twin2k_mega_bystudy",
           "cummins2025", "val_bisbee", "subpop2025", "opinionqa_ideal", "argyle_ideal", "huang2025_ideal",
           "huang2025_ideal_k", "cummins2025_splithalf"]


def arm(folder, fam):
    if fam.startswith("human_split_half") or fam.startswith("human_retest") or "_ideal" in folder or "splithalf" in folder:
        return "faithful control"
    return "models"


rows = []
for folder in FOLDERS:
    d = pd.read_csv(os.path.join(HERE, "results", folder, "l2_contrasts.csv"), low_memory=False)
    for c in ("df_g", "df_gamma", "cov"):
        if c not in d:
            d[c] = np.inf if c != "cov" else 0.0
    d["df_g"] = d.df_g.fillna(np.inf)
    d["df_gamma"] = d.df_gamma.fillna(np.inf)
    d["cov"] = d["cov"].fillna(0.0)
    for rule, (bounds, bonf) in RULES.items():
        verdicts = []
        with np.errstate(all="ignore"):
            for r in d.itertuples():
                o = vl.level2_contrast(r.g, r.se_g, r.gamma, r.se_gamma, df_g=r.df_g, df_gamma=r.df_gamma,
                                       cov=r.cov, family_size=int(r.family_size) if bonf else 1, bounds=bounds)
                verdicts.append(o["verdict"])
        d["v"] = verdicts
        for fam, g in d.groupby("family"):
            live = [v for v in g.v if v != "not read"]
            rows.append(dict(folder=folder, arm=arm(folder, fam), rule=rule, family=fam,
                             verdict=vl.level2_model_verdict(live)))
    print(folder, "done", flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "results", "LENIENCY_families.csv"), index=False)
tab = res.groupby(["folder", "arm", "rule"]).verdict.value_counts().unstack(fill_value=0).reset_index()
tab.to_csv(os.path.join(HERE, "results", "LENIENCY.csv"), index=False)
print(tab.to_string(index=False))
print()
tot = res.groupby(["arm", "rule"]).verdict.value_counts().unstack(fill_value=0)
tot["families"] = tot.sum(axis=1)
tot.reset_index().to_csv(os.path.join(HERE, "results", "LENIENCY_totals.csv"), index=False)
print(tot)
