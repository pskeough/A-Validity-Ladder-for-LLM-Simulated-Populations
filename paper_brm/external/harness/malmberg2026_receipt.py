"""Receipt for adapters/malmberg2026.py: full-sample AMCEs recomputed from the released raw files against the
t0 column of the authors' released bootstrap files (the data behind Figures 5a, 6a, 7a).
Output: results/malmberg2026/receipt.csv (item, ours, published, source_location, abs_diff)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adapters import malmberg2026 as M  # noqa: E402

BOOT = {"GPT-4o-mini": ("GPTsiliconboot", "GPTrealboot"),
        "Llama 3.2": ("llamasiliconboot", "llamarealboot"),
        "Magistral-small-2506": ("Mistralsiliconboot", "Mistralrealboot")}
# row order of the authors' bootstrap files -> (attribute, index among the non-baseline levels)
ORDER = [("Abortion", 0), ("Abortion", 1), ("Abortion", 2), ("COL", 0), ("COL", 1), ("Crime", 0), ("Crime", 1),
         ("Guns", 0), ("Guns", 1), ("LGBTQ", 0), ("LGBTQ", 1), ("LGBTQ", 2), ("Schools", 0), ("Schools", 1)]

rows = []
for label, fn, ch in M.MODELS:
    m = M.build(fn, ch)
    ones = np.ones((1, m["R"]))
    est = {"silicon": M.wls(m["S"], m["ts"], ones)[0], "live": M.wls(m["S"], m["th"], ones)[0]}
    for sample, bootfile in zip(("silicon", "live"), BOOT[label]):
        pub = pd.read_csv(os.path.join(M.RAW, bootfile))
        assert len(pub) == 14
        for pos, (attr, k) in enumerate(ORDER):
            lv = next(lvs for a, _, lvs in M.ATTRS if a == attr)[1:][k]
            ours = est[sample][M.level_index(attr, lv)]
            rows.append(dict(item=f"{label} | {sample} | AMCE {attr}: {lv}", ours=float(ours),
                             published=float(pub.t0.iloc[pos]),
                             source_location=f"{bootfile} row {pos + 1} (name '{pub.name.iloc[pos]}'), column t0",
                             abs_diff=abs(float(ours) - float(pub.t0.iloc[pos]))))
out = pd.DataFrame(rows)
os.makedirs(os.path.join(HERE, "results", "malmberg2026"), exist_ok=True)
out.to_csv(os.path.join(HERE, "results", "malmberg2026", "receipt.csv"), index=False)
print(out.groupby(out.item.str.split(" \\| ").str[:2].str.join(" | ")).abs_diff.agg(["count", "max"]))
print("overall max abs diff", out.abs_diff.max())
print(out.head(14).to_string())
