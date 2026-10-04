"""Rewrite every results/<name>/l2_models.csv from its stored l2_contrasts.csv with the current model
table (step 0: certifiable contrasts and the four-way reading). Gaps, SEs and verdicts are not
recomputed; only the family table is. Usage: python refresh.py"""
import glob
import os

import pandas as pd

import harness

for f in sorted(glob.glob(os.path.join(harness.OUT_ROOT, "*", "l2_contrasts.csv"))):
    l2 = harness.add_certifiable(pd.read_csv(f, low_memory=False))
    l2.to_csv(f, index=False)
    m = harness.model_table(l2)
    m.to_csv(os.path.join(os.path.dirname(f), "l2_models.csv"), index=False)
    print(os.path.basename(os.path.dirname(f)), m.reading.value_counts().to_dict(), flush=True)
