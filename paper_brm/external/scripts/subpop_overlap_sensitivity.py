"""Re-read SubPOP at level 2 without the questions whose item (key stem) appears in SubPOP's training set.

Reads harness/results/subpop2025/training_overlap.csv (written by subpop_overlap.py), drops every question
with stem_in_train, and reads the remaining gaps through the harness exactly as the release reading does.
Output: harness/results/subpop2025/l2_contrasts_no_overlap.csv and a summary on stdout.
"""
import os
import sys

import pandas as pd

EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
HARNESS = os.path.join(EXT, "harness")
sys.path.insert(0, HARNESS)

import harness  # noqa: E402
from adapters import subpop2025 as ad  # noqa: E402

ov = pd.read_csv(os.path.join(HARNESS, "results", "subpop2025", "training_overlap.csv"))
drop = set(ov[ov.stem_in_train].qkey)
ad.DROP_CONSERVATIVE = lambda q: q in drop
gaps = ad.load()["gaps"]
gaps = gaps[gaps.family.str.endswith(": conservative")].copy()
gaps["family"] = gaps.family.str.replace(": conservative", ": no training overlap", regex=False)
l2 = harness.read_gaps(gaps)
out = os.path.join(HARNESS, "results", "subpop2025", "l2_contrasts_no_overlap.csv")
l2.to_csv(out, index=False)
cols = [c for c in ("family", "contrast", "n_questions", "ratio", "ci_lo", "ci_hi", "verdict") if c in l2.columns]
print(f"dropped {len(drop)} questions: {sorted(drop)}")
print(l2[cols].to_string())
