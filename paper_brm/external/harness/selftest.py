"""Check the harness against the paper's stored verdicts before any new dataset is run through it.
Every validation row must give the stored verdict (and, for Twin-2K, the stored label)."""
import sys

import harness

bad = 0
for a in ["val_opinionqa", "val_argyle", "val_twin2k", "val_bisbee"]:
    l2, _ = harness.run(a)
    n = (l2.verdict != l2.stored_verdict).sum()
    if "stored_label" in l2:
        n += (l2.label != l2.stored_label).sum()
    print(f"  {a}: {len(l2)} rows, {n} mismatches")
    bad += n
if "--quick" not in sys.argv:
    harness.run("val_synthetic")
print("SELFTEST", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
