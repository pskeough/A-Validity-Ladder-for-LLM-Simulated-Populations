"""Relabel stored level-2 rows under the 3 Oct 2026 stop-2 bug fix, and validate the relabel.

The fix (78c/78e r3) labels a reading "reversed to attenuated" when it crosses more than one region
boundary and stays below the kept region; before, such a reading was "undetermined" and the frozen
stop 2 fired on it. For a contiguous interval the reading stays below kept exactly when the
interval's upper limit is below 0.75, so stored rows can be relabelled from ci_hi alone.

usage:
  python 97_stop2_relabel.py validate OLD_DIR NEW_DIR   relabel OLD_DIR's rows, compare with NEW_DIR's rerun
  python 97_stop2_relabel.py apply OLD_DIR OUT_DIR      write relabelled copies of OLD_DIR's rows
"""
import glob
import os
import sys

import pandas as pd

LOW = 0.75
NEW = "reversed to attenuated"
UNSTOPPED = ("r3_unstopped", "verdict_unstopped")


def relabel(d):
    d = d.copy()
    ucol = next(c for c in UNSTOPPED if c in d.columns)
    below = (d[ucol] == "undetermined") & (pd.to_numeric(d.ci_hi, errors="coerce") < LOW)
    d.loc[below, ucol] = NEW
    # stop 1 (no population gap) still stands; stop 2 no longer fires on a reading that excludes kept
    for c in ("verdict", "verdict_auditprec"):
        if c in d.columns:
            hit = below & d[c].isin(["reference too imprecise", "undetermined"])
            d.loc[hit, c] = NEW
    if "verdict_popstop" in d.columns:
        d.loc[below & (d.verdict_popstop == "undetermined"), "verdict_popstop"] = NEW
    if "verdict_conditional" in d.columns:
        d.loc[below & (d.verdict_conditional == "undetermined"), "verdict_conditional"] = NEW
    return d, int(below.sum())


def main():
    mode, a, b = sys.argv[1:4]
    files = sorted(glob.glob(os.path.join(a, "rep_*.csv")))
    tot, bad = 0, 0
    if mode == "apply":
        os.makedirs(b, exist_ok=True)
    for f in files:
        old = pd.read_csv(f, low_memory=False, keep_default_na=False, na_values=[""])
        new, n = relabel(old)
        tot += n
        if mode == "apply":
            new.to_csv(os.path.join(b, os.path.basename(f)), index=False)
            continue
        if not os.path.exists(os.path.join(b, os.path.basename(f))):
            continue
        ref = pd.read_csv(os.path.join(b, os.path.basename(f)), low_memory=False, keep_default_na=False,
                          na_values=[""])
        for c in [c for c in ("verdict", "verdict_auditprec", "verdict_popstop", "verdict_conditional",
                              *UNSTOPPED) if c in ref.columns]:
            m = new[c].astype(str).values != ref[c].astype(str).values
            if m.any():
                bad += int(m.sum())
                print(os.path.basename(f), c, int(m.sum()), "mismatches")
                print(pd.DataFrame({"relabel": new[c][m], "rerun": ref[c][m], "ci_lo": new.ci_lo[m],
                                    "ci_hi": new.ci_hi[m]}).head(5).to_string())
    print(f"{mode}: {len(files)} files, {tot} rows relabelled" + (f", {bad} mismatches" if mode == "validate" else ""))
    if mode == "validate" and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
