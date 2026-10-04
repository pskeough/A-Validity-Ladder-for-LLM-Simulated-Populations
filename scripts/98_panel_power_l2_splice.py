"""Splice level-2 rows rerun under the 3 Oct 2026 stop-2 fix into the panel-power replicates.

90 was rerun with PP_SKIP_R4=1 into l2fix_2026-10-03/reps. R4 draws from its own seeds, so every row
that is not R4 must come out identical; this script checks that for each replicate (proving the
generator stream is unchanged), checks that the rerun's level-2 rows differ from the stored ones only
in the readings the fix relabels, then writes the stored replicate with its level-2 rows replaced.

usage: python 98_panel_power_l2_splice.py
Reads panel_power/reps_prefix_2026-10-03 (the stored replicates) and l2fix_2026-10-03/reps;
writes panel_power/reps.
"""
import glob
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PP = os.path.join(BASE, "paper_brm", "analysis_brm", "panel_power")
OLD = os.path.join(PP, os.environ.get("PP_OLD", "reps_prefix_2026-10-03"))
FIX = os.path.join(PP, "l2fix_2026-10-03", "reps")
OUT = os.path.join(PP, os.environ.get("PP_OUT", "reps"))
KEY = ["rep", "grid", "type", "rung", "rule", "model", "unit"]
FIXED = "reversed to attenuated"


def S(x):
    return x.astype(object).where(x.notna(), "").astype(str).to_numpy()


def read(f):
    return pd.read_csv(f, low_memory=False, keep_default_na=False, na_values=[""])


def main():
    os.makedirs(OUT, exist_ok=True)
    tot_relab, tot_model, trans = 0, 0, {}
    for f in sorted(glob.glob(os.path.join(OLD, "rep_*.csv"))):
        name = os.path.basename(f)
        old, new = read(f), read(os.path.join(FIX, name))
        # every row outside R4 identical (generator alignment); the level-4 model verdicts need R4,
        # so the rerun does not write them and the stored ones are kept
        def no_r4(d):
            return d[~d.rung.str.startswith("L4-R4") & ~((d.rung == "L4") & (d.unit == "model"))]
        o = no_r4(old).reset_index(drop=True)
        n = no_r4(new).reset_index(drop=True)
        assert len(o) == len(n), (name, len(o), len(n))
        assert (S(o[KEY]) == S(n[KEY])).all(), name
        nonl2 = o.rung != "L2"
        skip = {"r4_seconds"} | {c for c in o.columns if c.startswith("R4") or c.startswith("steps_")}
        # gate rows: early runners stored verdict, pass_min and k under the pre-freeze gate rule; 90b
        # re-reads the gate from the stored SE(30), so only the estimates (SE, phi, components) must match
        gate_rule = {"verdict", "pass_min", "pass_rec", "pass_single"} | {c for c in o.columns if c.startswith("k_")}
        for c in o.columns:
            if c in skip or c not in n.columns:
                continue
            rows = nonl2 & ~((o.rung == "gate") & (c in gate_rule))
            a, b = o.loc[rows, c], n.loc[rows, c]
            num = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
            same = np.allclose(a, b, equal_nan=True) if num else (S(a) == S(b)).all()
            assert same, (name, c)
        # level 2: identical intervals; verdicts move only to the fixed label (or a model verdict to fail)
        l2 = ~nonl2
        for c in ("ratio", "ci_lo", "ci_hi", "k_rel"):
            assert np.allclose(o.loc[l2, c], n.loc[l2, c], equal_nan=True), (name, c)
        # verdict_unstopped and verdict_popstop: only the fixed label may appear. The stored `verdict`
        # column was written by early runners under the pre-freeze conditional stop and by the rerun
        # under the frozen rule, so its changes are logged, not asserted (90b reads the frozen model
        # verdict from verdict_unstopped and k_rel, the "keptable" rule).
        for c in ("verdict_unstopped", "verdict_popstop"):
            d = l2 & pd.Series(S(o[c]) != S(n[c]), index=o.index)
            assert (n.loc[d, c] == FIXED).all(), (name, c, n.loc[d, c].unique())
            assert (o.unit[d] != "model").all(), (name, c)
            if c == "verdict_unstopped":
                tot_relab += int(d.sum())
        d = l2 & pd.Series(S(o.verdict) != S(n.verdict), index=o.index)
        for a, b, u in zip(S(o.verdict[d]), S(n.verdict[d]), o.unit[d]):
            trans[(u == "model", a, b)] = trans.get((u == "model", a, b), 0) + 1
        tot_model += int((d & (o.unit == "model")).sum())
        # replace the level-2 rows in place, keeping the stored row order
        out = old.copy()
        io, inew = np.flatnonzero(old.rung == "L2"), np.flatnonzero(new.rung == "L2")
        assert len(io) == len(inew), name
        assert (S(old.iloc[io][KEY]) == S(new.iloc[inew][KEY])).all(), name
        for c in ("verdict", "verdict_unstopped", "verdict_popstop", "detail"):
            out.loc[out.index[io], c] = new.iloc[inew][c].values
        out.to_csv(os.path.join(OUT, name), index=False)
    print(f"spliced; {tot_relab} unstopped contrast readings relabelled; {tot_model} stored model verdicts changed")
    log = pd.DataFrame([dict(model_row=k[0], stored=k[1], rerun=k[2], n=v) for k, v in trans.items()])
    log.sort_values("n", ascending=False).to_csv(os.path.join(PP, "98_verdict_column_changes.csv"), index=False)
    print(log.sort_values("n", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
