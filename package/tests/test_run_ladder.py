"""Smoke tests of run_ladder on a small synthetic dataset."""
import numpy as np
import pandas as pd
import pytest

import validity_ladder as vl
from validity_ladder.synthetic import simulate_example

VERDICTS = {"pass", "fail", "unresolved", "not applicable", "not run"}
FAST = {"gate": {"n_boot": 20}, "level1": {"n_boot": 20}, "level4": {"n_boot": 20}}


@pytest.fixture(scope="module")
def data():
    sim, ref = simulate_example(n_ref=1500, personas_per_cell=6, draws=8, seed=2)
    return sim, ref, [c for c in sim.columns if c.startswith("item_")]


def test_full_ladder(data):
    sim, ref, items = data
    t, det = vl.run_ladder(sim, ref, items=items, by=["model", "framing"], ref_weight="w",
                           contrasts=[("sex", "F", "M"), ("income", "Low", "High")], r4="not applicable",
                           options=FAST, return_details=True)
    assert list(t.columns) == ["model", "framing", "rung", "verdict", "summary", "reason"]
    assert len(t) == 4 * 5
    assert set(t.verdict) <= VERDICTS
    assert set(t.rung) == set(vl.RUNGS)
    assert ("faithful", "plain", "level2") in det
    # the flat model has no general factor and misses the population's mean
    flat = t[t.model == "flat"].set_index(["framing", "rung"]).verdict
    assert (flat.xs("level4", level="rung") == "fail").all()
    assert (flat.xs("level3", level="rung") == "fail").all()
    # the faithful model is never ruled out at level 3 (its cell means are the reference's)
    faithful = t[t.model == "faithful"].set_index(["framing", "rung"]).verdict
    assert (faithful.xs("level3", level="rung") != "fail").all()


def test_skips_with_reasons(data):
    sim, ref, items = data
    one = sim[(sim.model == "faithful") & (sim.framing == "plain")]
    # no reference: only the gate runs, with a given reference SD
    t = vl.run_ladder(one, None, items=items, sd_ref=4.0, options=FAST).set_index("rung")
    assert t.loc["gate", "verdict"] in {"pass", "fail"}
    for r in ("level1", "level2", "level3", "level4"):
        assert t.loc[r, "verdict"] == "not run" and t.loc[r, "reason"]
    # scores only (no items): levels 1 and 4 not run; no shared attributes: levels 2 and 3 not applicable
    s = one.assign(score=one[items].sum(axis=1)).drop(columns=items + ["sex", "income"])
    rf = ref.assign(score=ref[items].sum(axis=1))
    t = vl.run_ladder(s, rf, score="score", ref_weight="w", options=FAST).set_index("rung")
    assert t.loc["level1", "verdict"] == "not run" and t.loc["level4", "verdict"] == "not run"
    assert t.loc["level2", "verdict"] == "not applicable" and t.loc["level3", "verdict"] == "not applicable"
    # one draw per persona: gate and level 1 not applicable
    d1 = one[one.draw == 1]
    t = vl.run_ladder(d1, ref, items=items, ref_weight="w", rungs=("gate", "level1"), options=FAST).set_index("rung")
    assert t.loc["gate", "verdict"] == "not applicable" and t.loc["level1", "verdict"] == "not applicable"


def test_thresholds_are_keywords(data):
    sim, ref, items = data
    one = sim[(sim.model == "faithful") & (sim.framing == "plain")]
    strict = vl.run_ladder(one, ref, items=items, ref_weight="w", rungs=("level3",),
                           options={"level3": {"tolerance_sd": 0.0001}})
    loose = vl.run_ladder(one, ref, items=items, ref_weight="w", rungs=("level3",),
                          options={"level3": {"tolerance_sd": 5.0}})
    assert strict.verdict.iloc[0] != "pass" and loose.verdict.iloc[0] == "pass"


def test_attribute_detection(data):
    sim, ref, items = data
    t, det = vl.run_ladder(sim[sim.model == "faithful"], ref, items=items, by="framing", ref_weight="w",
                           rungs=("level2",), return_details=True)
    c = det[("plain", "level2")]["contrasts"]
    # default contrasts: every level pair of each detected attribute (draw is excluded: varies within persona)
    assert set(c.attribute) == {"sex", "income"}
