"""Verdict logic on synthetic inputs: every reading each rung can return."""
import numpy as np
import pandas as pd
import pytest

import validity_ladder as vl
from validity_ladder.level2 import _r3cov


# ------------------------------------------------------------------------------------------ gate
def _draws(s2p, s2r, n_p=60, k=30, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, np.sqrt(s2p), n_p)
    y = np.repeat(mu, k) + rng.normal(0, np.sqrt(s2r), n_p * k)
    return y, np.repeat(np.arange(n_p), k)


def test_gate_pass_and_fail():
    y, p = _draws(4.0, 1.0)
    r = vl.gate(y, p, sd_ref=4.0, n_boot=50)
    assert r["verdict"] == "pass" and r["rule_met"] == "recommended" and r["phi_k"] > 0.9
    assert r["se_k_lo"] <= r["se_k"] <= r["se_k_hi"]
    y, p = _draws(1.0, 900.0, k=5)
    r = vl.gate(y, p, sd_ref=4.0, n_boot=0)
    assert r["verdict"] == "fail" and r["rule_met"] == "none"
    assert r["k_min"] == np.ceil(r["s2_r"] / (0.25 * 4.0) ** 2)   # k needed = ceil(s2_r / t^2)


def test_gate_k_rules():
    assert vl.k_for_se(2.0285, 0.5) == 9
    assert vl.k_for_phi(2.1354, 2.0285, 0.80) == 4
    assert vl.k_for_phi(0.0, 1.0, 0.8) == np.inf
    assert vl.phi_k(-1.0, 1.0, 30) == 0.0           # negative component truncated
    with pytest.raises(ValueError):
        vl.gate([1, 2, 3], [0, 1, 2], sd_ref=1.0)    # no repeated draws


# --------------------------------------------------------------------------------------- level 1
def _r1(mlo, mhi, olo, ohi):
    return dict(misfit_ratio_ci90_lo=mlo, misfit_ratio_ci90_hi=mhi, overfit_ratio_ci90_lo=olo, overfit_ratio_ci90_hi=ohi)


@pytest.mark.parametrize("r, want", [
    (_r1(0.8, 1.3, 0.9, 1.2), "pass"),
    (_r1(0.0, 0.2, 0.0, 0.1), "fail: compressed (both tails in deficit)"),
    (_r1(1.6, 2.4, 0.9, 1.1), "fail: excess misfit"),
    (_r1(0.9, 1.2, 0.0, 0.3), "fail: overfit deficit"),
    (_r1(1.2, 1.9, 0.9, 1.1), "undetermined"),
])
def test_level1_readings(r, want):
    assert vl.read_verdict(r, 1.5)[2] == want


def test_resolved_departure_and_tau():
    assert vl.resolved_departure(1.2, 1.4) == 1.2
    assert vl.resolved_departure(0.5, 0.8) == pytest.approx(1.25)
    assert vl.resolved_departure(0.9, 1.3) == 1.0
    assert vl.tau_from_intervals([1.6], [1.9], [0.9], [1.1]) == (2.0, 1.6)
    assert vl.tau_from_intervals([1.1], [1.2], [0.9], [1.1])[0] == 1.5          # floor
    assert vl.tau_from_intervals([3.5], [4.0], [0.9], [1.1])[0] == np.inf


# --------------------------------------------------------------------------------------- level 2
def test_level2_kept():
    o = vl.level2_contrast(1.0, 0.03, 1.0, 0.02, df_g=200, df_gamma=200)
    assert o["verdict"] == "kept" and o["label"] == "kept"
    assert 0.75 < o["ci_lo"] < 1 < o["ci_hi"] < 1.25


@pytest.mark.parametrize("g, verdict", [(-1.0, "reversed"), (0.0, "missing"), (0.5, "attenuated"),
                                        (2.0, "steepened")])
def test_level2_not_kept(g, verdict):
    o = vl.level2_contrast(g, 0.03, 1.0, 0.02, df_g=200, df_gamma=200)
    assert o["verdict"] == verdict and o["label"] == "not kept"


def test_level2_unresolved():
    o = vl.level2_contrast(1.0, 0.4, 1.0, 0.02, df_g=50, df_gamma=200)
    assert o["verdict"] == "undetermined" and o["label"] == "unresolved"
    o = vl.level2_contrast(0.75, 0.05, 1.0, 0.01, df_g=200, df_gamma=200)
    assert o["verdict"] == "attenuated or kept" and o["label"] == "unresolved"


def test_level2_reversed_to_attenuated_excludes_kept():
    """Stop-2 bug fix (3 Oct 2026): a reading spanning reversed..attenuated excludes kept."""
    o = vl.level2_contrast(0.1, 0.25, 1.0, 0.01, df_g=200, df_gamma=200)
    assert o["reading"] == "reversed to attenuated"
    assert o["label"] == "not kept"
    assert vl.level2_model_verdict([o["verdict"]]) == "fail"
    # with an imprecise reference the reading still stands (it excludes kept)
    o = vl.level2_contrast(0.1, 0.25, 1.0, 0.2, df_g=200, df_gamma=200)
    assert o["stop_population"] and o["verdict"] == "reversed to attenuated"


def test_level2_stops_not_read():
    o = vl.level2_contrast(0.5, 0.1, 0.1, 0.2, df_g=50, df_gamma=100)
    assert o["verdict"] == "no population gap" and o["label"] == "not read"
    o = vl.level2_contrast(1.0, 0.05, 1.0, 0.3, df_g=100, df_gamma=100)      # k = t se / |gamma| > 1/4
    assert o["k_rel_halfwidth"] > 0.25
    assert o["verdict"] == "reference too imprecise" and o["label"] == "not read"
    o = vl.level2_contrast(-1.0, 0.05, 1.0, 0.3, df_g=100, df_gamma=100)     # a reading that excludes kept stands
    assert o["stop_population"] and o["verdict"] == "reversed" and o["label"] == "not kept"


def test_level2_bonferroni_widens():
    a = vl.level2_contrast(1.0, 0.1, 1.0, 0.05, df_g=40, df_gamma=100)
    b = vl.level2_contrast(1.0, 0.1, 1.0, 0.05, df_g=40, df_gamma=100, family_size=7)
    assert b["interval_level"] == pytest.approx(1 - 2 * 0.05 / 7)
    assert b["ci_lo"] < a["ci_lo"] and b["ci_hi"] > a["ci_hi"]


def test_fieller_numeric_equals_closed_form():
    """With infinite df and no covariance the numeric inversion (78c) equals the closed form (78e)."""
    for g, se_g, p, se_p in [(0.8, 0.1, 1.0, 0.05), (-0.3, 0.2, -0.6, 0.1), (2.1, 0.3, 1.5, 0.2)]:
        a = vl.level2_contrast(g, se_g, p, se_p)
        b = _r3cov(g, se_g, p, se_p, 0.0, np.inf, 0.05, np.array(vl.thresholds.L2_BOUNDS), vl.thresholds.L2_LABELS)
        assert a["ci_lo"] == pytest.approx(b["ci_lo"], abs=1e-9)
        assert a["ci_hi"] == pytest.approx(b["ci_hi"], abs=1e-9)
        assert a["verdict"] == b["verdict"]


@pytest.mark.parametrize("verdicts, want", [
    (["kept", "kept", "no population gap"], "pass"),
    (["kept", "attenuated or kept", "reference too imprecise"], "unresolved"),
    (["kept", "steepened"], "fail"),
    (["kept", "reversed or missing"], "fail"),
    (["undetermined", "kept"], "unresolved"),
    (["no population gap", "reference too imprecise"], "unresolved"),
])
def test_level2_model_verdict(verdicts, want):
    assert vl.level2_model_verdict(verdicts) == want


def test_three_way():
    assert vl.three_way("kept", 0.8, 1.2) == "kept"
    assert vl.three_way("steepened", 1.3, 1.9) == "not kept"
    assert vl.three_way("kept or steepened", 0.9, 1.4) == "unresolved"
    assert vl.three_way("reference too imprecise", 0.9, 1.1) == "not read"
    assert vl.three_way("undetermined", np.nan, np.nan) == "unresolved"


def test_level2_estimators_on_synthetic_long_data():
    """A simulated population built from the reference's own cell means keeps both gaps."""
    rng = np.random.default_rng(3)
    n = 40000
    ref = pd.DataFrame(dict(sex=rng.choice(["F", "M"], n), inc=rng.choice(["Low", "Mid", "High"], n)))
    ref["score"] = 5 + 2.0 * (ref.sex == "F") + 3.0 * (ref.inc == "Low") + rng.normal(0, 2, n)
    rows = []
    pid = 0
    for s in ("F", "M"):
        for i in ("Low", "Mid", "High"):
            for _ in range(30):
                pid += 1
                m = 5 + 2.0 * (s == "F") + 3.0 * (i == "Low") + rng.normal(0, 0.3)
                for d in range(10):
                    rows.append(dict(persona=pid, sex=s, inc=i, score=m + rng.normal(0, 1)))
    sim = pd.DataFrame(rows)
    r = vl.level2(sim, ref, [("sex", "F", "M"), ("inc", "Low", "High")], attributes=["sex", "inc"])
    t = r["contrasts"].set_index("contrast")
    assert t.loc["F minus M", "g"] == pytest.approx(2.0, abs=0.3)
    assert t.loc["F minus M", "gamma"] == pytest.approx(2.0, abs=0.1)
    assert t.loc["F minus M", "n_pairs"] == 3                     # strata of income
    assert list(t.label) == ["kept", "kept"] and r["verdict"] == "pass"
    assert r["family_size"] == 2


# --------------------------------------------------------------------------------------- level 3
@pytest.mark.parametrize("resid, want", [(0.1, "pass"), (3.0, "fail"), (-3.0, "fail"), (1.8, "unresolved")])
def test_level3_tost(resid, want):
    o = vl.level3_tost(resid, 0.01, 0.04, 100, 2.0)
    assert o["verdict"] == want
    assert o["ci90_lo"] < resid < o["ci90_hi"]


def test_level3_long_data_pass_and_fail():
    rng = np.random.default_rng(4)
    n = 20000
    ref = pd.DataFrame(dict(sex=rng.choice(["F", "M"], n, p=[0.6, 0.4]), inc=rng.choice(["Low", "High"], n),
                            psu=rng.integers(1, 3, n), stratum=rng.integers(1, 30, n), w=rng.uniform(0.5, 2, n)))
    ref["score"] = 4 + 1.5 * (ref.sex == "F") + 2.0 * (ref.inc == "Low") + rng.normal(0, 3, n)
    rows = []
    for pid, (s, i) in enumerate([(s, i) for s in ("F", "M") for i in ("Low", "High")] * 5):
        mu = 4 + 1.5 * (s == "F") + 2.0 * (i == "Low")
        for d in range(20):
            rows.append(dict(persona=pid, sex=s, inc=i, score=mu + rng.normal(0, 1)))
    sim = pd.DataFrame(rows)
    good = vl.level3(sim, ref, ["sex", "inc"], weight="w", psu="psu", stratum="stratum", prevalence_cut=8)
    assert good["verdict"] == "pass"
    assert good["tolerance"] == pytest.approx(0.5 * good["sd_ref"])
    assert set(good["groups"].group) == {"Overall", "sex=F", "sex=M", "inc=Low", "inc=High"}
    assert good["prevalence"] is not None and "ratio_ci90_lo" in good["prevalence"]
    bad = vl.level3(sim.assign(score=sim.score + 6), ref, ["sex", "inc"], weight="w", psu="psu", stratum="stratum")
    assert bad["verdict"] == "fail"
    assert (bad["groups"].verdict == "fail").all()


@pytest.mark.parametrize("v, want", [(["pass", "pass"], "pass"), (["pass", "fail"], "fail"),
                                     (["pass", "unresolved"], "unresolved")])
def test_level3_model_verdict(v, want):
    assert vl.level3_model_verdict(v) == want


# --------------------------------------------------------------------------------------- level 4
@pytest.mark.parametrize("args, want", [((0.97, 0.99, 0.03, 0.08), "pass"), ((0.80, 0.93, 0.03, 0.08), "fail"),
                                        ((0.97, 0.99, 0.12, 0.20), "fail"), ((0.93, 0.99, 0.03, 0.08), "unresolved"),
                                        ((0.97, 0.99, 0.05, 0.12), "unresolved")])
def test_r2_verdict(args, want):
    assert vl.r2_verdict(*args) == want


def test_r1_relative_to_reference():
    lam_ref = np.array([0.6, 0.7, 0.28, 0.65])         # one reference loading below .30 (Twin-2K item 35)
    ev_ref = np.array([3.0, 0.5, 0.3, 0.2])
    assert vl.general_factor_vs_ref(np.array([0.6, 0.7, 0.29, 0.6]), ev_ref, lam_ref, ev_ref)
    assert not vl.general_factor_vs_ref(np.array([0.6, 0.7, 0.20, 0.6]), ev_ref, lam_ref, ev_ref)
    assert not vl.general_factor_vs_ref(np.array([0.6, 0.7, 0.4, 0.6]), np.array([2.0, 1.0]), lam_ref, ev_ref)


@pytest.mark.parametrize("R1, R2, r4, want", [
    (False, "pass", "pass", "fail"), (True, "fail", "pass", "fail"), (True, "pass", "fail", "fail"),
    (True, "pass", "pass", "pass"), (True, "pass", "not applicable", "pass"), (True, "pass", None, "unresolved"),
    (True, "unresolved", "pass", "unresolved"),
])
def test_level4_verdict(R1, R2, r4, want):
    assert vl.level4_verdict(R1, R2, r4) == want
