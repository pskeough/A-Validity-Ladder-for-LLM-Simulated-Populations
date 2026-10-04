"""Level 2: reproduce stored Fieller intervals and verdicts.

Stored files:
  analysis/brm/l2_verdicts.csv (headline, band_symmetric, unadjusted_90 rows), with the gaps at
    6 dp from l2_sim_gaps.csv and l2_reference.csv (l2_verdicts.csv itself rounds them to 4 dp,
    which moves interval ends by up to 0.02).
  analysis/brm/l2_headline.csv (the pooled rows of the paper's level-2 table).
  analysis/brm/l2_external_r3.csv (OpinionQA and Argyle, covariance form), and
  paper_brm/external/results/bisbee/level2_contrasts_rr1.csv (Bisbee).
  paper_brm/external/results/l2_threeway_rows.csv (kept / not kept / unresolved / not read).
"""
import numpy as np
import pandas as pd
import pytest

import validity_ladder as vl
from conftest import fixture, repo_file

BANDS = {"asym": (-0.25, 0.25, 0.75, 1.25), "sym": (-0.25, 0.25, 0.80, 1.25)}


def _inputs(r, sim, ref):
    rs = r.race_set if ("SES" in r.contrast or "Women" in r.contrast) else "n/a"
    s = sim[(sim.race_set == rs) & (sim.contrast == r.contrast) & (sim.framing == r.framing)
            & (sim.scope == r.scope)].iloc[0]
    p = ref[(ref.contrast == r.contrast) & (ref.estimand == r.estimand) & (ref.variant == r.ref_variant)].iloc[0]
    return s, p


def test_78c_rows_reproduce():
    v = fixture("l2_verdicts_headline.csv")
    sim = fixture("l2_sim_gaps.csv")
    sim["race_set"] = sim.race_set.fillna("n/a")
    ref = fixture("l2_reference.csv")
    for r in v.itertuples():
        s, p = _inputs(r, sim, ref)
        a = 0.05 / 7 if r.mult == "bonferroni7" else 0.05
        o = vl.level2_contrast(s.simulated, s.se, p.estimate, p.se_taylor, df_g=s.df, df_gamma=p.df,
                               alpha=a, bounds=BANDS[r.band])
        assert o["reading"] == r.r3_unstopped, (r.analysis, r.contrast, r.scope)
        assert o["verdict"] == r.verdict, (r.analysis, r.contrast, r.scope)
        assert o["interval_level"] == pytest.approx(r.interval_level, abs=6e-5)
        got = [o["ratio"], o["ci_lo"], o["ci_hi"], o["k_rel_halfwidth"], o["pop_ci_lo"], o["pop_ci_hi"]]
        want = [r.ratio, r.ci_lo, r.ci_hi, r.k_rel_halfwidth, r.pop_ci_lo, r.pop_ci_hi]
        np.testing.assert_allclose(got, want, atol=6e-5)
        assert o["ci_unbounded"] == bool(r.ci_unbounded) and o["ci_split"] == bool(r.ci_split)
        assert o["no_population_gap"] == bool(r.no_population_gap)


def test_headline_table_pooled_rows():
    """The paper's level-2 table: pooled ratio, its 98.6% interval and verdict, and the 90% verdict."""
    h = fixture("l2_headline.csv")
    sim = fixture("l2_sim_gaps.csv")
    sim["race_set"] = sim.race_set.fillna("n/a")
    ref = fixture("l2_reference.csv")
    for r in h.itertuples():
        rs = "four" if ("SES" in r.contrast or "Women" in r.contrast) else "n/a"
        s = sim[(sim.race_set == rs) & (sim.contrast == r.contrast) & (sim.framing == "combined")
                & (sim.scope == "pooled")].iloc[0]
        p = ref[(ref.contrast == r.contrast) & (ref.estimand == r.estimand) & (ref.variant == "primary")].iloc[0]
        o = vl.level2_contrast(s.simulated, s.se, p.estimate, p.se_taylor, df_g=s.df, df_gamma=p.df, family_size=7)
        u = vl.level2_contrast(s.simulated, s.se, p.estimate, p.se_taylor, df_g=s.df, df_gamma=p.df, family_size=1)
        assert o["ratio"] == pytest.approx(r.pooled_ratio, abs=6e-5)
        assert o["k_rel_halfwidth"] == pytest.approx(r.k, abs=6e-5)
        ci = "unbounded" if o["ci_unbounded"] else f"{o['ci_lo']:.2f} to {o['ci_hi']:.2f}" + (
            " (split)" if o["ci_split"] else "")
        assert ci == r.pooled_ci
        assert o["verdict"] == r.pooled_verdict
        assert o["reading"] == r.pooled_unstopped
        assert u["verdict"] == r.pooled_unadjusted_90


def test_external_rows_with_covariance():
    e = fixture("l2_external_r3_headline.csv")
    for r in e.itertuples():
        o = vl.level2_contrast(r.g, r.se_g, r.p, r.se_p, df_g=r.df, df_gamma=r.df, cov=r.cov,
                               family_size=r.family_size)
        assert o["verdict"] == r.verdict, r.row
        assert o["reading"] == r.r3_unstopped, r.row
        np.testing.assert_allclose([o["ratio"], o["ci_lo"], o["ci_hi"], o["k_rel_halfwidth"]],
                                   [r.ratio, r.ci_lo, r.ci_hi, r.k_rel_halfwidth], atol=1e-9, equal_nan=True)


def test_bisbee_rows():
    b = fixture("bisbee_level2_rr1.csv")
    for r in b.itertuples():
        o = vl.level2_contrast(r.g, r.se_g, r.p, r.se_p, cov=r.cov, family_size=r.m_family)
        assert o["verdict"] == r.verdict
        np.testing.assert_allclose([o["ci_lo"], o["ci_hi"]], [r.ci_lo, r.ci_hi], atol=1e-9, equal_nan=True)


def test_exact_zero_simulated_gap():
    # Twin-2K: a twin answering identically in both groups gives g = 0 with se_g = 0; the ratio is
    # exactly 0, "missing", not kept (scripts/93 reads it through the closed form).
    o = vl.level2_contrast(0.0, 0.0, -0.117053, 0.028747, cov=0.0, family_size=511)
    assert o["verdict"] == "missing" and o["label"] == "not kept"
    assert o["ci_lo"] == 0 and o["ci_hi"] == 0


def test_exact_zero_population_gap():
    # Cummins (2025) split halves: integer scores can give an exact zero reference gap; the
    # contrast is stopped as "no population gap", not a ZeroDivisionError.
    for g, se_g, cov in [(0.3, 0.2, 0.0), (0.0, 0.0, 0.0), (0.3, 0.2, 0.01)]:
        o = vl.level2_contrast(g, se_g, 0.0, 0.4, cov=cov, family_size=8)
        assert o["verdict"] == "no population gap" and o["label"] == "not read"


def test_certifiability_step0():
    # a precise reference gap is certifiable; an imprecise one and a gap covering 0 are not
    assert vl.certifiable(1.0, 0.02, family_size=8)
    assert not vl.certifiable(1.0, 0.2, family_size=8)
    assert not vl.certifiable(0.05, 0.1, family_size=8)
    # certifiable exactly when a noise-free faithful simulator is read kept
    for se in (0.02, 0.05, 0.07, 0.08, 0.1):
        o = vl.level2_contrast(1.0, 0.0, 1.0, se, family_size=8)
        assert vl.certifiable(1.0, se, family_size=8) == (o["verdict"] == "kept")
    # model readings
    assert vl.level2_model_reading(["no population gap", "reference too imprecise"], [False, False]) == \
        "reference cannot certify"
    assert vl.level2_model_reading(["undetermined", "kept"], [True, True]) == "unresolved"
    assert vl.level2_model_reading(["missing", "reference too imprecise"], [False, False]) == "fail"
    assert vl.level2_model_reading(["kept", "no population gap"], [True, False]) == "pass"


def test_faithful_simulation_planning():
    r = vl.faithful_simulation([1.0, 0.5], [0.02, 0.4], [0.01, 0.01], family_size=2, n_rep=200, seed=1)
    assert r["certifiable"] == [True, False]
    assert r["kept_share"][0] > 0.9 and r["kept_share"][1] == 0
    assert abs(sum(r["family_shares"].values()) - 1) < 1e-12


def test_three_way_labels():
    t = fixture("l2_threeway_rows.csv")
    got = [vl.three_way(r.verdict, r.ci_lo, r.ci_hi) for r in t.itertuples()]
    assert got == list(t.label)


# ------------------------------------------------------------------------- estimators (repository)
def _nhanes_l2_frame():
    fr = pd.read_csv(repo_file("analysis", "brm", "80a_nhanes_frame.csv"), low_memory=False)
    d = fr[fr.cycle.isin(list("DEFGHIJ"))].copy()
    d["w"] = d.wraw / 7
    d["sex"] = d.sex.map({"Men": "M", "Women": "F"})
    d = d.rename(columns={"pir_band": "ses", "phq8": "score"})
    return d


def test_population_gap_reproduces_reference():
    """Standardised population gaps and Taylor SEs (78a) from the stored NHANES frame (80a)."""
    d = _nhanes_l2_frame()
    ref = fixture("l2_reference.csv")
    races = ["White", "Black", "Hispanic", "Asian"]
    axes = {"race": races, "sex": ["M", "F"], "ses": ["Low", "Middle", "High"], "mar": ["Married", "Single"]}
    specs = [("Black minus White", "race", "Black", "White"), ("Women minus Men", "sex", "F", "M"),
             ("Low minus High SES", "ses", "Low", "High"), ("Asian minus White", "race", "Asian", "White")]
    for name, attr, hi, lo in specs:
        strata = [a for a in axes if a != attr]
        cells = pd.MultiIndex.from_product([axes[a] for a in strata], names=strata).to_frame(index=False)
        g = vl.population_gap(d, attr, hi, lo, strata=strata, cells=cells, weight="w", psu="psu", stratum="stratum")
        w = ref[(ref.contrast == name) & (ref.estimand == "standardised") & (ref.variant == "primary")].iloc[0]
        assert g["gamma"] == pytest.approx(w.estimate, abs=2e-6)
        assert g["se"] == pytest.approx(w.se_taylor, abs=2e-6)
        assert g["df"] == w.df
        m = vl.population_gap(d, attr, hi, lo, weight="w", psu="psu", stratum="stratum")
        wm = ref[(ref.contrast == name) & (ref.estimand == "marginal") & (ref.variant == "primary")].iloc[0]
        if attr != "race":          # 78a's marginal race groups include Asian only from 2011, as here
            assert m["gamma"] == pytest.approx(wm.estimate, abs=2e-6)


def test_simulated_gap_reproduces_pairs():
    """Persona-pair gaps and SEs (78b) per model and framing from the raw corpus."""
    c = pd.read_csv(repo_file("data", "model_outputs_v3.csv"), low_memory=False)
    c = c[c.phq8_valid.astype(bool) & c.gender.isin(["Cisgender Man", "Cisgender Woman"])]
    c = c.rename(columns={"phq8_total_clipped": "score", "ses_normalized": "ses_n"})
    sim = fixture("l2_sim_gaps.csv")
    sim["race_set"] = sim.race_set.fillna("n/a")
    specs = [("Black minus White", "race", "Black", "White", ["gender", "ses_n", "relationship"], "n/a"),
             ("Women minus Men", "gender", "Cisgender Woman", "Cisgender Man", ["race", "ses_n", "relationship"], "four"),
             ("Low minus Middle SES", "ses_n", "Low", "Middle", ["race", "gender", "relationship"], "four")]
    for (m, f), g in c.groupby(["model", "prompt_condition"]):
        for name, attr, hi, lo, strata, rs in specs:
            gg = g if rs == "n/a" else g[g.race != "Multiracial"]
            r = vl.simulated_gap(gg, attr, hi, lo, score="score", persona="profile_id", strata=strata)
            w = sim[(sim.contrast == name) & (sim.scope == m) & (sim.framing == f) & (sim.race_set == rs)].iloc[0]
            assert r["n_pairs"] == w.n_pairs
            assert r["g"] == pytest.approx(w.simulated, abs=2e-6)
            assert r["se"] == pytest.approx(w.se, abs=2e-6)
            assert r["df"] == w.df
