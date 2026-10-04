"""Gate: reproduce the stored G-study numbers of the PHQ-8 corpus.

Stored files: analysis/brm/gate_dstudy.csv (components, phi, k), analysis/brm/91a_core_regression.csv
(SE(30) and the SD-unit tolerances), analysis/brm/80a_tolerance_basis.csv (reference SD 3.9352).
"""
import numpy as np
import pandas as pd
import pytest

import validity_ladder as vl
from conftest import SHORT, fixture, repo_file

SD_REF = 3.9352     # NHANES 2005-2018 adults, 80a_tolerance_basis.csv


def test_reference_sd_and_tolerances():
    tb = fixture("l3_tolerance_basis.csv")
    sd = float(tb[(tb.window == "2005-2018") & (tb.population == "all adults 18+")].sd.iloc[0])
    assert sd == pytest.approx(SD_REF)
    rg = fixture("91a_core_regression.csv")
    row = rg[rg.check == "gate verdict, SD tolerances vs points"].iloc[0]
    assert vl.thresholds.GATE_SE_MIN_SD * sd == pytest.approx(row.tol_min, abs=5e-5)    # 0.9838
    assert vl.thresholds.GATE_SE_REC_SD * sd == pytest.approx(row.tol_rec, abs=5e-5)    # 0.4919


def test_dstudy_from_stored_components():
    """phi(1), phi(30), k for phi .80 / .90 and k for the SE tolerances, from the stored s2_p, s2_e."""
    d = fixture("gate_dstudy.csv")
    for r in d.itertuples():
        assert vl.phi_k(r.s2_p, r.s2_e, 30) == pytest.approx(r.phi_nested_30, abs=6e-5)
        assert vl.phi_k(r.s2_p, r.s2_e, 1) == pytest.approx(r.phi_nested_1, abs=6e-5)
        assert vl.k_for_phi(r.s2_p, r.s2_e, 0.80) == r.k_phi80_nested
        assert vl.k_for_phi(r.s2_p, r.s2_e, 0.90) == r.k_phi90_nested
        assert vl.k_for_se(r.s2_e, r.tol1) == r.k_se_tol1
        assert vl.k_for_se(r.s2_e, r.tol2) == r.k_se_tol2
        assert np.sqrt(r.s2_e / 30) == pytest.approx(r.se_at_30, abs=6e-5)


def test_gate_on_corpus_reproduces_dstudy():
    """gate() on the raw PHQ-8 corpus returns the stored components, SE(30) and verdicts."""
    c = pd.read_csv(repo_file("data", "model_outputs_v3.csv"), low_memory=False)
    c = c[c.phq8_valid.astype(str) == "True"]
    assert len(c) == 28799
    d = fixture("gate_dstudy.csv")
    rg = fixture("91a_core_regression.csv")
    rg = rg[rg.check == "gate verdict, SD tolerances vs points"].set_index("unit")
    for (m, f), g in c.groupby(["model", "prompt_condition"]):
        for outcome, y in (("PHQ-8 total", g.phq8_total.astype(float)),
                           ("PHQ-8 >= 10", (g.phq8_total >= 10).astype(float))):
            r = vl.gate(y, g.profile_id, SD_REF, k=30, n_boot=0)
            w = d[(d.outcome == outcome) & (d.model == SHORT[m]) & (d.framing == f)].iloc[0]
            assert r["n_personas"] == w.n_persona == 120
            assert r["s2_p"] == pytest.approx(w.s2_p, rel=6e-5)
            assert r["s2_r"] == pytest.approx(w.s2_e, rel=6e-5)
            assert r["phi_k"] == pytest.approx(w.phi_nested_30, abs=6e-6)
            assert r["k_sep_min"] == w.k_phi80_nested
            if outcome == "PHQ-8 total":
                x = rg.loc[f"{SHORT[m]} {f}"]
                assert r["se_k"] == pytest.approx(x.se30, abs=1e-12)
                # every model passes the recommended rule at k = 30 (LADDER_SPEC change 2)
                assert r["pass_min"] and r["pass_rec"] and r["verdict"] == "pass"
