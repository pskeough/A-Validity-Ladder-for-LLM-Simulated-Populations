"""Levels 1 and 4: reproduce the stored GRM, lz*, tail ratios, verdicts, tau and loadings.

Stored files: analysis/brm/l1_scores_nhanes.csv and l1_scores_corpus.csv (items, weights, design,
lz*), l1_grm_params.csv, l1_personfit_main.csv, l1_personfit_tolerance.csv, l4_structure.csv.
The NHANES files are outputs the analysis scripts wrote, not the NHANES downloads.
"""
import numpy as np
import pandas as pd
import pytest

import validity_ladder as vl
from conftest import DPQ, PHQ_ITEMS, fixture, repo_file
from validity_ladder._grm import GRM


def test_tau_from_stored_subgroup_intervals():
    """tau = 1.5 from the NHANES subgroups, worst resolved departure 1.27 (non-Hispanic Black)."""
    t = fixture("l1_personfit_tolerance.csv")
    tau, worst = vl.tau_from_intervals(t.misfit_ci90_lo, t.misfit_ci90_hi, t.overfit_ci90_lo, t.overfit_ci90_hi)
    rg = fixture("91a_core_regression.csv")
    want = rg[rg.check == "tau, resolved vs point departure"].iloc[0]
    assert tau == 1.5
    assert worst == pytest.approx(want.worst_resolved, abs=1e-12)
    assert round(worst, 2) == 1.27


@pytest.fixture(scope="module")
def nhanes():
    nh = pd.read_csv(repo_file("analysis", "brm", "l1_scores_nhanes.csv"))
    return nh[nh.reference == "2005_2018"].reset_index(drop=True)


@pytest.fixture(scope="module")
def corpus_scores():
    return pd.read_csv(repo_file("analysis", "brm", "l1_scores_corpus.csv"))


def _params():
    par = fixture("l1_grm_params.csv")
    par = par[par.fit == "weighted"].set_index("item").loc[DPQ]
    return par.a.to_numpy(float), par[["b1", "b2", "b3"]].to_numpy(float)


def test_grm_fit_reproduces_published(nhanes):
    fit = GRM(8, 4).fit(nhanes[DPQ].to_numpy(int), nhanes.w.to_numpy())
    a, b = _params()
    np.testing.assert_allclose(fit["a"], a, atol=1e-6)
    np.testing.assert_allclose(fit["b"], b, atol=1e-6)


def test_level1_ratios_and_verdicts(nhanes, corpus_scores):
    ref = vl.L1Reference(nhanes[DPQ].to_numpy(int), weights=nhanes.w.to_numpy(), psu=nhanes.psu.to_numpy(),
                         stratum=nhanes.stratum.to_numpy(), n_boot=100, grm_params=_params())
    np.testing.assert_allclose(ref.lzstar, nhanes.lzstar_weighted.to_numpy(), atol=1e-9)
    pub = fixture("l1_personfit_main.csv")
    for (mdl, f), g in corpus_scores.groupby(["short", "framing"]):
        X = g[PHQ_ITEMS].to_numpy(int)
        _, lz = ref.lzstar_of(X)
        np.testing.assert_allclose(lz, g.lzstar_weighted.to_numpy(), atol=1e-9)
        r = vl.level1(X, g.profile_id.to_numpy(), ref, tau=1.5)
        q = pub[(pub.model == mdl) & (pub.framing == f)].iloc[0]
        assert r["misfit_ratio"] == pytest.approx(q.misfit_ratio, abs=1e-12)
        assert r["overfit_ratio"] == pytest.approx(q.overfit_ratio, abs=1e-12)
        assert r["mean_mid_pit"] == pytest.approx(q.mean_mid_pit, abs=1e-12)
        # intervals are bootstrap (B = 100 here, 2,000 published); the verdicts are far from the band
        assert r["verdict_detail"] == q.verdict, (mdl, f)
        assert r["verdict"] == "fail"


def test_level4_loadings_and_r1(nhanes, corpus_scores):
    st = fixture("l4_structure.csv")
    lcols = [f"load_{i}" for i in range(1, 9)]
    ref = vl.L4Reference(nhanes[DPQ].to_numpy(int), weights=nhanes.w.to_numpy(), cluster=nhanes.psu.to_numpy(),
                         n_boot=2)
    q = st[st.source == "NHANES"].iloc[0]
    np.testing.assert_allclose(ref.lam, q[lcols].to_numpy(float), atol=1e-8)
    assert ref.ev[0] / ref.ev[1] == pytest.approx(q.ev_ratio_12, abs=1e-8)      # 7.47 (LADDER_SPEC)
    assert round(float(ref.lam.min()), 2) == 0.67
    for (mdl, f), g in corpus_scores.groupby(["short", "framing"]):
        R = vl.polychoric_matrix(g[PHQ_ITEMS].to_numpy(int), n_categories=4)
        lam = vl.one_factor_uls(R)
        ev = np.sort(np.linalg.eigvalsh(R))[::-1]
        q = st[(st.model == mdl) & (st.framing == f)].iloc[0]
        np.testing.assert_allclose(lam, q[lcols].to_numpy(float), atol=1e-8)
        assert ev[0] / ev[1] == pytest.approx(q.ev_ratio_12, abs=1e-8)
        # frozen R1 (relative to the reference) equals the absolute R1 on NHANES (LADDER_SPEC level 4)
        assert vl.general_factor_vs_ref(lam, ev, ref.lam, ref.ev) == bool(q.general_factor)
