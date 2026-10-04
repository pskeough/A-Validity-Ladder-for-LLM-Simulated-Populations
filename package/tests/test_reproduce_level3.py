"""Level 3: reproduce the post-stratified residuals and TOST verdicts of analysis/brm/80c_l3_results.csv
(spec S1 primary: NHANES 2005-2018, PIR income bands, marital status coded as in 78a).

The fixture test needs only the stored rows; the full test rebuilds every residual from the stored
NHANES frame (80a_nhanes_frame.csv) and simulated cells (80b_sim_cells.csv).
"""
import numpy as np
import pandas as pd
import pytest

import validity_ladder as vl
from conftest import fixture, repo_file

SD0 = 3.9352
DESIGN_DF = 109        # 80c_stdout.txt: S1 primary, design df 109 (PSUs minus strata)


def test_tost_from_stored_residuals():
    """CI, SE and every pass flag of the stored rows from their residual and SE parts."""
    t = fixture("l3_s1_primary_ps.csv")
    for r in t.itertuples():
        mean = r.outcome == "mean"
        o = vl.level3_tost(r.resid, r.se_sim ** 2, r.se_ref_part ** 2, DESIGN_DF, 2.0 if mean else 5.0)
        assert o["se"] == pytest.approx(r.se, abs=2e-6)
        assert o["df"] == pytest.approx(r.df, rel=2e-3)        # df is sensitive to the 6-dp rounding
        assert o["ci90_lo"] == pytest.approx(r.ci90_lo, abs=3e-6)
        assert o["ci90_hi"] == pytest.approx(r.ci90_hi, abs=3e-6)
        tols = {"pass_0.2SD": 0.2 * SD0, "pass_1pt": 1.0, "pass_2pt": 2.0} if mean else \
            {"pass_2.5pp": 2.5, "pass_5pp": 5.0}
        for col, tol in tols.items():
            v = vl.level3_tost(r.resid, r.se_sim ** 2, r.se_ref_part ** 2, DESIGN_DF, tol)["verdict"]
            assert (v == "pass") == bool(t.loc[r.Index, col]), (r.model, r.group, col)


def _frame():
    fr = pd.read_csv(repo_file("analysis", "brm", "80a_nhanes_frame.csv"), low_memory=False)
    d = fr[fr.cycle.isin(list("DEFGHIJ"))].copy()
    d["w"] = d.wraw / 7
    d.loc[d.race == "Asian", "w"] = d.loc[d.race == "Asian", "wraw"] / 4     # 80_l3_lib.window_frame
    return d.rename(columns={"pir_band": "inc"})


RACES, SEXES, INCS, MARS = ["White", "Black", "Asian", "Hispanic"], ["Men", "Women"], ["Low", "Middle", "High"], \
    ["Married", "Single"]
CELLS = [(r, s, i, m) for r in RACES for s in SEXES for i in INCS for m in MARS]


def test_full_reproduction_from_stored_frames():
    """Every PS row (5 scopes x 10 groups x 2 outcomes, framings combined) from the stored frames."""
    d = _frame()
    sc = pd.read_csv(repo_file("analysis", "brm", "80b_sim_cells.csv"))
    sc = sc[sc.corpus == "all"]
    pub = fixture("l3_s1_primary_ps.csv")
    cidx = {c: k for k, c in enumerate(CELLS)}
    ref_cell = np.array([cidx.get(k, -1) for k in zip(d.race, d.sex, d.inc, d.mar)])
    groups = {"Overall": np.ones(48, bool)}
    for j, lv in enumerate([RACES, SEXES, INCS]):
        for x in lv:
            groups[x] = np.array([c[j] == x for c in CELLS])
    n_rows = 0
    for outcome in ("mean", "prev10"):
        y = d.phq8.to_numpy(float) if outcome == "mean" else d.dep10.to_numpy(float) * 100
        k = sc.assign(s=sc["mean"] if outcome == "mean" else sc.prev10 * 100,
                      v=sc["var"] / sc.n if outcome == "mean" else sc.prev10_var * 1e4)
        per = {}
        for mdl, g in k.groupby("model"):
            c = g[g.framing == "clinical"].set_index("profile_id")
            n = g[g.framing == "narrative"].set_index("profile_id")
            per[mdl] = pd.DataFrame(dict(s=(c.s + n.s) / 2, v=(c.v + n.v) / 4)).join(c[["race", "sex", "inc", "mar"]])
        allm = pd.concat(per.values())
        pooled = allm.groupby(level=0).agg(s=("s", "mean"), v=("v", "sum"), race=("race", "first"),
                                           sex=("sex", "first"), inc=("inc", "first"), mar=("mar", "first"))
        pooled["v"] = pooled.v / 16
        per["pooled"] = pooled
        for mdl, tab in per.items():
            s, v = np.full(48, np.nan), np.full(48, np.nan)
            for r in tab.itertuples():
                key = (r.race, r.sex, r.inc, r.mar)
                if key in cidx:
                    s[cidx[key]], v[cidx[key]] = r.s, r.v
            res = vl.level3_from_cells(s, v, ref_cell, y, d.w.to_numpy(), groups,
                                       tolerance=2.0 if outcome == "mean" else 2.5,
                                       psu=d.psu.to_numpy(), stratum=d.stratum.to_numpy(),
                                       ratio_band=(0.80, 1.25) if outcome == "prev10" else None)
            p = pub[(pub.outcome == outcome) & (pub.model == mdl) & (pub.framing == "combined")].set_index("group")
            for r in res.itertuples():
                q = p.loc[r.group]
                cols = ["sim", "ref", "resid", "se_sim", "se_ref_part", "se", "ci90_lo", "ci90_hi"]
                np.testing.assert_allclose([getattr(r, c) for c in cols], q[cols].to_numpy(float), atol=1e-6)
                assert r.n_ref == q.n_nhanes
                if outcome == "mean":
                    assert (r.verdict == "pass") == bool(q.pass_2pt)
                else:
                    assert r.ratio_ci90_lo == pytest.approx(q.ratio_ci90_lo, abs=1e-6)
                    assert r.ratio_ci90_hi == pytest.approx(q.ratio_ci90_hi, abs=1e-6)
                    assert r.ratio_pass == bool(q.ratio_tost_pass)
                n_rows += 1
    assert n_rows == 100


def test_level3_long_data_api_matches_80c():
    """level3() on long draws (one model, one framing) gives the stored PS rows."""
    d = _frame()
    c = pd.read_csv(repo_file("data", "model_outputs_v3.csv"), low_memory=False)
    c = c[c.phq8_valid.astype(bool) & c.gender.isin(["Cisgender Man", "Cisgender Woman"]) & (c.race != "Multiracial")]
    c = c.assign(sex=c.gender.map({"Cisgender Man": "Men", "Cisgender Woman": "Women"}), inc=c.ses_normalized,
                 mar=c.relationship, score=c.phq8_total.astype(float))
    pub = fixture("l3_s1_primary_ps.csv")
    for mdl in ("openai/gpt-4o-mini", "z-ai/glm-4.7"):
        g = c[(c.model == mdl) & (c.prompt_condition == "clinical")]
        r = vl.level3(g, d, ["race", "sex", "inc", "mar"], score="score", persona="profile_id", ref_score="phq8",
                      weight="w", psu="psu", stratum="stratum", group_attributes=["race", "sex", "inc"],
                      tolerance=2.0, sd_ref=SD0)
        t = r["groups"].assign(group=lambda x: x.group.str.split("=").str[-1]).set_index("group")
        p = pub[(pub.outcome == "mean") & (pub.model == mdl.split("/")[1]) & (pub.framing == "clinical")].set_index("group")
        for grp, q in p.iterrows():
            cols = ["resid", "se_sim", "se_ref_part", "ci90_lo", "ci90_hi"]
            np.testing.assert_allclose(t.loc[grp, cols].to_numpy(float), q[cols].to_numpy(float), atol=1e-6)
            assert t.loc[grp, "pass_0.2sd"] == bool(q["pass_0.2SD"])
