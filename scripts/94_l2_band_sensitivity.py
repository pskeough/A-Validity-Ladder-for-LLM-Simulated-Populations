"""Level-2 kept-region sensitivity (reviewer request; SENSITIVITY ANALYSIS, not a change to the
frozen rule in paper_brm/LADDER_SPEC.md).

Question. The frozen level-2 kept region is [0.75, 1.25]. How do the per-contrast counts
(kept / not kept / unresolved / not read) move when the kept region is
  [0.67, 1.50]  log-symmetric, wider
  [0.80, 1.25]  log-symmetric, narrower (the "symmetric band" already carried by 78c / 78e)
  [0.90, 1.10]  tight
with every interval, stop and multiplicity setting held fixed?

What changes with the band and what does not.
  * The Fieller interval for rho = g / gamma does not depend on the band (same g, SEs, df,
    covariance and Bonferroni level). It is recomputed here by the frozen engines
    (78c.r3 for the worked example, 78e.r3cov for the external data) and checked against the
    stored interval.
  * Region boundaries passed to the engines are (-0.25, 0.25, lo, hi): only the kept region's
    two boundaries move; the reversed/missing boundaries stay at -0.25 and 0.25.
  * Stop 1 ("no population gap") is band-free.
  * Stop 2 ("reference too imprecise") is applied with each band's own threshold, exactly as the
    frozen code derives it: with W = hi / lo, no ratio interval can fit inside the kept region
    when k = t SE_gamma / |gamma| > k_max = (W - 1) / (W + 1). The engines compute k_max from the
    bounds they receive, so passing the alternative bounds applies the stop consistently:
      [0.75, 1.25] k_max = 0.2500   [0.67, 1.50] k_max = 0.3825
      [0.80, 1.25] k_max = 0.2195   [0.90, 1.10] k_max = 0.1000
    As in LADDER_SPEC, the stop fires when k > k_max and the reading does not exclude kept; a
    reading that excludes kept stands. Since the 3 Oct 2026 bug fix the engines label a reading
    that crosses more than one boundary below kept "reversed to attenuated", so it stands; the
    count n_undet_outside_kept_stopped checks that no stopped reading excludes kept (0 per band).
  * Three-way label, as in the article (paper_brm/external/scripts/l2_threeway.py and
    scripts/84_fig_level2_brm.three_way): stopped -> not read; interval wholly inside the kept
    region -> kept; wholly outside -> not kept; otherwise unresolved. A missing (NaN) external
    interval is "unresolved" as in l2_threeway.py; an unbounded worked-example interval falls back
    to the engine's region reading.

Subsets (as Figure 3, scripts/84f_fig3_l2_dots.py):
  Worked example  analysis/brm/l2_verdicts.csv, analysis == headline, estimand standardised, the
                  28 per-model rows (pooled scope excluded); 98.57% intervals (Bonferroni over 7).
  Bisbee          paper_brm/external/results/bisbee/level2_contrasts_rr1.csv, framing == full
                  (77 contrasts, Bonferroni over 77, df infinite).
  Argyle          analysis/brm/l2_external_r3.csv, config headline, family t0.7_main (61).
  OpinionQA       analysis/brm/l2_external_r3.csv, config headline, source OpinionQA (154).

Gate: the [0.75, 1.25] column must reproduce the published counts (worked example 0/15/5/8,
Bisbee 3/16/7/51, Argyle 2/12/1/46, OpinionQA 1/33/115/5; after the 3 Oct 2026 stop-2 fix) and every row's stored verdict and
three-way label, or the script stops.

Output: analysis/brm/94_l2_band_sensitivity.csv (dataset x band x counts) and
analysis/brm/94_l2_band_sensitivity_rows.csv (per contrast x band). Closed form; no resampling.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUTD = os.path.join(BASE, "analysis", "brm")
EXT = os.path.join(BASE, "paper_brm", "external", "results")
FIG = os.path.join(BASE, "paper_brm", "manuscript", "figures")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = [old[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


R78C = _load("r78c", "78c_l2_r3_verdicts.py")
R78E = _load("r78e", "78e_l2_external_r3.py")

BANDS = [("[0.75, 1.25] frozen", 0.75, 1.25), ("[0.67, 1.50]", 0.67, 1.50),
         ("[0.80, 1.25]", 0.80, 1.25), ("[0.90, 1.10]", 0.90, 1.10)]
STOPS = {"reference too imprecise", "no population gap"}
ORDER = ["kept", "not kept", "unresolved", "not read"]
PUBLISHED = {"Worked example": (0, 15, 5, 8), "Bisbee": (3, 16, 7, 51), "Argyle": (2, 12, 1, 46),
             "OpinionQA": (1, 33, 115, 5)}


def excludes_kept(lab):
    """True when a named region reading excludes kept (undetermined is not decided here)."""
    return lab != "undetermined" and "kept" not in lab.split(" or ")


def three_way(verdict, lab, lo, hi, kl, kh, unbounded=False):
    """Article's per-contrast label against the kept region [kl, kh]. Returns (label, agrees)
    where agrees says whether the interval reading matches the engine's region reading."""
    if verdict in STOPS:
        return "not read", True
    if unbounded:
        if lab == "kept":
            return "kept", True
        return ("not kept" if excludes_kept(lab) else "unresolved"), True
    if np.isnan(lo) or np.isnan(hi):
        return "unresolved", lab == "undetermined" or "kept" in lab.split(" or ")
    if lo >= kl and hi <= kh:
        out = "kept"
        ok = lab == "kept"
    elif hi < kl or lo > kh:
        out = "not kept"
        ok = lab == "undetermined" or excludes_kept(lab)
    else:
        out = "unresolved"
        ok = lab == "undetermined" or "kept" in lab.split(" or ")
    return out, ok


def collapse_fig3(label, lo=None, hi=None):
    """84f_fig3_l2_dots.collapse, restated so the published labels can be compared per row."""
    if label.startswith("kept"):
        return "kept"
    if label.startswith("not kept") or label in ("missing or attenuated", "reversed or missing",
                                                  "reversed to attenuated"):
        return "not kept"
    if label.startswith("not read") or label in ("no population gap", "reference too imprecise"):
        return "not read"
    if label == "undetermined" and lo is not None and not (hi >= 0.75 and lo <= 1.25):
        return "not kept"
    return "unresolved"


# ------------------------------------------------------------------------------------- inputs
def worked_rows():
    """Headline per-model rows. Inputs are taken unrounded from 78a / 78b exactly as 78c.main
    selects them (l2_verdicts.csv stores them rounded to 4 dp); the stored verdict and interval
    from l2_verdicts.csv are carried for the gate."""
    v = pd.read_csv(os.path.join(OUTD, "l2_verdicts.csv"))
    h = v[(v.analysis == "headline") & (v.estimand == "standardised") & (v.scope != "pooled")].copy()
    assert len(h) == 28, len(h)
    ref = pd.read_csv(os.path.join(OUTD, "l2_reference.csv"))
    sim = pd.read_csv(os.path.join(OUTD, "l2_sim_gaps.csv"))
    sim["race_set"] = sim.race_set.fillna("n/a")
    raw = []
    for r in h.itertuples():
        rr = ref[(ref.contrast == r.contrast) & (ref.estimand == "standardised") &
                 (ref.variant == "primary")]
        assert len(rr) == 1
        rr = rr.iloc[0]
        rs = "n/a" if r.contrast in R78C.CONTRASTS[:3] else "four"
        ss = sim[(sim.corpus == "all") & (sim.panel == "four models") & (sim.race_set == rs) &
                 (sim.contrast == r.contrast) & (sim.framing == "combined") &
                 (sim.scope == r.scope)]
        assert len(ss) == 1
        ss = ss.iloc[0]
        assert abs(ss.simulated - r.simulated) < 1e-4 and abs(rr.estimate - r.population) < 1e-4
        raw.append(dict(simulated=ss.simulated, se_sim=ss.se, df_sim=ss.df,
                        population=rr.estimate, se_pop=rr.se_taylor, df_pop=rr.df))
    raw = pd.DataFrame(raw, index=h.index)
    h[raw.columns] = raw
    f = pd.read_csv(os.path.join(FIG, "fig2_level2_data.csv"))
    f = f[~f.pooled_descriptive.astype(bool)].set_index(["contrast", "scope"])
    h["published_label"] = [collapse_fig3(f.loc[(c, s)].verdict_printed)
                            for c, s in zip(h.contrast, h.scope)]
    h["dataset"] = "Worked example"
    h["row"] = h.contrast + " | " + h.scope
    return h.reset_index(drop=True)


def external_rows():
    tw = pd.read_csv(os.path.join(EXT, "l2_threeway_rows.csv"))
    b = pd.read_csv(os.path.join(EXT, "bisbee", "level2_contrasts_rr1.csv"))
    b = b[b.framing == "full"].copy()
    assert len(b) == 77
    tb = tw[(tw.dataset == "Bisbee") & (tw.framing == "full")]
    assert len(tb) == 77
    # rows of l2_threeway_rows are in the same order as the Bisbee file within a framing
    assert (tb.contrast.to_numpy() == b.contrast.to_numpy()).all()
    assert np.allclose(tb.ratio.to_numpy(float), b.ratio.to_numpy(float))
    b["published_label"] = [collapse_fig3(l, lo, hi) for l, lo, hi in
                            zip(tb.label, tb.ci_lo, tb.ci_hi)]
    b = b.assign(dataset="Bisbee", row=b.outcome + " | " + b.contrast, df=np.inf,
                 alpha=0.05 / b.m_family, published_verdict_stored=b.verdict)
    e = pd.read_csv(os.path.join(OUTD, "l2_external_r3.csv"))
    e = e[e.config == "headline"]
    a = e[(e.source == "Argyle") & (e.family == "t0.7_main")].copy()
    o = e[e.source == "OpinionQA"].copy()
    ta = tw[(tw.dataset == "Argyle") & (tw.run == "t0.7_main")]
    to = tw[tw.dataset == "OpinionQA"]
    out = [b]
    for name, d, t in (("Argyle", a, ta), ("OpinionQA", o, to)):
        assert len(d) == len(t)
        assert np.allclose(d.ratio.to_numpy(float), t.ratio.to_numpy(float))
        d = d.assign(dataset=name, alpha=0.05 / d.family_size, published_verdict_stored=d.verdict)
        d["published_label"] = [collapse_fig3(l, lo, hi) for l, lo, hi in
                                zip(t.label, t.ci_lo, t.ci_hi)]
        out.append(d)
    cols = ["dataset", "row", "contrast", "g", "se_g", "p", "se_p", "cov", "df", "alpha", "ci_lo",
            "ci_hi", "published_verdict_stored", "published_label"]
    return pd.concat([x[cols] for x in out], ignore_index=True)


# ---------------------------------------------------------------------------------------- run
def main():
    rows = []
    w = worked_rows()
    for r in w.itertuples():
        for bname, kl, kh in BANDS:
            bounds = np.array([-0.25, 0.25, kl, kh])
            # headline multiplicity: Bonferroni over the family of 7 (the stored alpha column is
            # rounded to 4 dp, so the exact value is used, as 78c.main does)
            assert abs(r.alpha_one_sided - R78C.ALPHA / R78C.FAMILY) < 1e-4
            o = R78C.r3(r.simulated, r.se_sim, r.df_sim, r.population, r.se_pop, r.df_pop,
                        R78C.ALPHA / R78C.FAMILY, bounds)
            lab3, ok = three_way(o["verdict"], o["r3_unstopped"], o["ci_lo"], o["ci_hi"], kl, kh,
                                 bool(o["ci_unbounded"]))
            rows.append(dict(dataset="Worked example", row=r.row, band=bname, kept_lo=kl,
                             kept_hi=kh, k_rel_halfwidth=o["k_rel_halfwidth"], k_max=o["k_max"],
                             ci_lo=o["ci_lo"], ci_hi=o["ci_hi"], stored_ci_lo=r.ci_lo,
                             stored_ci_hi=r.ci_hi, ci_unbounded=bool(o["ci_unbounded"]),
                             r3_unstopped=o["r3_unstopped"], verdict=o["verdict"], label=lab3,
                             interval_agrees_with_tests=ok, stored_verdict=r.verdict,
                             published_label=r.published_label))
    x = external_rows()
    for r in x.itertuples():
        for bname, kl, kh in BANDS:
            bounds = np.array([-0.25, 0.25, kl, kh])
            o = R78E.r3cov(r.g, r.se_g, r.p, r.se_p, r.cov, r.df, r.alpha, bounds)
            W = kh / kl
            lab3, ok = three_way(o["verdict"], o["r3_unstopped"], o["ci_lo"], o["ci_hi"], kl, kh)
            rows.append(dict(dataset=r.dataset, row=r.row, band=bname, kept_lo=kl, kept_hi=kh,
                             k_rel_halfwidth=o["k_rel_halfwidth"], k_max=(W - 1) / (W + 1),
                             ci_lo=o["ci_lo"], ci_hi=o["ci_hi"], stored_ci_lo=r.ci_lo,
                             stored_ci_hi=r.ci_hi, ci_unbounded=False,
                             r3_unstopped=o["r3_unstopped"], verdict=o["verdict"], label=lab3,
                             interval_agrees_with_tests=ok,
                             stored_verdict=r.published_verdict_stored,
                             published_label=r.published_label))
    res = pd.DataFrame(rows)
    res["undet_outside_kept_stopped"] = (
        (res.verdict == "reference too imprecise") & (res.r3_unstopped == "undetermined") &
        ((res.ci_hi < res.kept_lo) | (res.ci_lo > res.kept_hi)))

    # ---------------------------------------------------------------------------- gate
    fz = res[res.band == BANDS[0][0]]
    bad_v = fz[fz.verdict != fz.stored_verdict]
    bad_l = fz[fz.label != fz.published_label]
    ci_ok = np.isclose(res.ci_lo.astype(float), res.stored_ci_lo.astype(float), atol=2e-4,
                       equal_nan=True) & \
        np.isclose(res.ci_hi.astype(float), res.stored_ci_hi.astype(float), atol=2e-4,
                   equal_nan=True)
    print(f"gate: frozen band, rows {len(fz)}; verdict mismatches {len(bad_v)}; "
          f"three-way label mismatches {len(bad_l)}; interval mismatches (all bands) "
          f"{int((~ci_ok).sum())}")
    if (~ci_ok).any():
        print(res[~ci_ok][["dataset", "row", "band", "ci_lo", "stored_ci_lo", "ci_hi", "stored_ci_hi", "ci_unbounded"]].head(30).to_string())
    if len(bad_v) or len(bad_l) or (~ci_ok).any():
        print(bad_v.to_string())
        print(bad_l.to_string())
        raise SystemExit("frozen band does not reproduce the stored verdicts / labels / intervals")

    summ = []
    for (ds, bname), g in res.groupby(["dataset", "band"], sort=False):
        c = g.label.value_counts()
        summ.append(dict(dataset=ds, band=bname, kept_lo=g.kept_lo.iloc[0],
                         kept_hi=g.kept_hi.iloc[0], k_max=round(float(g.k_max.iloc[0]), 4),
                         total=len(g), **{k: int(c.get(k, 0)) for k in ORDER},
                         not_read_no_population_gap=int((g.verdict == "no population gap").sum()),
                         not_read_reference_imprecise=int(
                             (g.verdict == "reference too imprecise").sum()),
                         n_undet_outside_kept_stopped=int(g.undet_outside_kept_stopped.sum()),
                         # LADDER_SPEC wording ("a reading that excludes kept stands") read
                         # literally: those undetermined-but-outside rows count as not kept
                         spec_literal_not_kept=int(c.get("not kept", 0) +
                                                   g.undet_outside_kept_stopped.sum()),
                         spec_literal_not_read=int(c.get("not read", 0) -
                                                   g.undet_outside_kept_stopped.sum()),
                         interval_test_disagreements=int((~g.interval_agrees_with_tests).sum())))
    summ = pd.DataFrame(summ)
    for ds, want in PUBLISHED.items():
        got = summ[(summ.dataset == ds) & (summ.band == BANDS[0][0])][ORDER].iloc[0]
        if tuple(int(v) for v in got) != want:
            raise SystemExit(f"{ds}: frozen-band counts {tuple(got)} != published {want}")
    print("gate: frozen-band counts reproduce the published counts for all four datasets")
    if summ.interval_test_disagreements.sum():
        print("WARNING: interval and boundary-test readings disagree on some rows")

    dsorder = list(PUBLISHED)
    summ["_d"] = summ.dataset.map(dsorder.index)
    summ["_b"] = summ.band.map([b[0] for b in BANDS].index)
    summ = summ.sort_values(["_d", "_b"]).drop(columns=["_d", "_b"])
    summ.to_csv(os.path.join(OUTD, "94_l2_band_sensitivity.csv"), index=False)
    res.to_csv(os.path.join(OUTD, "94_l2_band_sensitivity_rows.csv"), index=False)

    pd.set_option("display.width", 250)
    print()
    print(summ.to_string(index=False))
    # which rows change label relative to the frozen band
    piv = res.pivot_table(index=["dataset", "row"], columns="band", values="label",
                          aggfunc="first")
    piv = piv[[b[0] for b in BANDS]]
    moved = piv[(piv.ne(piv.iloc[:, 0], axis=0)).any(axis=1)]
    print(f"\nrows whose label differs from the frozen band in at least one band: {len(moved)}")
    print("worked example rows that move:")
    print(moved.loc["Worked example"].to_string() if "Worked example" in
          moved.index.get_level_values(0) else "  none")
    print("\nwritten analysis/brm/94_l2_band_sensitivity.csv and 94_l2_band_sensitivity_rows.csv")


if __name__ == "__main__":
    main()
