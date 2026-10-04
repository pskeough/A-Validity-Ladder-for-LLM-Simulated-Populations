"""EXPLORATORY, POST-FREEZE. Not part of the frozen ladder (paper_brm/LADDER_SPEC.md v1.0).

Question. Level 2's stop 1 ("no population gap": the population gap's own interval covers 0)
leaves a contrast unread, so a model that invents a gap where people show none is never tested.
The fix the paper names: for those contrasts, test the simulated gap itself for equivalence to
zero (TOST, 90% interval) in raw score units against a tolerance delta.

Rule used here (exploratory):
  interval   g -/+ t_{.95}(df_g) * SE_g   (90%; df_g = the simulated gap's df, infinite for the
             external bootstrap SEs). SE_g and df_g are exactly the ones the level-2 pipeline
             already uses for the ratio test (78b for the worked example; the respondent-bootstrap
             SEs in the external result files).
  delta      0.10, 0.20 and 0.25 reference SD, converted to raw units with the reference SD of
             that outcome.
  verdict    pass   (equivalent to zero): interval wholly inside (-delta, +delta)
             fail   (invented gap): interval wholly outside [-delta, +delta]
             unresolved: otherwise.
  A secondary column repeats the reading at the family's Bonferroni level (the same family as the
  frozen level-2 rule: 7 contrasts for the worked example, 77 for Bisbee full, 61 for Argyle
  t0.7_main), i.e. interval level 1 - 2 * .05 / m. The 90% column is the requested reading.

Rows (the contrasts the frozen rule stops as "no population gap", in the Figure 3 subsets):
  Worked example  analysis/brm/l2_verdicts.csv headline, standardised estimand, per-model rows
                  (Hispanic minus White x 4 models). Raw units: PHQ-8 points; SE and df unrounded
                  from analysis/brm/l2_sim_gaps.csv (78b). Reference SD 3.9352 (NHANES 2005-2018,
                  analysis/brm/80a_tolerance_basis.csv, as 83_controls_lib.SD0).
  Bisbee          paper_brm/external/results/bisbee/level2_contrasts_rr1.csv, framing full;
                  thermometer points 0-100; reference SD = SD of the same respondents' thermometer
                  (gate_rr1.csv, sd_ref, framing full), as the Bisbee gate uses.
  Argyle          analysis/brm/l2_external_r3.csv headline, family t0.7_main; outcome units
                  (share or scale points); reference SD = SD of the same respondents' human answers
                  among respondents with both answers, recovered exactly from
                  results/argyle/level3_groups.csv (group all: strict_tol = 1.959964 * SD / sqrt(n)).
  OpinionQA       analysis/brm/l2_external_r3.csv headline: no row is stopped as "no population
                  gap" (its 5 stops are "reference too imprecise"), so it contributes no rows.

Output: analysis/brm/95_null_gap_equivalence.csv. Closed form; no resampling.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUTD = os.path.join(BASE, "analysis", "brm")
EXT = os.path.join(BASE, "paper_brm", "external", "results")
DELTAS = (0.10, 0.20, 0.25)
Z95 = 1.959964
NPG = "no population gap"


def tq(level, df):
    return float(stats.t.ppf(0.5 + level / 2, df if np.isfinite(df) else 1e9))


def tost(g, se, df, delta, level):
    t = tq(level, df)
    lo, hi = g - t * se, g + t * se
    if lo > -delta and hi < delta:
        v = "pass"
    elif lo > delta or hi < -delta:
        v = "fail"
    else:
        v = "unresolved"
    return lo, hi, v


def worked():
    tb = pd.read_csv(os.path.join(OUTD, "80a_tolerance_basis.csv"))
    sd = float(tb[(tb.window == "2005-2018") & (tb.population == "all adults 18+")].sd.iloc[0])
    v = pd.read_csv(os.path.join(OUTD, "l2_verdicts.csv"))
    h = v[(v.analysis == "headline") & (v.estimand == "standardised") & (v.scope != "pooled") &
          (v.verdict == NPG)]
    sim = pd.read_csv(os.path.join(OUTD, "l2_sim_gaps.csv"))
    sim["race_set"] = sim.race_set.fillna("n/a")
    rows = []
    for r in h.itertuples():
        rs = "n/a" if "White" in r.contrast else "four"
        s = sim[(sim.corpus == "all") & (sim.panel == "four models") & (sim.race_set == rs) &
                (sim.contrast == r.contrast) & (sim.framing == "combined") & (sim.scope == r.scope)]
        assert len(s) == 1
        s = s.iloc[0]
        assert abs(s.simulated - r.simulated) < 1e-4
        rows.append(dict(dataset="Worked example", unit="PHQ-8 points", outcome="PHQ-8 total",
                         contrast=r.contrast, model=r.scope, g=s.simulated, se_g=s.se, df_g=s.df,
                         p=r.population, se_p=r.se_pop, pop_ci_lo=r.pop_ci_lo,
                         pop_ci_hi=r.pop_ci_hi, ref_sd=sd, family_size=7))
    return rows


def bisbee():
    b = pd.read_csv(os.path.join(EXT, "bisbee", "level2_contrasts_rr1.csv"))
    b = b[(b.framing == "full") & (b.verdict == NPG)]
    gt = pd.read_csv(os.path.join(EXT, "bisbee", "gate_rr1.csv"))
    sd = gt[gt.framing == "full"].set_index("outcome").sd_ref
    rows = []
    for r in b.itertuples():
        t = tq(1 - 2 * 0.05 / r.m_family, np.inf)
        rows.append(dict(dataset="Bisbee", unit="thermometer points (0-100)", outcome=r.outcome,
                         contrast=r.contrast, model="ChatGPT (full prompt)", g=r.g, se_g=r.se_g,
                         df_g=np.inf, p=r.p, se_p=r.se_p, pop_ci_lo=r.p - t * r.se_p,
                         pop_ci_hi=r.p + t * r.se_p, ref_sd=float(sd.loc[r.outcome]),
                         family_size=int(r.m_family)))
    return rows


def argyle():
    e = pd.read_csv(os.path.join(OUTD, "l2_external_r3.csv"))
    e = e[(e.config == "headline") & (e.source == "Argyle") & (e.family == "t0.7_main") &
          (e.verdict == NPG)]
    l3 = pd.read_csv(os.path.join(EXT, "argyle", "level3_groups.csv"))
    l3 = l3[(l3.run == "t0.7_main") & (l3.group == "all:All respondents")].set_index("outcome")
    sd = l3.strict_tol / Z95 * np.sqrt(l3.n)
    rows = []
    for r in e.itertuples():
        outcome = r.row.split(" | ")[0]
        t = tq(1 - 2 * 0.05 / r.family_size, r.df)
        rows.append(dict(dataset="Argyle", unit="outcome units", outcome=outcome,
                         contrast=r.contrast, model="GPT-3 t0.7_main", g=r.g, se_g=r.se_g,
                         df_g=r.df, p=r.p, se_p=r.se_p, pop_ci_lo=r.p - t * r.se_p,
                         pop_ci_hi=r.p + t * r.se_p, ref_sd=float(sd.loc[outcome]),
                         family_size=int(r.family_size)))
    return rows


def main():
    e = pd.read_csv(os.path.join(OUTD, "l2_external_r3.csv"))
    n_oq = int(((e.config == "headline") & (e.source == "OpinionQA") & (e.verdict == NPG)).sum())
    base = pd.DataFrame(worked() + bisbee() + argyle())
    # the stop is the frozen one: the population interval at the family level covers zero
    assert ((base.pop_ci_lo <= 0) & (base.pop_ci_hi >= 0)).all()
    out = []
    for r in base.to_dict("records"):
        for dsd in DELTAS:
            delta = dsd * r["ref_sd"]
            lo, hi, v = tost(r["g"], r["se_g"], r["df_g"], delta, 0.90)
            blo, bhi, bv = tost(r["g"], r["se_g"], r["df_g"], delta,
                                1 - 2 * 0.05 / r["family_size"])
            out.append(dict(r, g_in_sd=r["g"] / r["ref_sd"], delta_sd=dsd, delta_raw=delta,
                            ci90_lo=lo, ci90_hi=hi, verdict_90=v, ci_bonf_lo=blo,
                            ci_bonf_hi=bhi, verdict_bonferroni=bv,
                            analysis="exploratory, post-freeze"))
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(OUTD, "95_null_gap_equivalence.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print("EXPLORATORY, POST-FREEZE: TOST of the simulated gap on contrasts stopped as "
          "'no population gap'")
    print(f"OpinionQA headline rows stopped as no population gap: {n_oq}")
    cols = ["dataset", "outcome", "contrast", "model", "g", "se_g", "ref_sd", "g_in_sd",
            "delta_sd", "delta_raw", "ci90_lo", "ci90_hi", "verdict_90", "verdict_bonferroni"]
    print(res[cols].round(4).to_string(index=False))
    s = (res.groupby(["dataset", "delta_sd", "verdict_90"]).size().unstack(fill_value=0)
         .reindex(columns=["pass", "fail", "unresolved"], fill_value=0))
    s["rows"] = s.sum(axis=1)
    print("\ncounts at 90% (pass = equivalent to zero, fail = gap invented):")
    print(s.to_string())
    sb = (res.groupby(["dataset", "delta_sd", "verdict_bonferroni"]).size().unstack(fill_value=0)
          .reindex(columns=["pass", "fail", "unresolved"], fill_value=0))
    print("\ncounts at the family Bonferroni level (secondary):")
    print(sb.to_string())
    print("\nwritten analysis/brm/95_null_gap_equivalence.csv")


if __name__ == "__main__":
    main()
