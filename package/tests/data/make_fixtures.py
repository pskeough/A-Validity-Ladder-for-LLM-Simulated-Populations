"""Copy the stored outputs the reproduction tests check against into tests/data.

Run from the repository (PsychBench/UpdatedRun) once; the copies let the tests run from a clone
that does not carry analysis/brm. Every fixture is a verbatim row subset of the named file.

  gate_dstudy.csv              analysis/brm/gate_dstudy.csv (all rows)
  91a_core_regression.csv      analysis/brm/91a_core_regression.csv (all rows)
  l2_verdicts_headline.csv     analysis/brm/l2_verdicts.csv, analysis in {headline, band_symmetric,
                               unadjusted_90}
  l2_headline.csv              analysis/brm/l2_headline.csv (all rows)
  l2_sim_gaps.csv              analysis/brm/l2_sim_gaps.csv, corpus all, panel four models
                               (inputs at 6 dp; l2_verdicts.csv rounds them to 4 dp)
  l2_reference.csv             analysis/brm/l2_reference.csv (all rows)
  l2_external_r3_headline.csv  analysis/brm/l2_external_r3.csv, config == headline
  bisbee_level2_rr1.csv        paper_brm/external/results/bisbee/level2_contrasts_rr1.csv
  l2_threeway_rows.csv         paper_brm/external/results/l2_threeway_rows.csv
  l3_s1_primary_ps.csv         analysis/brm/80c_l3_results.csv, spec S1 primary, corpus all,
                               estimand PS
  l3_tolerance_basis.csv       analysis/brm/80a_tolerance_basis.csv (all rows)
  l1_personfit_tolerance.csv   analysis/brm/l1_personfit_tolerance.csv (all rows)
  l1_personfit_main.csv        analysis/brm/l1_personfit_main.csv, 2005_2018 weighted GRM, full, all
  l1_grm_params.csv            analysis/brm/l1_grm_params.csv (all rows)
  l4_structure.csv             analysis/brm/l4_structure.csv, subset all, populations NHANES and the
                               eight model x framing rows of the corpus
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BRM = os.path.join(BASE, "analysis", "brm")
EXT = os.path.join(BASE, "paper_brm", "external", "results")


def main():
    def put(df, name):
        df.to_csv(os.path.join(HERE, name), index=False)
        print(f"{name}: {len(df)} rows")

    put(pd.read_csv(os.path.join(BRM, "gate_dstudy.csv")), "gate_dstudy.csv")
    put(pd.read_csv(os.path.join(BRM, "91a_core_regression.csv")), "91a_core_regression.csv")
    v = pd.read_csv(os.path.join(BRM, "l2_verdicts.csv"))
    put(v[v.analysis.isin(["headline", "band_symmetric", "unadjusted_90"])], "l2_verdicts_headline.csv")
    put(pd.read_csv(os.path.join(BRM, "l2_headline.csv")), "l2_headline.csv")
    g = pd.read_csv(os.path.join(BRM, "l2_sim_gaps.csv"))
    put(g[(g.corpus == "all") & (g.panel == "four models")], "l2_sim_gaps.csv")
    put(pd.read_csv(os.path.join(BRM, "l2_reference.csv")), "l2_reference.csv")
    e = pd.read_csv(os.path.join(BRM, "l2_external_r3.csv"))
    put(e[e.config == "headline"], "l2_external_r3_headline.csv")
    put(pd.read_csv(os.path.join(EXT, "bisbee", "level2_contrasts_rr1.csv")), "bisbee_level2_rr1.csv")
    put(pd.read_csv(os.path.join(EXT, "l2_threeway_rows.csv")), "l2_threeway_rows.csv")
    r = pd.read_csv(os.path.join(BRM, "80c_l3_results.csv"), low_memory=False)
    put(r[(r.spec == "S1 primary") & (r.corpus == "all") & (r.estimand == "PS")], "l3_s1_primary_ps.csv")
    put(pd.read_csv(os.path.join(BRM, "80a_tolerance_basis.csv")), "l3_tolerance_basis.csv")
    put(pd.read_csv(os.path.join(BRM, "l1_personfit_tolerance.csv")), "l1_personfit_tolerance.csv")
    p = pd.read_csv(os.path.join(BRM, "l1_personfit_main.csv"))
    put(p[(p.reference == "2005_2018 weighted GRM") & (p["sample"] == "full") & (p.subset == "all")],
        "l1_personfit_main.csv")
    put(pd.read_csv(os.path.join(BRM, "l1_grm_params.csv")), "l1_grm_params.csv")
    s = pd.read_csv(os.path.join(BRM, "l4_structure.csv"))
    put(s[s.subset == "all"], "l4_structure.csv")


if __name__ == "__main__":
    main()
