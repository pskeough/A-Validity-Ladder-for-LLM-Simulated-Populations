"""Receipt for the twin2k_mega adapter: numbers the Mega Study authors print in their repository
(mega_study_evaluation/meta_analysis_results/combined_all_specifications_meta_analysis.csv, one row per
study x specification x variable: human-twin correlation, accuracy, mean_human, mean_twin, std_human,
std_twin, sample size) recomputed from the raw consolidated answer files in results.zip, for every
variable of the authors' list that is a raw survey column (not a variable the authors derive in a
study script). Reads files only; writes results/twin2k_mega/receipt.csv with columns
item, ours, published, source_location (plus match).

Usage: python twin2k_mega_receipt.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adapters import twin2k_mega as M  # noqa: E402

PUB = os.path.join(M.MEGA, "github", "Twin-2K-500-Mega-Study-main", "mega_study_evaluation",
                   "meta_analysis_results", "combined_all_specifications_meta_analysis.csv")
PUB_REL = "Twin-2K-500-Mega-Study/mega_study_evaluation/meta_analysis_results/combined_all_specifications_meta_analysis.csv"
SMAP = {"hiring algorithms": "hiring_algorithms", "Defaults": "default_eric", "context effects": "context_effects",
        "Consumer Minimalism": "consumer_minimalism", "Privacy": "privacy", "quantitative intuition": "quantitative_intuition",
        "Affective Primes": "affective_priming", "Accuracy Nudge": "accuracy_nudges", "idea evaluation": "idea_evaluation",
        "infotainment news sharing": "infotainment", "recommendation systems": "recommendation_algorithms",
        "Promiscuous donors": "promiscuous_donors", "Story Beliefs": "story_beliefs", "Obedient twins": "obedient_twins",
        "Preferences for redistribution": "preference_redistribution", "Junk Fees": "junk_fees",
        "Digital certificates for luxury consumption": "digital_certification", "idea generation": "idea_generation",
        "Targeting fairness": "targeting_fairness"}
CORR = "correlation between the responses from humans vs. their twins"
ACC = "accuracy between humans vs. their twins"
TOL = 5e-6
# recodes the authors apply before analysis (mega_study_evaluation/hiring_algorithms/mega_study_evaluation.py,
# lines 116-126): Q9 {3: 1, 4: 2, 2: 3, 1: 4}, Q14 {1: 1, 3: 2, 4: 3, 2: 4}
RECODE = {("hiring_algorithms", "Q9"): {3: 1, 4: 2, 2: 3, 1: 4}, ("hiring_algorithms", "Q14"): {1: 1, 3: 2, 4: 3, 2: 4}}


def main():
    pub = pd.read_csv(PUB, low_memory=False)
    rows = []
    n_var = 0
    for spec in ["full_persona_without_reasoning", "GPT5", "temperature_zero"]:
        p = pub[pub["persona specification"] == spec]
        for sname, d in p.groupby("study name"):
            study = SMAP[sname]
            specs, _ = M.spec_dirs(study)
            if spec not in specs:
                continue
            H, T, _ = M.pair_frames(study, specs[spec], list(d["variable name"].unique()))
            for r in d.itertuples(index=False):
                v = r[list(d.columns).index("variable name")]
                if v not in H.columns:
                    continue            # a variable the authors derive in a study script
                n_var += 1
                h, t = H[v], T[v]
                if (study, v) in RECODE:
                    h, t = h.map(RECODE[(study, v)]), t.map(RECODE[(study, v)])
                ok = h.notna() & t.notna()
                loc = f"{PUB_REL}{' (authors recode ' + v + ' in hiring_algorithms/mega_study_evaluation.py)' if (study, v) in RECODE else ''}; row study name={sname!r}, persona specification={spec!r}, variable name={v!r}"
                get = lambda c: r[list(d.columns).index(c)]
                ours = {
                    "correlation": (float(np.corrcoef(h[ok], t[ok])[0, 1]) if t[ok].std() > 0 else np.nan, get(CORR)),
                    "mean_human": (float(h.mean()), get("mean_human")),
                    "mean_twin": (float(t.mean()), get("mean_twin")),
                    "std_human": (float(h.std()), get("std_human")),
                    "std_twin": (float(t.std()), get("std_twin")),
                    "sample size": (float(ok.sum()), get("sample size")),
                }
                for k, (o, pb) in ours.items():
                    same = (np.isnan(o) and (pd.isna(pb))) or (not np.isnan(o) and not pd.isna(pb) and abs(o - float(pb)) < TOL * max(1, abs(float(pb))))
                    rows.append(dict(item=f"{study} / {v} / {spec}: {k}", ours=o, published=pb, source_location=loc, match=bool(same)))
    r = pd.DataFrame(rows)
    out = os.path.join(M.OUT, "receipt.csv")
    os.makedirs(M.OUT, exist_ok=True)
    r.to_csv(out, index=False)
    print(f"{n_var} variable-specification rows, {len(r)} numbers, {int(r.match.sum())} match within {TOL}")
    print(r.groupby(r["item"].str.split(": ").str[-1]).match.agg(["sum", "size"]))
    bad = r[~r.match]
    if len(bad):
        print(bad.head(30).to_string())


if __name__ == "__main__":
    main()
