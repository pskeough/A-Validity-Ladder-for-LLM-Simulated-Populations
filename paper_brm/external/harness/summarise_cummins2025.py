"""Counts for cummins2025 and cummins2025_splithalf, read from the harness CSVs. Prints; writes
results/<name>/summary_*.csv."""
import os

import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)


def vc(s):
    return s.value_counts().to_dict()


m = pd.read_csv(os.path.join(R, "cummins2025", "l2_models.csv"))
c = pd.read_csv(os.path.join(R, "cummins2025", "l2_contrasts.csv"), keep_default_na=False)
l3 = pd.read_csv(os.path.join(R, "cummins2025", "l3.csv"), keep_default_na=False)
print("== MAIN level 2")
print("contrast readings", len(c), "families", len(m))
print("labels", vc(c.label))
print("verdicts", vc(c.verdict))
print("family verdicts", vc(m.verdict))
print("kept count per family", vc(m.kept))
print("contrasts per family", vc(m.contrasts))
print("stopped no_pop_gap per family", vc(m.stopped_no_population_gap))
print("stopped ref_imprecise per family", vc(m.stopped_reference_imprecise))
print("families with all contrasts stopped:", int(((m.stopped_no_population_gap + m.stopped_reference_imprecise) == m.contrasts).sum()))
print("not kept per family", vc(m["not kept"]), "unresolved per family", vc(m["unresolved"]))
# human reference by contrast and outcome
ref = c.drop_duplicates(["outcome", "attribute"])[["outcome", "attribute", "gamma", "se_gamma", "pop_ci_lo", "pop_ci_hi", "k_rel_halfwidth", "n_hi", "n_lo"]]
print(ref.to_string(index=False))
print(c.groupby(["outcome", "attribute"]).verdict.value_counts().unstack(fill_value=0))
print(c.groupby(["outcome", "attribute"]).label.value_counts().unstack(fill_value=0))
# by informed, by demographics
print(c.groupby("informed").label.value_counts().unstack(fill_value=0))
print(c.groupby("demographics_used").label.value_counts().unstack(fill_value=0))
mm = m.merge(c[["family", "demographics_used", "model", "authors_excluded"]].drop_duplicates("family"), on="family")
print(mm.groupby("demographics_used").verdict.value_counts().unstack(fill_value=0))
print("kept>0 families:")
print(mm[mm.kept > 0][["family", "kept", "contrasts", "verdict", "not kept", "unresolved"]].to_string(index=False))
print("families with a 'not kept' contrast:", int((m["not kept"] > 0).sum()))
print("authors-excluded config:", mm[mm.authors_excluded].family.tolist(), mm[mm.authors_excluded].verdict.tolist())
print("without it:", vc(mm[~mm.authors_excluded].verdict))
print("== MAIN level 3")
print("rows", len(l3), "verdicts", vc(l3.verdict))
print(l3.groupby("outcome").verdict.value_counts().unstack(fill_value=0))
fam3 = l3.pivot(index="family", columns="outcome", values="verdict")


def f3(r):
    v = [x for x in r if isinstance(x, str)]
    return "fail" if "fail" in v else ("pass" if v and all(x == "pass" for x in v) else "unresolved")


fam3["family"] = fam3.apply(f3, axis=1)
print("family-level L3 (both outcomes)", vc(fam3["family"]))
print("l3 by demographics", l3.groupby("demographics_used").verdict.value_counts().unstack(fill_value=0))
print("resid in SD units: ", l3.groupby("outcome").resid_sd_units.describe())
print("== SPLIT HALF")
sm = pd.read_csv(os.path.join(R, "cummins2025_splithalf", "l2_models.csv"))
sc = pd.read_csv(os.path.join(R, "cummins2025_splithalf", "l2_contrasts.csv"), keep_default_na=False)
sl3 = pd.read_csv(os.path.join(R, "cummins2025_splithalf", "l3.csv"))
print("families", len(sm), "contrast readings", len(sc), "contrasts per family", vc(sm.contrasts))
print("family verdicts", vc(sm.verdict), {k: v / len(sm) for k, v in vc(sm.verdict).items()})
print("contrast labels", vc(sc.label), {k: round(v / len(sc), 4) for k, v in vc(sc.label).items()})
print("contrast verdicts", vc(sc.verdict))
print("kept per family", vc(sm.kept))
print("not kept per family", vc(sm["not kept"]))
print("all stopped families", int(((sm.stopped_no_population_gap + sm.stopped_reference_imprecise) == sm.contrasts).sum()))
print(sc.groupby(["outcome", "attribute"]).label.value_counts().unstack(fill_value=0))
print(sc.groupby(["outcome", "attribute"]).verdict.value_counts().unstack(fill_value=0))
print("gamma exactly 0 nudged:", int((sc.gamma_raw == 0).sum()))
print("L3 split-half verdicts", vc(sl3.verdict))
print(sl3.groupby("outcome").verdict.value_counts().unstack(fill_value=0))
sf3 = sl3.pivot(index="family", columns="outcome", values="verdict")
sf3["family"] = sf3.apply(f3, axis=1)
print("split-half family-level L3", vc(sf3["family"]))
# summary csvs
rows = []
for name, d in [("cummins2025", m), ("cummins2025_splithalf", sm)]:
    for v, n in d.verdict.value_counts().items():
        rows.append(dict(adapter=name, level="2 family", verdict=v, n=int(n)))
for name, d in [("cummins2025", c), ("cummins2025_splithalf", sc)]:
    for v, n in d.label.value_counts().items():
        rows.append(dict(adapter=name, level="2 contrast label", verdict=v, n=int(n)))
for name, d in [("cummins2025", l3), ("cummins2025_splithalf", sl3)]:
    for v, n in d.verdict.value_counts().items():
        rows.append(dict(adapter=name, level="3 row", verdict=v, n=int(n)))
pd.DataFrame(rows).to_csv(os.path.join(R, "cummins2025", "summary_counts.csv"), index=False)
print("== MAIN descriptive (reading before the stops; not a verdict)")
print("reading", vc(c.reading))
print(c.groupby("informed").reading.value_counts().unstack(fill_value=0))
print("k_rel_halfwidth by outcome/attribute (same for every config):")
print(c.groupby(["outcome", "attribute"]).k_rel_halfwidth.first())
print("se_g == 0 readings:", int((c.se_g == 0).sum()), "configs with any:", c[c.se_g == 0].family.nunique())
print("g range by outcome/attribute")
print(c.groupby(["outcome", "attribute"]).g.agg(["min", "median", "max"]))
