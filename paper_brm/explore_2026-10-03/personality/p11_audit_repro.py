"""Reproduction check for the "LLM Psychometric Fidelity Audit" release: recompute, from the released
item-level files, the numbers the authors' notebook (analysis_full.ipynb, committed with outputs)
printed, using only our own parsing of big5_llm.csv and human_data_cleaned.csv.

Authors' numbers read from the notebook outputs (HTML tables, saved cell outputs):
  cell 13  level1_central: human_n, ai_n, human_mean, ai_mean, human_sd, ai_sd per model x trait
           (trait score = mean of the 10 items after the authors' manual reverse key; 1-5 scale)
  cell 20  level3_reliability: Cronbach alpha per model x trait and the human alpha
The notebook displays only the head of each table, so the comparison covers the rows it printed.

Output: out/p11_audit_repro.csv, printed summary.
"""
import io
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "C:/Research/PsychBench/UpdatedRun/paper_brm/external/raw/ipip_audit"
OUT = os.path.join(HERE, "out")
PREFIX = {"Extraversion": "EXT", "Emotional_Stability": "EST", "Agreeableness": "AGR",
          "Conscientiousness": "CSN", "Openness": "OPN"}
REVERSE = {"Extraversion": [2, 4, 6, 8, 10], "Emotional_Stability": [1, 3, 5, 6, 7, 8, 9, 10],
           "Agreeableness": [1, 3, 5, 7], "Conscientiousness": [2, 4, 6, 8], "Openness": [2, 4, 6]}


def alpha(X):
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(0, ddof=1).sum() / X.sum(1).var(ddof=1))


def scored(df, trait):
    p = PREFIX[trait]
    cols = []
    for i in range(1, 11):
        x = df[f"{p}{i}"].to_numpy(float)
        cols.append(6 - x if i in REVERSE[trait] else x)
    return np.column_stack(cols)


def notebook_table(nb, cell):
    html = None
    for o in nb["cells"][cell]["outputs"]:
        if "data" in o and "text/html" in o["data"]:
            html = "".join(o["data"]["text/html"])
    return pd.read_html(io.StringIO(html))[0].drop(columns=["Unnamed: 0"], errors="ignore")


def main():
    os.makedirs(OUT, exist_ok=True)
    nb = json.load(open(os.path.join(RAW, "analysis_full.ipynb"), encoding="utf-8"))
    central = notebook_table(nb, 13)
    rel = notebook_table(nb, 20)
    item_cols = [f"{p}{i}" for p in PREFIX.values() for i in range(1, 11)]
    hum = pd.read_csv(os.path.join(RAW, "human_data_cleaned.csv"), usecols=item_cols)
    sim = pd.read_csv(os.path.join(RAW, "big5_llm.csv"), encoding="utf-8-sig")
    rows = []
    for _, r in central.iterrows():
        trait, model = r["trait"], r["model"]
        h = scored(hum, trait).mean(1)
        a = scored(sim[sim.model == model], trait).mean(1)
        for stat, ours, theirs in [("human_n", len(h), r["human_n"]), ("ai_n", len(a), r["ai_n"]),
                                   ("human_mean", h.mean(), r["human_mean"]), ("ai_mean", a.mean(), r["ai_mean"]),
                                   ("human_sd", h.std(ddof=1), r["human_sd"]), ("ai_sd", a.std(ddof=1), r["ai_sd"])]:
            rows.append(dict(table="level1_central", model=model, trait=trait, statistic=stat, ours=float(ours),
                             authors=float(theirs), abs_diff=abs(float(ours) - float(theirs))))
    for _, r in rel.iterrows():
        trait, model = r["trait"], r["model"]
        a_ours = alpha(scored(sim[sim.model == model], trait))
        h_ours = alpha(scored(hum, trait))
        rows.append(dict(table="level3_reliability", model=model, trait=trait, statistic="ai_alpha", ours=float(a_ours),
                         authors=float(r["alpha"]), abs_diff=abs(a_ours - float(r["alpha"]))))
        rows.append(dict(table="level3_reliability", model=model, trait=trait, statistic="human_alpha", ours=float(h_ours),
                         authors=float(r["human_alpha"]), abs_diff=abs(h_ours - float(r["human_alpha"]))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "p11_audit_repro.csv"), index=False)
    print("notebook central rows read:", len(central), "| reliability rows read:", len(rel))
    print(out.groupby(["table", "statistic"]).agg(n=("abs_diff", "size"), max_abs_diff=("abs_diff", "max")).to_string())
    print("all comparisons within 1e-5:", bool((out.abs_diff < 1e-5).all()), "| max", out.abs_diff.max())
    print(out[out.statistic.isin(["ai_mean", "ai_sd", "ai_alpha", "human_alpha"])].head(14).to_string())


if __name__ == "__main__":
    main()
