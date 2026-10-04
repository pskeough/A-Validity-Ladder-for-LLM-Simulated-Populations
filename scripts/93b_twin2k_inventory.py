"""Inventory of wave 4 of Twin-2K-500 for script 93: every column of the wave-4 mapping with its
catalog entry (block, type, text, options or scale), and its distribution in the human wave-4 CSV,
the humans' own waves 1-3 answers, and the default GPT-4.1-mini simulation.

Writes paper_brm/external/results/twin2k/93b_wave4_inventory.csv. Read-only on raw/.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(BASE, "paper_brm", "external", "raw", "twin2k")
OUT = os.path.join(BASE, "paper_brm", "external", "results", "twin2k")
SIM = os.path.join(RAW, "LLM_simulation_results", "GPT4.1-mini-simulation-llm-vs-human")


def catalog():
    cat = json.load(open(os.path.join(RAW, "question_catalog.json"), encoding="utf-8"))
    by = {}
    for q in cat:
        for c in q.get("csv_columns", []):
            by[c] = q
    return by


def main():
    mp = json.load(open(os.path.join(RAW, "LLM_simulation_results", "wave4_formatted_to_catalog_mapping.json"), encoding="utf-8"))
    cat = catalog()
    h4 = pd.read_csv(os.path.join(RAW, "question_catalog_and_human_response_csv", "wave4_response.csv"))
    f4 = pd.read_csv(os.path.join(SIM, "responses_wave4_formatted.csv"), low_memory=False)
    f13 = pd.read_csv(os.path.join(SIM, "responses_wave1_3_formatted.csv"), low_memory=False)
    fl = pd.read_csv(os.path.join(SIM, "responses_llm_imputed_formatted.csv"), low_memory=False)
    print("shapes h4", h4.shape, "f4", f4.shape, "f13", f13.shape, "llm", fl.shape)
    print("formatted columns not in mapping:", [c for c in f4.columns if c not in {m["formatted_column"] for m in mp}][:60])
    rows = []
    for m in mp:
        fc, cc = m["formatted_column"], m["catalog_csv_column"]
        q = cat.get(cc, {})
        def desc(s):
            x = pd.to_numeric(s, errors="coerce")
            return dict(n=int(x.notna().sum()), mean=x.mean(), sd=x.std(), nuniq=int(x.nunique()),
                        min=x.min(), max=x.max())
        dh = desc(h4[cc]) if cc in h4 else {}
        df4 = desc(f4[fc]) if fc in f4 else {}
        d13 = desc(f13[fc]) if fc in f13 else {}
        dl = desc(fl[fc]) if fc in fl else {}
        opts = q.get("Options") or q.get("Columns") or q.get("Statements") or q.get("Range")
        rows.append(dict(formatted_column=fc, qid=m["QuestionID"], csv_col=cc, block=q.get("BlockName"),
                         qtype=q.get("QuestionType"), selector=(q.get("Settings") or {}).get("Selector"),
                         text=(q.get("QuestionText") or "")[:300].replace("\n", " "),
                         rows=json.dumps(q.get("Rows"))[:300] if q.get("Rows") else "",
                         options=json.dumps(opts)[:400] if opts else "",
                         **{f"h4_{k}": v for k, v in dh.items()}, **{f"f4_{k}": v for k, v in df4.items()},
                         **{f"w13_{k}": v for k, v in d13.items()}, **{f"llm_{k}": v for k, v in dl.items()}))
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT, "93b_wave4_inventory.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 500)
    print(inv[["formatted_column", "csv_col", "block", "qtype", "selector", "h4_n", "h4_mean", "h4_nuniq",
               "f4_n", "f4_mean", "w13_n", "w13_mean", "llm_n", "llm_mean"]].to_string())


if __name__ == "__main__":
    main()
