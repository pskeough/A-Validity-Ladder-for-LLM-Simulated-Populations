"""Receipt for the huang2025 adapter: numbers the paper or data_info.md print, and the overlap with the
OpinionQA release the paper's own OpinionQA adapter uses (Meister et al. 2024 human_resp counts).
Writes results/huang2025/receipt.csv (item, ours, published, source_location)."""
import json
import os

import numpy as np
import pandas as pd

EXT = r"C:\Research\PsychBench\UpdatedRun\paper_brm\external"
RAW = os.path.join(EXT, "raw", "huang2025")
OQA = os.path.join(RAW, "data", "OpinionQA", "opinionqa_data.json")
CLEAN = os.path.join(RAW, "data", "OpinionQA", "synthetic_answers", "clean")
HUM = os.path.join(EXT, "raw", "meister2024", "opinions_qa", "data", "human_resp", "Pew_American_Trends_Panel_disagreement_100")
OUT = os.path.join(EXT, "harness", "results", "huang2025", "receipt.csv")
ATTR12 = ["CREGION", "AGE", "SEX", "EDUCATION", "CITIZEN", "MARITAL", "RELIG", "RELIGATTEND", "POLPARTY",
          "INCOME", "POLIDEOLOGY", "RACE"]

d = json.load(open(OQA, encoding="utf-8"))
rows = []


def add(item, ours, published, loc):
    rows.append(dict(item=item, ours=ours, published=published, source_location=loc))


n_resp, per_q, profiles, surveys = 0, [], [], {}
for q, v in d.items():
    s = pd.DataFrame(v["survey"])
    n_resp += len(s)
    per_q.append(len(s))
    profiles.append(s[ATTR12].drop_duplicates())
    surveys[v["old_id"]] = s
add("OpinionQA responses (all questions)", n_resp, 1476868, "arXiv:2502.17773 Sec 5.1 Datasets: 385 unique questions and 1,476,868 responses")
add("OpinionQA questions", len(d), 385, "arXiv:2502.17773 Sec 5.1 and App. G.1")
add("min responses per question", min(per_q), ">= 400", "arXiv:2502.17773 App. G.1: each question has at least 400 responses")
allp = pd.concat(profiles, ignore_index=True).drop_duplicates()
add("unique respondent profiles (12 attributes, all rows)", len(allp), "32,864 (stated as 'at least')",
    "arXiv:2502.17773 App. G.1: bootstrapping the 32,864 unique real profiles")
no_ref = allp[~(allp == "Refused").any(axis=1)]
add("unique respondent profiles (12 attributes, rows with a Refused attribute removed)", len(no_ref), "32,864 (stated as 'at least')",
    "same; the paper says surveyees with missing information are excluded")
add("synthetic profiles per question", int(pd.Series([len(v["synthetic_profile"]) for v in d.values()]).unique()[0]), 200,
    "data_info.md: opinionqa_data.json synthetic_profile, length 200")
for m in ["claude-3.5-haiku", "deepseek-v3", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-5-mini",
          "llama-3.3-70B-instruct-turbo", "mistral-7B-instruct-v0.3", "random"]:
    c = json.load(open(os.path.join(CLEAN, m + ".json")))
    lens = sorted({len(v) for v in c.values()})
    pub = "200" if m in ("gpt-4o", "random") else "typically 100"
    add(f"clean answers per question: {m}", "/".join(map(str, lens)), pub,
        "data_info.md OpinionQA/synthetic_answers/clean: typically 100, gpt-4o and random 200 (the released files hold 100, 150 or 200)")

# overlap with the Meister et al. (2024) release: per-group option counts for the same Pew question
cells = same = 0
per_question_cells = {}
ratios, dprop = [], []
qs_over = set()
for gv in ["POLPARTY", "SEX", "RACE"]:
    mj = json.load(open(os.path.join(HUM, f"{gv}_data.json"), encoding="utf-8"))
    for q, rec in mj.items():
        if q not in surveys:
            continue
        s = surveys[q]
        qs_over.add(q)
        for grp, cnt in rec.items():
            if grp in ("MC_options", "question_text") or grp in ("Refused",):
                continue
            ours = s[s[gv] == grp].RESPONSE.value_counts()
            tot_m, tot_o = sum(cnt.values()), int(ours.sum())
            for opt, c in cnt.items():
                cells += 1
                eq = int(int(ours.get(opt, 0)) == int(c))
                same += eq
                qcell = per_question_cells.setdefault(q, [0, 0])
                qcell[0] += 1
                qcell[1] += eq
            if tot_m > 0 and tot_o > 0:
                ratios.append(tot_o / tot_m)
                pm = np.array([cnt.get(o, 0) / tot_m for o in cnt])
                po = np.array([ours.get(o, 0) / tot_o for o in cnt])
                dprop.append(np.abs(pm - po).max())
add("questions in both Huang (385) and Meister (100) releases", len(qs_over), "n/a (derived)", "old_id of Huang vs keys of Meister human_resp/*_data.json")
add("group x option count cells identical to Meister counts", f"{same} of {cells}", "n/a (derived)",
    "Meister human_resp POLPARTY/SEX/RACE_data.json vs Huang survey records, same question, same group")
n_exact = sum(1 for n, k in per_question_cells.values() if n == k)
add("overlap questions whose every cell is identical", f"{n_exact} of {len(per_question_cells)}", "n/a (derived)",
    "same; the other 7 questions differ in respondent counts in some groups (the paper says it excluded surveyees with missing information; cause not traced)")
add("group size ratio Huang/Meister: median over question x group", round(float(np.median(ratios)), 4), "n/a (derived)", "same")
add("group size ratio Huang/Meister: min and max", f"{min(ratios):.4f} / {max(ratios):.4f}", "n/a (derived)", "same")
add("max abs difference in option proportion: median over question x group", round(float(np.median(dprop)), 5), "n/a (derived)", "same")
add("max abs difference in option proportion: maximum over question x group", round(float(np.max(dprop)), 5), "n/a (derived)", "same")
pd.DataFrame(rows).to_csv(OUT, index=False)
print(pd.DataFrame(rows).to_string())
