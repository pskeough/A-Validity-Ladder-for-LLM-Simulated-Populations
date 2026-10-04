"""Ideal-simulator control on OpinionQA (Meister et al. 2024 release, 100 contested Pew questions).

The simulator is a fresh sample of the same people: for each question and group, a multinomial
draw of the group's released size from its released Pew counts. It is read exactly as the
models were read by external/scripts/opinionqa_ladder.py: scalar = ordinal mean rescaled to 0-1
over the question's ordinal range (options without an ordinal dropped), gaps oriented by the sign
of the human gap per question, g and gamma = mean oriented gap over questions, SEs = across-question
SD / sqrt(n), cov = across-question covariance / n, df = n - 1, Bonferroni family of 8 contrasts
(the frozen family; only the 5 contrasts with released counts can be simulated).

Families: rep_000 ... rep_199 (seeds 0..199). Receipt: the human gap from the counts equals the
human_gap the paper's script stored for the same contrast (checked in load()).
"""
import ast
import json
import os

import numpy as np
import pandas as pd

NAME = "opinionqa_ideal"
KIND = "ideal simulator (fresh multinomial sample of the released Pew counts)"
SOURCE = ("Santurkar et al. (2023) OpinionQA; Meister, Guestrin & Hashimoto (2024) release, "
          "github.com/nicolemeister/benchmarking-distributional-alignment commit 36869b5")

EXT = r"C:\Research\PsychBench\UpdatedRun\paper_brm\external"
RAW = os.path.join(EXT, "raw", "meister2024")
HUMAN = os.path.join(RAW, "opinions_qa", "data", "human_resp")
WAVE = "Pew_American_Trends_Panel_disagreement_100"
T0 = os.path.join(RAW, "results", "opinionqa", "express_distribution", "gpt-4", "task0", WAVE, "NONE", "Democrat.json")
STORED = os.path.join(EXT, "results", "opinionqa", "level2_contrasts.csv")
CONTRASTS = [("POLPARTY", "Republican", "Democrat"), ("SEX", "Female", "Male"), ("RACE", "Black", "White"),
             ("RACE", "Hispanic", "White"), ("RACE", "Asian", "White")]
FAMILY_SIZE = 8
REPS = 200


def load():
    info = pd.read_csv(os.path.join(HUMAN, "Pew_American_Trends_Panel_disagreement_500", "info.csv"))
    info = info.drop_duplicates("key").set_index("key")
    t0 = json.load(open(T0, encoding="utf-8"))
    qids = [q for q in t0 if len(set(ast.literal_eval(info.loc[q, "option_ordinal"]))) >= 2]
    counts = {g: json.load(open(os.path.join(HUMAN, WAVE, f"{g}_data.json"), encoding="utf-8"))
              for g in ("POLPARTY", "SEX", "RACE")}

    def omap(q):
        refs = ast.literal_eval(info.loc[q, "references"])
        ords = ast.literal_eval(info.loc[q, "option_ordinal"])
        m = {refs[i]: float(ords[i]) for i in range(len(ords))}
        lo, hi = min(m.values()), max(m.values())
        return {k: (v - lo) / (hi - lo) for k, v in m.items()}

    # per question and group: ordinal positions x and counts (ordinal options only)
    cells = {}
    for gv, a, b in CONTRASTS:
        for grp in (a, b):
            for q in qids:
                c = counts[gv].get(q, {}).get(grp)
                if not c:
                    continue
                om = omap(q)
                names = [k for k in c if k in om]
                cnt = np.array([c[k] for k in names], float)
                if cnt.sum() < 2:
                    continue
                cells[(q, gv, grp)] = (np.array([om[k] for k in names]), cnt)

    stored = pd.read_csv(STORED)
    rows = []
    for rep in range(REPS):
        rng = np.random.default_rng(rep)
        for gv, a, b in CONTRASTS:
            qs = [q for q in qids if (q, gv, a) in cells and (q, gv, b) in cells]
            h, m = [], []
            for q in qs:
                hm, mm = [], []
                for grp in (a, b):
                    x, cnt = cells[(q, gv, grp)]
                    n = int(cnt.sum())
                    hm.append((cnt / n) @ x)
                    mm.append((rng.multinomial(n, cnt / n) / n) @ x)
                h.append(hm[0] - hm[1])
                m.append(mm[0] - mm[1])
            h, m = np.array(h), np.array(m)
            s = np.sign(h)
            s[s == 0] = 1
            hg, mg = s * h, s * m
            n = len(qs)
            if rep == 0:  # receipt against the paper's stored human gap (same orientation and questions)
                sel = stored[(stored.contrast == f"{a} - {b}") & (stored.n_questions == n)]
                if len(sel):
                    assert np.allclose(sel.human_gap, hg.mean(), atol=1e-12), (a, b, sel.human_gap.unique(), hg.mean())
            rows.append(dict(family=f"rep_{rep:03d}", contrast=f"{a} - {b}", n_questions=n,
                             g=mg.mean(), se_g=mg.std(ddof=1) / np.sqrt(n), gamma=hg.mean(),
                             se_gamma=hg.std(ddof=1) / np.sqrt(n), cov=np.cov(mg, hg, ddof=1)[0, 1] / n,
                             df_g=n - 1, df_gamma=n - 1, family_size=FAMILY_SIZE))
    return dict(mode="gaps", gaps=pd.DataFrame(rows), inputs=[T0, STORED] +
                [os.path.join(HUMAN, WAVE, f"{g}_data.json") for g in ("POLPARTY", "SEX", "RACE")],
                notes="ideal simulator; 200 replicate families; family_size 8 as in the frozen OpinionQA families")
