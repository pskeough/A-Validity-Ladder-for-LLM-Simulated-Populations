"""SubPOP (Suh et al., ACL 2025): Mistral-7B-v0.1 fine-tuned on Pew answer distributions of 22 subpopulations,
read against its own base model on the paper's 100 contested OpinionQA questions.

Distributions come from external/scripts/subpop_generate.py (SubPOP's own prompt and next-token scoring).
The human side is SubPOP's processed OpinionQA file (data/opinionqa/processed/opinionqa.csv), the target
SubPOP was evaluated against. Read as the models were read by external/scripts/opinionqa_ladder.py:
scalar = ordinal mean rescaled to 0-1 over the question's ordinal range, gaps oriented by the sign of the
human gap per question, g and gamma = mean oriented gap over questions, SEs = across-question SD / sqrt(n),
cov = across-question covariance / n, df = n - 1, Bonferroni family of 8 contrasts.

Training overlap: SubPOP trained on ATP waves 68-131; the 100 questions come from waves 26-92 and none of
their keys is in the training set. The conservative subset also drops the questions fielded inside the
training window (waves 82 and 92) and CONTROLCO_W49, whose stem recurs in training as CONTROLCO_W127.

Families: base / subpop x all / conservative. Receipt: per contrast, the human gap from SubPOP's file is
compared with the gap the paper's OpinionQA script stored from the Meister release (written to the notes;
the two files process Pew differently, so agreement is reported rather than asserted).
"""
import ast
import os
import re

import numpy as np
import pandas as pd

NAME = "subpop2025"
KIND = "fine-tuned simulator (SubPOP) and its base model"
SOURCE = ("Suh, Jahanparast, Moon, Kang & Chang (2025), Language model fine-tuning on scaled survey data for "
          "predicting distributions of public opinions, ACL 2025; weights jjssuh/mistral-7b-v0.1-subpop "
          "(LoRA) on mistralai/Mistral-7B-v0.1; questions from Santurkar et al. (2023) OpinionQA")

EXT = r"C:\Research\PsychBench\UpdatedRun\paper_brm\external"
GEN = os.path.join(EXT, "raw", "subpop_gen", "subpop_distributions.csv")
STORED = os.path.join(EXT, "results", "opinionqa", "level2_contrasts.csv")
CONTRASTS = [("POLPARTY", "Republican", "Democrat"), ("SEX", "Female", "Male"), ("RACE", "Black", "White"),
             ("RACE", "Hispanic", "White"), ("RACE", "Asian", "White"),
             ("EDUCATION", "Less than high school", "College graduate/some postgrad"),
             ("INCOME", "Less than $30,000", "$100,000 or more"), ("CREGION", "South", "Northeast")]
FAMILY_SIZE = 8
DROP_CONSERVATIVE = lambda q: q.endswith("_W82") or q.endswith("_W92") or q == "CONTROLCO_W49"  # noqa: E731


def _vec(s):
    return np.array([float(x) for x in re.findall(r"[-+0-9.eE]+", s)])


def load():
    d = pd.read_csv(GEN)
    # SubPOP's processed file carries IDIMPORT_W43 five times per group under one key and one human
    # distribution, with prompts the model answers differently, so which row matches the paper's item is
    # ambiguous. The question is dropped; every other key must be unique.
    key = ["arm", "qkey", "attribute", "group"]
    d = d[d.qkey != "IDIMPORT_W43"]
    assert not d.duplicated(key).any()
    d = d.reset_index(drop=True)
    d["mean_m"] = np.nan
    d["mean_h"] = np.nan
    for i, r in d.iterrows():
        x = np.array(ast.literal_eval(r.ordinal), float)
        x = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.full_like(x, np.nan)
        pm, ph = _vec(r.dist), _vec(r.human)
        assert len(pm) == len(x) == len(ph), (r.qkey, len(pm), len(x), len(ph))
        d.at[i, "mean_m"] = pm @ x / pm.sum()
        d.at[i, "mean_h"] = ph @ x / ph.sum()
    d = d.dropna(subset=["mean_m", "mean_h"])
    stored = pd.read_csv(STORED)
    rows, notes = [], []
    for arm in ("base", "subpop"):
        a_ = d[d.arm == arm].set_index(["qkey", "attribute", "group"])
        for subset in ("all", "conservative"):
            for gv, a, b in CONTRASTS:
                qs = sorted(set(a_.loc[(slice(None), gv, a), :].index.get_level_values(0)) &
                            set(a_.loc[(slice(None), gv, b), :].index.get_level_values(0)))
                if subset == "conservative":
                    qs = [q for q in qs if not DROP_CONSERVATIVE(q)]
                h = np.array([a_.loc[(q, gv, a), "mean_h"] - a_.loc[(q, gv, b), "mean_h"] for q in qs])
                m = np.array([a_.loc[(q, gv, a), "mean_m"] - a_.loc[(q, gv, b), "mean_m"] for q in qs])
                s = np.sign(h)
                s[s == 0] = 1
                hg, mg = s * h, s * m
                n = len(qs)
                if arm == "base" and subset == "all":
                    lab = f"{a} - {b}".replace("College graduate/some postgrad", "College graduate or some postgrad")
                    sel = stored[(stored.contrast == lab)].human_gap
                    notes.append(f"{a} - {b}: human gap {hg.mean():.4f} over {n} questions; paper's Meister-based "
                                 f"value {sel.iloc[0]:.4f}" if len(sel) else f"{a} - {b}: no stored value")
                rows.append(dict(family=f"{arm} : {subset}", contrast=f"{a} - {b}", n_questions=n,
                                 g=mg.mean(), se_g=mg.std(ddof=1) / np.sqrt(n), gamma=hg.mean(),
                                 se_gamma=hg.std(ddof=1) / np.sqrt(n), cov=np.cov(mg, hg, ddof=1)[0, 1] / n,
                                 df_g=n - 1, df_gamma=n - 1, family_size=FAMILY_SIZE))
    return dict(mode="gaps", gaps=pd.DataFrame(rows), inputs=[GEN, STORED],
                notes="; ".join(notes))
