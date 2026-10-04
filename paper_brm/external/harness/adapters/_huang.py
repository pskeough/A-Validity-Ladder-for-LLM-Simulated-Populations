"""Shared reading of Huang, Wu & Wang (2025) OpinionQA data for the huang2025* adapters.

Contrasts are fixed here, before any ratio was computed (3 Oct 2026). gap = first minus second group.
Every contrast is informed: the synthetic profiles carry all 12 respondent attributes, and each
attribute's value is written into the LLM prompt (Appendix G.1 of the paper; checked on the prompts).

Scalar per question and group: mean of the answer on the paper's 4-point sentiment scale
(-1, -1/3, 1/3, 1) rescaled to 0-1 as (v + 1) / 2. Option 5 "Refused" is coded 0 (the scale midpoint)
by the authors for humans and for LLM answers. Here it is DROPPED on both sides by default
(REFUSED = "drop"), as opinionqa_ladder.py drops options without an ordinal; REFUSED = "mid" keeps it
as the midpoint (the authors' coding) and is run as a sensitivity only. No survey weights are present
in the released human records, so human means are unweighted.

Gaps are then built as in opinionqa_ideal.py / opinionqa_ladder.py: oriented by the sign of the human
gap per question, g and gamma = mean oriented gap over questions, SE = across-question SD / sqrt(n),
cov = across-question covariance / n, df = n - 1; at least MIN_Q questions per contrast.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

EXT = r"C:\Research\PsychBench\UpdatedRun\paper_brm\external"
RAW = os.path.join(EXT, "raw", "huang2025", "data", "OpinionQA")
OQA = os.path.join(RAW, "opinionqa_data.json")
CLEAN = os.path.join(RAW, "synthetic_answers", "clean")
CACHE = os.path.join(EXT, "harness", "_work_huang2025", "cache.pkl")

# (variable, group a, group b): gap = a - b. Fixed before any ratio was computed.
CONTRASTS = [
    ("POLPARTY", "Republican", "Democrat"),
    ("SEX", "Female", "Male"),
    ("RACE", "Black", "White"),
    ("RACE", "Hispanic", "White"),
    ("RACE", "Asian", "White"),
    ("EDUCATION", "Less than high school", "Postgraduate"),
    ("INCOME", "Less than $30,000", "$100,000 or more"),
    ("AGE", "18-29", "65+"),
    ("CREGION", "South", "Northeast"),
]
VARS = sorted({c[0] for c in CONTRASTS})
LLMS = ["claude-3.5-haiku", "deepseek-v3", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-5-mini",
        "llama-3.3-70B-instruct-turbo", "mistral-7B-instruct-v0.3"]
MIN_Q = 20
VALS = np.array([-1.0, -1 / 3, 1 / 3, 1.0])  # the four substantive options, in scale order
REFUSED = "drop"


def _rescale(v):
    return (np.asarray(v, float) + 1.0) / 2.0


def tables():
    """(S, P, A): human records, synthetic profiles, clean LLM answers; cached as a pickle.
    The pickle is a local cache written by this function from the hashed input files only."""
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    d = json.load(open(OQA, encoding="utf-8"))
    S, P = [], []
    for q, v in d.items():
        s = pd.DataFrame(v["survey"])[VARS + ["RESPONSE", "RESPONSE_NUMERIC"]]
        s["q"] = int(q)
        S.append(s)
        p = pd.DataFrame(v["synthetic_profile"])[VARS]
        p["q"] = int(q)
        p["j"] = np.arange(len(p))
        P.append(p)
    del d
    S, P = pd.concat(S, ignore_index=True), pd.concat(P, ignore_index=True)
    A = {}
    for f in sorted(os.listdir(CLEAN)):
        c = json.load(open(os.path.join(CLEAN, f), encoding="utf-8"))
        A[f[:-5]] = {int(q): np.array(v, float) for q, v in c.items()}
    out = (S, P, A)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def inputs():
    return [OQA] + [os.path.join(CLEAN, m + ".json") for m in LLMS]


def _refused_mask(v):
    return np.abs(np.asarray(v, float)) < 1e-9


def human_cells(S):
    """dict var -> DataFrame indexed (q, level): mean (0-1 scale), n, and counts c0..c3 over the 4 options
    (counts are used by the ideal simulators, which only support REFUSED = "drop")."""
    s = S[S.RESPONSE != "Refused"] if REFUSED == "drop" else S
    s = s.assign(x=_rescale(s.RESPONSE_NUMERIC), k=np.rint((s.RESPONSE_NUMERIC.to_numpy(float) + 1.0) * 1.5).astype(int))
    out = {}
    for var in VARS:
        m = s.groupby(["q", var]).x.agg(["mean", "size"]).rename(columns={"size": "n"})
        cnt = pd.crosstab([s.q, s[var]], s.k).reindex(columns=range(4), fill_value=0)
        cnt.columns = [f"c{i}" for i in range(4)]
        out[var] = m.join(cnt)
    return out


def llm_cells(P, answers):
    """dict var -> DataFrame indexed (q, level): mean, n for one LLM's answers (first len(answers) profiles)."""
    rows = []
    for q, a in answers.items():
        rows.append(pd.DataFrame({"q": q, "j": np.arange(len(a)), "v": a}))
    a = pd.concat(rows, ignore_index=True)
    d = P.merge(a, on=["q", "j"], how="inner")
    if REFUSED == "drop":
        d = d[~_refused_mask(d.v)]
    d = d.assign(x=_rescale(d.v))
    out = {}
    for var in VARS:
        out[var] = d.groupby(["q", var]).x.agg(["mean", "size"]).rename(columns={"size": "n"})
    return out


def profile_cells(P, n_first):
    """dict var -> Series indexed (q, level): number of the first n_first profiles per question in each group."""
    d = P[P.j < n_first]
    return {var: d.groupby(["q", var]).size().rename("n").to_frame() for var in VARS}


def build_gaps(H, M, family, extra=None, min_h=2, min_m=1):
    """Gaps rows for one family: H and M are var -> cell tables with 'mean' and 'n'."""
    rows = []
    for var, a, b in CONTRASTS:
        def side(tab, g):
            return tab[var].xs(g, level=1)[["mean", "n"]]
        d = (side(H, a).add_prefix("ha_").join(side(H, b).add_prefix("hb_"), how="inner")
             .join(side(M, a).add_prefix("ma_"), how="inner").join(side(M, b).add_prefix("mb_"), how="inner"))
        d = d[(d.ha_n >= min_h) & (d.hb_n >= min_h) & (d.ma_n >= min_m) & (d.mb_n >= min_m)].sort_index()
        n = len(d)
        if n < MIN_Q:
            continue
        dh = (d.ha_mean - d.hb_mean).to_numpy()
        dm = (d.ma_mean - d.mb_mean).to_numpy()
        s = np.sign(dh)
        s[s == 0] = 1
        hg, mg = s * dh, s * dm
        rows.append(dict(family=family, contrast=f"{a} - {b}", n_questions=n,
                         g=mg.mean(), se_g=mg.std(ddof=1) / np.sqrt(n),
                         gamma=hg.mean(), se_gamma=hg.std(ddof=1) / np.sqrt(n),
                         cov=np.cov(mg, hg, ddof=1)[0, 1] / n, df_g=n - 1, df_gamma=n - 1,
                         median_n_a=float(d.ma_n.median()), median_n_b=float(d.mb_n.median()),
                         share_q_model_gap_negative=float((mg < 0).mean()), **(extra or {})))
    return pd.DataFrame(rows)


def with_family_size(df, size=None):
    df = df.copy()
    if size is None:
        df["family_size"] = df.groupby("family")["contrast"].transform("size")
    else:
        df["family_size"] = size
    return df


def resample_cells(H, rng, size_from=None):
    """Fresh sample of the human respondents of each group: multinomial over the 4 options, which is
    resampling with replacement from the group's own answers. size_from: None = the group's own human
    n; else a var -> table of sizes (profile counts) giving the draw size per (q, level)."""
    out = {}
    for var in VARS:
        h = H[var]
        n_h = h["n"].to_numpy()
        cnt = h[[f"c{i}" for i in range(4)]].to_numpy(float)
        if size_from is None:
            n_draw = n_h
        else:
            n_draw = size_from[var]["n"].reindex(h.index).fillna(0).to_numpy(int)
        means = np.full(len(h), np.nan)
        x = np.array([0.0, 1 / 3, 2 / 3, 1.0])
        for i in range(len(h)):
            if n_draw[i] >= 1 and n_h[i] >= 1:
                means[i] = (rng.multinomial(int(n_draw[i]), cnt[i] / n_h[i]) @ x) / n_draw[i]
        o = pd.DataFrame({"mean": means, "n": n_draw}, index=h.index)
        out[var] = o.dropna()
    return out
