"""Shared loader for the mechanism diagnostics on the Twin-2K-500 digital twins.

Read-only: it reads the raw release under paper_brm/external/raw/twin2k and nothing else. The parsing
rules copy scripts/93_twin2k_ladder.py (formatted CSVs are joined on TWIN_ID, out-of-range codes are
set missing). One process, one thread, no API calls.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))            # .../paper_brm
RAW = os.path.join(BASE, "external", "raw", "twin2k")
LSR = os.path.join(RAW, "LLM_simulation_results")
SPECD = os.path.join(LSR, "llm_simulations_all_specifications")
RES = os.path.join(BASE, "external", "results", "twin2k")
OUT = HERE

HEADLINE = "text_gpt41mini"
CEILING = "human_retest_wave1_3"
SPECS = [
    ("text_gpt41mini", "Text Persona - GPT4.1-mini", 71.72),
    ("text_gemini25flash", "Text Persona - Gemini-Flash2.5", 69.40),
    ("json_gpt41mini", "JSON Persona - GPT4.1-mini", 70.48),
    ("json_gpt41", "JSON Persona - GPT4.1", 71.05),
    ("summary_gpt41mini", "Persona Summary - GPT4.1-mini", 68.02),
    ("summary_json_gpt41mini", "Persona Summary - JSON Persona - GPT4.1-mini", 67.88),
    ("text_reasoning_gpt41mini", "Text Persona (Reasoning) - GPT4.1-mini", 70.39),
    ("text_repeatq_gpt41mini", "Text Persona (Repeating Questions) - GPT4.1-mini", 70.45),
    ("text_t07_gpt41mini", "Text Persona (Default Temperature) - GPT4.1-mini", 71.24),
    ("json_predout_gpt41mini", "JSON Persona (Predicted Output) - GPT4.1-mini", 69.00),
    ("json_predout_gpt41", "JSON Persona (Predicted Output) - GPT4.1", 71.92),
    ("finetune500_gpt41mini", "LLM Finetuning (500 training samples) - GPT4.1-mini", 69.61),
    ("demog_only_gpt41mini", "Demographics Only - GPT4.1-mini", np.nan),
]
# the 8 certifiable Republican - Democrat contrasts (level2_certifiable.csv)
ITEMS = ["QID287_1", "QID287_2", "QID287_3", "QID287_4", "QID287_6", "QID287_7", "QID287_10", "QID287_11"]

MAPPING = json.load(open(os.path.join(LSR, "wave4_formatted_to_catalog_mapping.json"), encoding="utf-8"))
F2C = {m["formatted_column"]: m["catalog_csv_column"] for m in MAPPING}


def _catalog():
    cat = json.load(open(os.path.join(RAW, "question_catalog.json"), encoding="utf-8"))
    by = {}
    for q in cat:
        for c in q.get("csv_columns", []):
            by[c] = q
    return by


CAT = _catalog()


def valid_range(col):
    q = CAT[col]
    t = q.get("QuestionType")
    if t == "Slider":
        return 0.0, 100.0
    if t == "MC":
        return 1.0, float(len(q["Options"]))
    if t == "Matrix":
        return 1.0, float(len(q["Columns"]))
    return -np.inf, np.inf


def read_formatted(path):
    f = pd.read_csv(path, low_memory=False, dtype=str)
    assert f.iloc[0]["TWIN_ID"] == "TWIN_ID", path
    f = f.iloc[1:]
    f.index = f.TWIN_ID.astype(int)
    assert f.index.is_unique, path
    return f


def twin_answers(path, cols=None):
    """Twin answers by catalog column; out-of-range codes become missing."""
    f = read_formatted(path)
    out = {}
    for fc, cc in F2C.items():
        if cols is not None and cc not in cols:
            continue
        x = pd.to_numeric(f[fc], errors="coerce")
        lo, hi = valid_range(cc)
        out[cc] = x.where(~(x.notna() & ((x < lo) | (x > hi))))
    return pd.DataFrame(out, index=f.index)


def human(which, cols=None):
    if which == "w4":
        h = pd.read_csv(os.path.join(RAW, "question_catalog_and_human_response_csv", "wave4_response.csv"), low_memory=False)
    else:
        h = pd.read_csv(os.path.join(RAW, "wave1_3_response.csv"), low_memory=False)
    h = h.set_index("pid")
    use = list(F2C.values()) if cols is None else cols
    return h[use].apply(pd.to_numeric, errors="coerce")


def demographics():
    h = pd.read_csv(os.path.join(RAW, "wave1_3_response.csv"), low_memory=False).set_index("pid")
    d = pd.DataFrame(index=h.index)
    d["party"] = h.QID20.map({1: "Republican", 2: "Democrat", 3: "Independent", 4: "Something else"})
    d["ideology"] = pd.to_numeric(h.QID22, errors="coerce")        # 1 very conservative .. 5 very liberal
    return d


def spec_path(folder):
    return os.path.join(SPECD, folder, "csv_comparison", "csv_formatted", "responses_llm_imputed_formatted.csv")


def load_all(cols=None):
    """Return dict key -> DataFrame of answers (index TWIN_ID), plus h4, h13, demo."""
    cols = ITEMS if cols is None else cols
    h4, h13 = human("w4"), human("w13")
    cols_all = None if cols is None else cols
    h4, h13 = h4[cols], h13[cols]
    sims = {k: twin_answers(spec_path(folder), cols) for k, folder, _ in SPECS}
    sims[CEILING] = h13.copy()
    return sims, h4, h13, demographics()


ACC = {k: a for k, _, a in SPECS}
ACC[CEILING] = 81.72
