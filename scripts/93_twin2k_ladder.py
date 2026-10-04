"""Validity ladder (frozen rules, paper_brm/LADDER_SPEC.md v1.0) on the digital twins released with
Twin-2K-500 (Toubia, Gui, Peng, Merlau, Li and Chen, 2025; arXiv 2505.17479; Hugging Face dataset
LLM-Digital-Twin/Twin-2K-500, CC BY 4.0). Raw files fetched by 93a_twin2k_fetch.py; SOURCE.txt holds
their URLs and SHA-256.

Design. 2,058 US respondents answered four survey waves. The heuristics-and-biases experiments and
the pricing study of waves 1-3 were held out; each digital twin was built from one respondent's other
waves 1-3 answers and answered the held-out wave-4 questions (126 columns). Wave 4 repeated those
questions two weeks later, in the same between-subject condition. Every twin therefore has a human
twin. The design follows the Bisbee run (external/scripts/bisbee_ladder.py), the paper's other
same-respondent demonstration:
  reference   the same respondents' own wave-4 answers, unweighted (headline). The authors' own
              accuracy metric scores the twins against the waves 1-3 answers; that reference is run
              as a sensitivity ("ref_wave1_3").
  attributes  the waves 1-3 demographics: sex (QID12), age band (QID13), education (QID14), race or
              origin (QID15), party (QID20), family income (QID21).
  contrasts   the seven Bisbee contrasts: female - male, Black - White, Hispanic - White, high school
              or less - college graduate or more, 18-29 - 65+, Republican - Democrat, income under
              $30,000 - $100,000 or more.
  outcomes    every wave-4 column that is a single numeric or ordinal item (70), plus three
              composites: share of the 40 pricing items answered "would purchase" (prices were drawn
              at random per respondent, so single pricing items are not separate targets), and the
              authors' "maximiser" indicator for each probability-matching problem (every trial
              predicts the majority outcome; their P_MAX). Exclusions with reasons in exclusions.csv.

Rungs (per simulator specification; the specification is the model x framing family):
  Gate     not applicable: every specification gives one answer per twin (temperature 0, or 0.7
           for one run). A temperature-0 / temperature-0.7 comparison is written as a diagnostic,
           not as the gate (gate_diagnostic_temperature.csv).
  Level 1  not applicable: wave 4 holds no multi-item scale of one construct (see TWIN2K.md).
  Level 2  persona-mean gap g against the same respondents' gap p on the pairs with both answers;
           respondent bootstrap B = 2,000 that keeps each twin with its human (same draws in every
           specification, seeded per outcome x contrast); 78e.r3cov with df infinite, Fieller
           interval, boundaries -0.25, 0.25, 0.75, 1.25, stops 1 and 2; family = one specification,
           Bonferroni over its contrasts. Sensitivity: family = one outcome (7 contrasts).
           A contrast is formed when both groups hold at least 30 pairs.
  Level 3  residual of twin answers against the same respondents' answers, per group (all and the
           13 contrast groups); paired SE; TOST at 90% against 0.5 reference SD (default) and
           0.2 SD; reference SD = SD of the wave-4 answers of all respondents to that outcome.
  Level 4  not applicable: as level 1.
Ceiling. The respondents' own waves 1-3 answers to the wave-4 questions run as a "simulator" against
the wave-4 reference: what a simulator that recalls each person perfectly, up to that person's own
two-week test-retest noise, obtains on the same rungs.

Receipts (receipts.csv; the script stops on a data-integrity mismatch):
  - every specification's wave-4 and waves 1-3 human columns equal wave4_response.csv and
    wave1_3_response.csv, joined on TWIN_ID = pid (the three formatted files are not in the same
    row order: the twin and waves 1-3 files share one order, the wave-4 file has another);
  - the default folder's twin file is byte-identical to "Text Persona - GPT4.1-mini";
  - human means the paper prints for wave 4 and waves 1-3 (Appendix A.6-A.7) and twin shares it
    prints in Section 5.2;
  - the authors' column-level correlations and MAD (mad_accuracy_summary.xlsx) recomputed for
    every specification, and their task-level accuracy (71.72% for the default) where it can be.

Outputs: paper_brm/external/results/twin2k/. Seed 20261093. CPU only, one process.
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(BASE, "paper_brm", "external", "raw", "twin2k")
LSR = os.path.join(RAW, "LLM_simulation_results")
SPECD = os.path.join(LSR, "llm_simulations_all_specifications")
OUT = os.path.join(BASE, "paper_brm", "external", "results", "twin2k")
os.makedirs(OUT, exist_ok=True)
SEED = 20261093
B_L2 = 2000
MIN_N = 30
Z90 = 1.6448536269514722
HEADLINE = "text_gpt41mini"
CEILING = "human_retest_wave1_3"
# key, folder, model, persona input, temperature, note
SPECS = [
    ("text_gpt41mini", "Text Persona - GPT4.1-mini", "GPT-4.1-mini", "text persona", "0", "authors' default (headline)"),
    ("text_gemini25flash", "Text Persona - Gemini-Flash2.5", "Gemini-2.5-Flash", "text persona", "0", ""),
    ("json_gpt41mini", "JSON Persona - GPT4.1-mini", "GPT-4.1-mini", "JSON persona", "0", "released for a subset"),
    ("json_gpt41", "JSON Persona - GPT4.1", "GPT-4.1", "JSON persona", "0", ""),
    ("summary_gpt41mini", "Persona Summary - GPT4.1-mini", "GPT-4.1-mini", "persona summary", "0", ""),
    ("summary_json_gpt41mini", "Persona Summary - JSON Persona - GPT4.1-mini", "GPT-4.1-mini",
     "persona summary + JSON persona", "0", ""),
    ("text_reasoning_gpt41mini", "Text Persona (Reasoning) - GPT4.1-mini", "GPT-4.1-mini", "text persona, reasoning prompt", "0", ""),
    ("text_repeatq_gpt41mini", "Text Persona (Repeating Questions) - GPT4.1-mini", "GPT-4.1-mini",
     "text persona, repeat-the-question prompt", "0", ""),
    ("text_t07_gpt41mini", "Text Persona (Default Temperature) - GPT4.1-mini", "GPT-4.1-mini", "text persona", "0.7", ""),
    ("json_predout_gpt41mini", "JSON Persona (Predicted Output) - GPT4.1-mini", "GPT-4.1-mini",
     "JSON persona, Predicted Output", "0", ""),
    ("json_predout_gpt41", "JSON Persona (Predicted Output) - GPT4.1", "GPT-4.1", "JSON persona, Predicted Output", "0", ""),
    ("finetune500_gpt41mini", "LLM Finetuning (500 training samples) - GPT4.1-mini", "GPT-4.1-mini (fine-tuned on 500)",
     "text persona", "0", "test respondents only"),
    ("demog_only_gpt41mini", "Demographics Only - GPT4.1-mini", "GPT-4.1-mini", "demographics only", "0",
     "released, not in the paper's Table 2"),
]
# the paper's Table 2 accuracies, for the summary table (receipt: default recomputed below)
PAPER_ACC = {"text_gpt41mini": 71.72, "text_gemini25flash": 69.40, "json_gpt41mini": 70.48, "json_gpt41": 71.05,
             "summary_gpt41mini": 68.02, "summary_json_gpt41mini": 67.88, "text_reasoning_gpt41mini": 70.39,
             "text_repeatq_gpt41mini": 70.45, "text_t07_gpt41mini": 71.24, "json_predout_gpt41mini": 69.00,
             "json_predout_gpt41": 71.92, "finetune500_gpt41mini": 69.61, CEILING: 81.72}
CONTRASTS = [("sex", "Female", "Male"), ("race", "Black", "White"), ("race", "Hispanic", "White"),
             ("educ", "High school or less", "College graduate or more"), ("age", "18-29", "65+"),
             ("party", "Republican", "Democrat"), ("income", "Less than $30,000", "$100,000 or more")]
PRICING = [f"QID9_{i}" for i in range(1, 41)]
PM_CARD = [f"QID198_{i}" for i in range(1, 11)]
PM_DICE = [f"QID203_{i}" for i in range(1, 7)]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    old = sys.argv
    sys.argv = [old[0]]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = old
    return m


E78 = load_module("l2_external", os.path.join(BASE, "scripts", "78e_l2_external_r3.py"))
TW = load_module("l2_threeway", os.path.join(BASE, "paper_brm", "external", "scripts", "l2_threeway.py"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ===================================================================================== data
def catalog():
    cat = json.load(open(os.path.join(RAW, "question_catalog.json"), encoding="utf-8"))
    by = {}
    for q in cat:
        for c in q.get("csv_columns", []):
            by[c] = q
    return by


MAPPING = json.load(open(os.path.join(LSR, "wave4_formatted_to_catalog_mapping.json"), encoding="utf-8"))
F2C = {m["formatted_column"]: m["catalog_csv_column"] for m in MAPPING}
CAT = catalog()


def valid_range(col):
    """Admissible codes of a catalog column: 1..k for single-choice and matrix items, 0..100 for
    sliders, none for numeric text entry."""
    q = CAT[col]
    t = q.get("QuestionType")
    if t == "Slider":
        return 0.0, 100.0
    if t == "MC":
        return 1.0, float(len(q["Options"]))
    if t == "Matrix":
        return 1.0, float(len(q["Columns"]))
    return -np.inf, np.inf


def human(which):
    if which == "w4":
        h = pd.read_csv(os.path.join(RAW, "question_catalog_and_human_response_csv", "wave4_response.csv"), low_memory=False)
    else:
        h = pd.read_csv(os.path.join(RAW, "wave1_3_response.csv"), low_memory=False)
    h = h.set_index("pid")
    return h[list(F2C.values())].apply(pd.to_numeric, errors="coerce")


def demographics():
    h = pd.read_csv(os.path.join(RAW, "wave1_3_response.csv"), low_memory=False).set_index("pid")
    d = pd.DataFrame(index=h.index)
    d["sex"] = h.QID12.map({1: "Male", 2: "Female"})
    d["age"] = h.QID13.map({1: "18-29", 2: "30-49", 3: "50-64", 4: "65+"})
    d["educ"] = h.QID14.map({1: "High school or less", 2: "High school or less", 3: "Some college or associate",
                             4: "Some college or associate", 5: "College graduate or more", 6: "College graduate or more"})
    d["race"] = h.QID15.map({1: "White", 2: "Black", 3: "Asian", 4: "Hispanic", 5: "Other"})
    d["party"] = h.QID20.map({1: "Republican", 2: "Democrat", 3: "Independent", 4: "Something else"})
    d["income"] = h.QID21.map({1: "Less than $30,000", 2: "$30,000-$50,000", 3: "$50,000-$75,000",
                               4: "$75,000-$100,000", 5: "$100,000 or more"})
    return d


def read_formatted(path):
    f = pd.read_csv(path, low_memory=False, dtype=str)
    assert f.iloc[0]["TWIN_ID"] == "TWIN_ID", path       # Qualtrics label row
    f = f.iloc[1:]
    f.index = f.TWIN_ID.astype(int)
    assert f.index.is_unique, path
    return f


def twin_answers(path, spec, parse_rows):
    """Twin answers in catalog columns; non-numeric and out-of-range entries become missing and are
    counted in parse_rows."""
    f = read_formatted(path)
    out = {}
    for fc, cc in F2C.items():
        raw = f[fc]
        x = pd.to_numeric(raw, errors="coerce")
        nonnum = int((raw.notna() & (raw.str.strip() != "") & x.isna()).sum())
        lo, hi = valid_range(cc)
        oor = x.notna() & ((x < lo) | (x > hi))
        parse_rows.append(dict(run=spec, column=cc, answered=int(x.notna().sum()), non_numeric=nonnum,
                               out_of_range=int(oor.sum())))
        out[cc] = x.where(~oor)
    return pd.DataFrame(out, index=f.index)


def outcomes_frame(X):
    """Outcome values per respondent from catalog-coded answers (humans and twins alike)."""
    o = X[OUTCOME_ITEMS].copy()
    p = X[PRICING]
    o["pricing_share_purchase"] = (p == 1).sum(axis=1).where(p.notna().any(axis=1)) / p.notna().sum(axis=1)
    for name, cols in (("pm_card_maximiser", PM_CARD), ("pm_dice_maximiser", PM_DICE)):
        q = X[cols]
        full = q.notna().all(axis=1)
        o[name] = (q == 1).all(axis=1).astype(float).where(full)
    return o


def outcome_list():
    items, excl = [], []
    for fc, cc in F2C.items():
        q = CAT[cc]
        if cc in PRICING:
            excl.append(dict(kind="column", formatted_column=fc, catalog_column=cc, status="pooled",
                             reason="single pricing item at a randomly drawn price; pooled into pricing_share_purchase"))
        elif cc in PM_CARD + PM_DICE:
            excl.append(dict(kind="column", formatted_column=fc, catalog_column=cc, status="pooled",
                             reason="one trial of a probability-matching problem; pooled into the authors' maximiser indicator"))
        else:
            items.append(cc)
            excl.append(dict(kind="column", formatted_column=fc, catalog_column=cc, status="outcome",
                             reason=f"{q.get('QuestionType')} {(q.get('Settings') or {}).get('Selector')}: single numeric or ordinal item"))
    return items, excl


OUTCOME_ITEMS, EXCL_COLS = outcome_list()
OUTCOMES = OUTCOME_ITEMS + ["pricing_share_purchase", "pm_card_maximiser", "pm_dice_maximiser"]


def outcome_label(o):
    if o == "pricing_share_purchase":
        return "Pricing: share of 40 products purchased"
    if o.startswith("pm_"):
        return f"Probability matching ({o.split('_')[1]} problem): maximiser"
    q = CAT[o]
    lab = q.get("BlockName", "")
    sub = o.split("_")
    if q.get("Rows") and len(sub) > 1 and q.get("RowsID"):
        lab += ": " + q["Rows"][q["RowsID"].index(sub[1])][:60]
    elif q.get("Statements") and len(sub) > 1 and q.get("StatementsID"):
        lab += ": " + q["Statements"][q["StatementsID"].index(sub[1])][:60]
    return lab.replace("‘", "'").replace("’", "'")


# =================================================================================== receipts
def receipts_data(rec, h4, h13, files):
    fails = 0
    for spec, d in files.items():
        if spec == CEILING:
            continue
        for which, H, fn in (("wave 4", h4, "responses_wave4_formatted.csv"), ("waves 1-3", h13, "responses_wave1_3_formatted.csv")):
            f = read_formatted(os.path.join(d, fn))
            bad = 0
            for fc, cc in F2C.items():
                x = pd.to_numeric(f[fc], errors="coerce")
                y = H.loc[x.index, cc]
                bad += int(((x != y) & ~(x.isna() & y.isna())).sum())
            ok = bad == 0
            fails += not ok
            rec.append(dict(check=f"{spec}: formatted {which} file equals the catalog CSV on TWIN_ID = pid",
                            got=f"{len(f)} respondents, {bad} cells differ", want="0 cells differ", match=ok, hard=True))
    a = sha(os.path.join(LSR, "GPT4.1-mini-simulation-llm-vs-human", "responses_llm_imputed_formatted.csv"))
    b = sha(os.path.join(SPECD, "Text Persona - GPT4.1-mini", "csv_comparison", "csv_formatted", "responses_llm_imputed_formatted.csv"))
    rec.append(dict(check="default-simulation twin file is byte-identical to 'Text Persona - GPT4.1-mini'",
                    got=a[:16], want=b[:16], match=a == b, hard=True))
    fails += a != b
    return fails


def _rnd(x, want):
    dec = len(want.split(".")[1]) if "." in want else 0
    return f"{x:.{dec}f}"


def receipts_paper(rec, h4, h13, twin):
    """Means and shares the paper prints (arXiv 2505.17479v1, Appendix A.6-A.7 and Section 5.2)."""
    def m(H, c, sub=0.0):
        return H[c].mean() - sub

    def share(H, c, codes):
        x = H[c].dropna()
        return x.isin(codes).mean()

    def pmax(H, cols):
        q = H[cols]
        q = q[q.notna().all(axis=1)]
        return (q == 1).all(axis=1).mean()

    checks = []
    for lab, H in (("wave 4", h4), ("waves 1-3", h13)):
        w4 = lab == "wave 4"
        checks += [
            (f"{lab} base rate, 30 engineers (QID154) mean", m(H, "QID154"), "52.39" if w4 else "52.17"),
            (f"{lab} base rate, 70 engineers (QID156) mean", m(H, "QID156"), "70.71" if w4 else "68.01"),
            (f"{lab} outcome bias, success (QID161) mean on -3..3", m(H, "QID161", 4), "1.64" if w4 else "1.66"),
            (f"{lab} outcome bias, failure (QID162) mean on -3..3", m(H, "QID162", 4), "1.04" if w4 else "0.88"),
            (f"{lab} sunk cost, no (QID181) mean", m(H, "QID181_TEXT"), "14.46" if w4 else "14.88"),
            (f"{lab} sunk cost, yes (QID182) mean", m(H, "QID182_TEXT"), "11.01" if w4 else "10.64"),
            (f"{lab} Allais form 1 share A (QID192 = 1)", 100 * share(H, "QID192", [1]), "63.6" if w4 else "69.2"),
            (f"{lab} Allais form 2 share D (QID193 = 2)", 100 * share(H, "QID193", [2]), "62.3" if w4 else "57.2"),
            (f"{lab} framing, gain (QID157) mean", m(H, "QID157"), "2.83" if w4 else "2.85"),
            (f"{lab} framing, loss (QID158) mean", m(H, "QID158"), "3.76" if w4 else "3.84"),
            (f"{lab} myside, German car (QID195) mean", m(H, "QID195"), "4.54" if w4 else "4.46"),
            (f"{lab} myside, Ford (QID194) mean", m(H, "QID194"), "4.10" if w4 else "4.11"),
            (f"{lab} omission bias, share avoiding (QID291 in 1, 2)", share(H, "QID291", [1, 2]), "0.45"),
            (f"{lab} probability matching, card, P_MAX", pmax(H, PM_CARD), "0.36"),
            (f"{lab} probability matching, dice, P_MAX", pmax(H, PM_DICE), "0.29" if w4 else "0.30"),
            (f"{lab} denominator neglect, share large tray (QID196 = 2)", share(H, "QID196", [2]), "0.38" if w4 else "0.36"),
        ]
    checks += [
        ("wave 4 less is more, gamble A (QID171) mean", m(h4, "QID171"), "2.15"),
        ("wave 4 less is more, gamble B (QID172) mean", m(h4, "QID172"), "3.15"),
        ("wave 4 less is more, gamble C (QID173) mean", m(h4, "QID173"), "3.01"),
        ("wave 4 Thaler WTA-certainty (QID190) mean", m(h4, "QID190"), "7.24"),
        ("wave 4 Thaler WTP-certainty (QID189) mean", m(h4, "QID189"), "3.23"),
        ("waves 1-3 Thaler WTP-certainty (QID189) mean", m(h13, "QID189"), "3.27"),
        ("waves 1-3 Thaler WTP-noncertainty (QID191) mean", m(h13, "QID191"), "2.20"),
    ]
    af = pd.concat([twin["QID164_TEXT"].dropna(), twin["QID166_TEXT"].dropna()])
    checks += [
        ("default twins: share answering 54 African UN countries (both anchors)", 100 * (af == 54).mean(), "98.8"),
        ("default twins: share choosing the lower-risk option, Allais forms 1 and 2", np.mean(np.r_[twin.QID192.dropna() == 1, twin.QID193.dropna() == 1]), "1.000"),
        ("default twins: share refusing the vaccine (QID291 in 1, 2), percent", 100 * share(twin, "QID291", [1, 2]), "4.0"),
        ("default twins: share opposing increased deportations (QID287_11 in 1, 2), percent", 100 * share(twin, "QID287_11", [1, 2]), "74.1"),
        ("wave 4 humans: share supporting increased deportations (QID287_11 in 4, 5), percent, 'about 45%'", 100 * share(h4, "QID287_11", [4, 5]), "45"),
    ]
    for chk, got, want in checks:
        g = _rnd(got, want)
        rec.append(dict(check=chk, got=g, want=want, match=g == want, hard=False))


def receipts_authors(rec, specs_files, h4, h13):
    """The authors' mad_accuracy_summary.xlsx, column level (every specification that has it) and
    task level (default)."""
    w4o = pd.DataFrame({c: h4[c] for c in F2C.values()})
    for spec, d in specs_files.items():
        x = os.path.join(d, "..", "..", "accuracy_evaluation", "mad_accuracy_summary.xlsx")
        if spec == CEILING or not os.path.exists(x):
            continue
        sheet = pd.read_excel(x, sheet_name="Accuracy & Corr - column level")
        f = read_formatted(os.path.join(d, "responses_llm_imputed_formatted.csv"))
        L = pd.DataFrame({cc: pd.to_numeric(f[fc], errors="coerce") for fc, cc in F2C.items()}, index=f.index)
        up = {fc.upper(): cc for fc, cc in F2C.items()}
        n_c = n_m = n_c_ok = n_m_ok = 0
        free = []
        for r in sheet.itertuples(index=False):
            cc = up.get(str(r[0]).upper())
            if cc is None:
                continue
            a, b4, l = h13.loc[L.index, cc], w4o.loc[L.index, cc], L[cc]
            ok = a.notna() & l.notna()
            if cc.endswith("_TEXT") and cc.startswith(("QID164", "QID166", "QID168", "QID170")):
                free.append(cc)
                continue
            lo = np.nanmin(np.r_[a, b4, l]); hi = np.nanmax(np.r_[a, h4.loc[L.index, cc], l])
            mad = float((l[ok] - a[ok]).abs().mean() / (hi - lo))
            want_mad = float(r[5])
            n_m += 1
            n_m_ok += abs(mad - want_mad) < 5e-6
            if np.isfinite(r[17]) and l[ok].std() > 0:
                n_c += 1
                n_c_ok += abs(np.corrcoef(l[ok], a[ok])[0, 1] - r[17]) < 5e-6
        rec.append(dict(check=f"{spec}: authors' column-level MAD (twin vs waves 1-3), recomputed",
                        got=f"{n_m_ok} of {n_m} columns within 5e-6", want=f"{n_m} of {n_m}",
                        match=n_m_ok == n_m, hard=False))
        rec.append(dict(check=f"{spec}: authors' column-level Pearson r (twin vs waves 1-3), recomputed",
                        got=f"{n_c_ok} of {n_c} columns within 5e-6", want=f"{n_c} of {n_c}",
                        match=n_c_ok == n_c, hard=False))
        if free:
            rec.append(dict(check=f"{spec}: four free-text anchoring estimates ({', '.join(free)}) not recomputed",
                            got="authors' free-text normalisation is not in the release", want="", match=None, hard=False))
        if spec == HEADLINE:
            # W4 vs W13 test-retest correlation, aligned on TWIN_ID
            n_r = n_r_ok = 0
            for r in sheet.itertuples(index=False):
                cc = up.get(str(r[0]).upper())
                if cc is None or (cc.endswith("_TEXT") and cc.startswith(("QID164", "QID166", "QID168", "QID170"))):
                    continue
                a, b4 = h13.loc[L.index, cc], w4o.loc[L.index, cc]
                ok = a.notna() & b4.notna()
                if np.isfinite(r[16]):
                    n_r += 1
                    n_r_ok += abs(np.corrcoef(b4[ok], a[ok])[0, 1] - r[16]) < 5e-6
            rec.append(dict(check="authors' column-level test-retest r (wave 4 vs waves 1-3), recomputed on TWIN_ID",
                            got=f"{n_r_ok} of {n_r} columns within 5e-6", want=f"{n_r} of {n_r}",
                            match=n_r_ok == n_r, hard=False))
            task_level(rec, sheet_path=x, L=L, h4=h4, h13=h13)


TASK = {"QID287": "false consensus", "QID290": "false consensus", "QID156": "base rate", "QID154": "base rate",
        "QID158": "framing problem", "QID157": "framing problem", "QID160": "conjunction problem (Linda)",
        "QID159": "conjunction problem (Linda)", "QID162": "outcome bias", "QID161": "outcome bias",
        "QID181": "sunk cost fallacy", "QID182": "sunk cost fallacy", "QID183": "absolute vs. relative savings",
        "QID184": "absolute vs. relative savings", "QID189": "WTA/WTP-Thaler", "QID190": "WTA/WTP-Thaler",
        "QID191": "WTA/WTP-Thaler", "QID192": "Allais", "QID193": "Allais", "QID194": "myside", "QID195": "myside",
        "QID198": "prob matching vs. max", "QID203": "prob matching vs. max", "QID288": "non-separability of risks and benefits",
        "QID289": "non-separability of risks and benefits", "QID291": "omission", "QID196": "denominator neglect",
        "QID9": "pricing", **{f"QID{k}": "anchoring and adjustment" for k in range(163, 171)},
        **{f"QID{k}": "less is more" for k in range(171, 180)}}


def task_level(rec, sheet_path, L, h4, h13):
    """Per-respondent task accuracy = mean over the task's columns of 1 - |twin - waves 1-3| / range,
    averaged over respondents, then over the 17 tasks (paper Section 5.1)."""
    want = pd.read_excel(sheet_path, sheet_name="Accuracy & Corr - task level").set_index("Task")
    cols = list(F2C.values())
    out = {}
    for name, A in (("twin", L), ("wave 4", h4.loc[L.index])):
        acc = {}
        for cc in cols:
            a, b = A[cc], h13.loc[L.index, cc]
            lo = np.nanmin(np.r_[h13.loc[L.index, cc], h4.loc[L.index, cc], L[cc]])
            hi = np.nanmax(np.r_[h13.loc[L.index, cc], h4.loc[L.index, cc], L[cc]])
            acc[cc] = 1 - (a - b).abs() / (hi - lo)
        acc = pd.DataFrame(acc)
        out[name] = pd.Series({t: acc[[c for c in cols if TASK[c.split("_")[0]] == t]].mean(axis=1).mean()
                               for t in sorted(set(TASK.values()))})
    wcol = {"twin": "llm vs. wave1_3 Accuracy", "wave 4": "wave4 vs. wave1_3 Accuracy"}
    for name in out:
        g = out[name]
        w = want[wcol[name]]
        same = [t for t in g.index if f"{g[t]:.3f}" == f"{w[t]:.3f}"]
        diff = [f"{t} {g[t]:.3f} vs {w[t]:.3f}" for t in g.index if t not in same]
        rec.append(dict(check=f"authors' task-level accuracy, {name} vs waves 1-3, default specification",
                        got=f"{len(same)} of 17 tasks equal to 3 decimals; differ: {'; '.join(diff)}",
                        want="17 of 17", match=len(same) == 17, hard=False))
        mixed = np.mean([g[t] if t in same else w[t] for t in g.index])
        paper = "71.72" if name == "twin" else "81.72"
        rec.append(dict(check=f"mean over 17 tasks ({name}), recomputed tasks + the authors' values for the tasks that differ",
                        got=f"{100 * mixed:.2f}", want=paper, match=f"{100 * mixed:.2f}" == paper, hard=False))
        rec.append(dict(check=f"mean over 17 tasks ({name}), all recomputed (anchoring free text on observed range)",
                        got=f"{100 * g.mean():.2f}", want=paper, match=None, hard=False))


# ===================================================================================== rungs
def l2_raw(S, Y, demo, run, ref, skips):
    rows = []
    for oi, o in enumerate(OUTCOMES):
        d = pd.DataFrame(dict(s=S[o], y=Y[o])).join(demo, how="left").dropna(subset=["s", "y"])
        for ci, (attr, a, b) in enumerate(CONTRASTS):
            A, Bb = d[d[attr] == a], d[d[attr] == b]
            if len(A) < MIN_N or len(Bb) < MIN_N:
                skips.append(dict(kind="contrast", run=run, reference=ref, outcome=o, contrast=f"{a} - {b}",
                                  status="not formed", reason=f"group below {MIN_N} pairs (n_a={len(A)}, n_b={len(Bb)})"))
                continue
            rng = np.random.default_rng([SEED, oi, ci])
            ia = rng.integers(0, len(A), (B_L2, len(A)))
            ib = rng.integers(0, len(Bb), (B_L2, len(Bb)))
            sa, ya, sb, yb = A.s.values, A.y.values, Bb.s.values, Bb.y.values
            bg = sa[ia].mean(1) - sb[ib].mean(1)
            bp = ya[ia].mean(1) - yb[ib].mean(1)
            rows.append(dict(run=run, reference=ref, outcome=o, contrast=f"{a} - {b}", attribute=attr,
                             n_a=len(A), n_b=len(Bb), g=sa.mean() - sb.mean(), se_g=bg.std(ddof=1),
                             p=ya.mean() - yb.mean(), se_p=bp.std(ddof=1), cov=np.cov(bg, bp, ddof=1)[0, 1]))
    return pd.DataFrame(rows)


def l2_verdicts(raw, config, family):
    d = raw.copy()
    d["config"] = config
    d["m_family"] = d.groupby(family).g.transform("size")
    with np.errstate(divide="ignore", invalid="ignore"):
        res = [E78.r3cov(r.g, r.se_g, r.p, r.se_p, r.cov, np.inf, E78.ALPHA / r.m_family, E78.BANDS["asym"])
               for r in d.itertuples()]
    d = pd.concat([d.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    d["label"] = [TW.three_way(v, lo, hi) for v, lo, hi in zip(d.verdict, d.ci_lo.astype(float), d.ci_hi.astype(float))]
    return d


def l2_model(l2):
    rows = []
    for (cfg, run), g in l2.groupby(["config", "run"]):
        stopped = g.verdict.isin(["no population gap", "reference too imprecise"])
        live = g[~stopped].verdict
        excl = live.map(lambda v: "kept" not in v.split(" or ") and v != "undetermined")
        verdict = "pass" if (live == "kept").all() and len(live) else ("fail" if excl.any() else "unresolved")
        rows.append(dict(config=cfg, run=run, contrasts=len(g), stopped=int(stopped.sum()),
                         kept=int((live == "kept").sum()), excludes_kept=int(excl.sum()), verdict=verdict))
    return pd.DataFrame(rows)


def level3_rows(S, Y, Yall, demo, run, ref):
    groups = [(None, None)] + sorted({(a, v) for a, x, y in CONTRASTS for v in (x, y)})
    rows = []
    for o in OUTCOMES:
        sd_ref = float(Yall[o].std(ddof=1))
        d = pd.DataFrame(dict(s=S[o], y=Y[o])).join(demo, how="left").dropna(subset=["s", "y"])
        for attr, val in groups:
            sub = d if attr is None else d[d[attr] == val]
            if len(sub) < 2:
                continue
            res = sub.s - sub.y
            r, se = res.mean(), res.std(ddof=1) / np.sqrt(len(sub))
            lo, hi = r - Z90 * se, r + Z90 * se
            rows.append(dict(run=run, reference=ref, outcome=o, group="all" if attr is None else f"{attr}:{val}",
                             n=len(sub), human_mean=sub.y.mean(), sim_mean=sub.s.mean(), residual=r, se=se,
                             ci90_lo=lo, ci90_hi=hi, sd_ref=sd_ref, residual_sd=r / sd_ref,
                             pass_05sd=bool(lo > -.5 * sd_ref and hi < .5 * sd_ref),
                             pass_02sd=bool(lo > -.2 * sd_ref and hi < .2 * sd_ref),
                             fail_05sd=bool(lo > .5 * sd_ref or hi < -.5 * sd_ref)))
    return pd.DataFrame(rows)


def gate_diagnostic(t0, t7, Yall):
    rows = []
    for o in OUTCOMES:
        d = pd.DataFrame(dict(a=t0[o], b=t7[o])).dropna()
        sd_ref = float(Yall[o].std(ddof=1))
        diff = d.a - d.b
        rows.append(dict(outcome=o, n=len(d), identical_share=float((diff == 0).mean()),
                         sd_ref=sd_ref, pseudo_within_sd=float(diff.std(ddof=1) / np.sqrt(2)),
                         pseudo_within_sd_in_ref_sd=float(diff.std(ddof=1) / np.sqrt(2) / sd_ref)))
    return pd.DataFrame(rows)


# ===================================================================================== main
def main():
    rec, parse_rows, skips = [], [], []
    h4, h13 = human("w4"), human("w13")
    demo = demographics()
    files = {k: os.path.join(SPECD, folder, "csv_comparison", "csv_formatted") for k, folder, *_ in SPECS}
    fails = receipts_data(rec, h4, h13, files)
    twins = {k: twin_answers(os.path.join(files[k], "responses_llm_imputed_formatted.csv"), k, parse_rows) for k in files}
    receipts_paper(rec, h4, h13, twins[HEADLINE])
    receipts_authors(rec, files, h4, h13)
    r = pd.DataFrame(rec)
    r.to_csv(os.path.join(OUT, "receipts.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 140, "display.max_rows", 500)
    print(r[["check", "got", "want", "match"]].to_string(index=False))
    if fails:
        sys.exit("data-integrity receipt mismatch")
    pd.DataFrame(parse_rows).to_csv(os.path.join(OUT, "parse_receipts.csv"), index=False)

    O4, O13 = outcomes_frame(h4), outcomes_frame(h13)
    sims = {k: outcomes_frame(v) for k, v in twins.items()}
    sims[CEILING] = O13
    runs = [k for k, *_ in SPECS] + [CEILING]

    raw = []
    for k in runs:
        S = sims[k]
        Y4 = O4.loc[S.index]
        raw.append(l2_raw(S, Y4, demo, k, "wave4", skips))
        if k != CEILING:
            raw.append(l2_raw(S, O13.loc[S.index], demo, k, "wave1_3", skips))
        print("level 2 raw", k, flush=True)
    raw = pd.concat(raw, ignore_index=True)
    w4raw = raw[raw.reference == "wave4"]
    l2 = pd.concat([l2_verdicts(w4raw, "headline", "run"),
                    l2_verdicts(raw[raw.reference == "wave1_3"], "ref_wave1_3", "run"),
                    l2_verdicts(w4raw, "family_outcome", ["run", "outcome"])], ignore_index=True)
    l2["outcome_label"] = l2.outcome.map(outcome_label)
    l2.to_csv(os.path.join(OUT, "level2_contrasts.csv"), index=False)
    l2m = l2_model(l2)
    l2m.to_csv(os.path.join(OUT, "level2_model.csv"), index=False)

    # rows in the Figure 3 format (l2_threeway_rows.csv)
    h = l2[l2.config == "headline"]
    fig = pd.DataFrame(dict(dataset="Twin-2K", run=h.run, framing="", family=h.run, contrast=h.contrast,
                            ratio=h.ratio, ci_lo=h.ci_lo, ci_hi=h.ci_hi, verdict=h.verdict, outcome=h.outcome,
                            label=h.label))
    fig.to_csv(os.path.join(OUT, "l2_threeway_rows_twin2k.csv"), index=False)
    # contrasts whose reference can certify "kept" at the family's level (neither stop fires on the
    # reference alone: gamma separable from zero and k <= 1/4); the set depends on the reference only
    h = h.assign(certifiable=~h.no_population_gap.astype(bool) & ~h.stop_population.astype(bool))
    cset = h[(h.run == CEILING) & h.certifiable][["outcome", "contrast"]]
    cert = h.merge(cset, on=["outcome", "contrast"])
    cert[["run", "outcome", "contrast", "n_a", "n_b", "g", "p", "se_p", "k_rel_halfwidth", "m_family", "ratio",
          "ci_lo", "ci_hi", "verdict", "label"]].assign(outcome_label=cert.outcome.map(outcome_label)).to_csv(
        os.path.join(OUT, "level2_certifiable.csv"), index=False)

    l3 = []
    for k in runs:
        S = sims[k]
        l3.append(level3_rows(S, O4.loc[S.index], O4, demo, k, "wave4"))
        if k != CEILING:
            l3.append(level3_rows(S, O13.loc[S.index], O13, demo, k, "wave1_3"))
    l3 = pd.concat(l3, ignore_index=True)
    l3.to_csv(os.path.join(OUT, "level3_groups.csv"), index=False)
    l3o = (l3.groupby(["run", "reference", "outcome"])
           .agg(groups=("group", "size"), pass_05sd=("pass_05sd", "all"), pass_02sd=("pass_02sd", "all"),
                groups_failing_05sd=("fail_05sd", "sum"), worst_abs_residual_sd=("residual_sd", lambda v: v.abs().max()))
           .reset_index())
    l3o["any_group_fails_05sd"] = l3o.groups_failing_05sd > 0
    l3o.to_csv(os.path.join(OUT, "level3_outcome.csv"), index=False)

    gd = gate_diagnostic(sims["text_gpt41mini"], sims["text_t07_gpt41mini"], O4)
    gd.to_csv(os.path.join(OUT, "gate_diagnostic_temperature.csv"), index=False)

    # summary per run
    meta = {k: dict(model=m, persona=p, temperature=t, note=n) for k, _, m, p, t, n in SPECS}
    meta[CEILING] = dict(model="none (human)", persona="same respondent's waves 1-3 answers", temperature="",
                         note="test-retest ceiling")
    rows = []
    for k in runs:
        row = dict(run=k, **meta[k], respondents=len(sims[k]), paper_accuracy=PAPER_ACC.get(k))
        for cfg in ("headline", "ref_wave1_3", "family_outcome"):
            g = l2[(l2.config == cfg) & (l2.run == k)]
            if not len(g):
                continue
            pre = "" if cfg == "headline" else cfg + "_"
            for lab in ("kept", "not kept", "unresolved", "not read"):
                row[pre + lab.replace(" ", "_")] = int((g.label == lab).sum())
            row[pre + "contrasts"] = len(g)
            if cfg == "headline":
                row["m_bonferroni"] = int(g.m_family.iloc[0])
                live = g[g.label != "not read"]
                row["point_ratio_in_kept"] = int((live.ratio.between(0.75, 1.25)).sum())
                row["median_ratio_read"] = float(live.ratio.median())
                row["l2_model_verdict"] = l2m[(l2m.config == cfg) & (l2m.run == k)].verdict.iloc[0]
                hc = h[(h.run == k) & h.certifiable]
                row["certifiable_own"] = len(hc)
                cc = cert[cert.run == k]
                row["certifiable_common"] = len(cc)
                for lab in ("kept", "not kept", "unresolved", "not read"):
                    row["cert_" + lab.replace(" ", "_")] = int((cc.label == lab).sum())
                row["cert_ratio_min"] = float(cc.ratio.min()) if len(cc) else None
                row["cert_ratio_max"] = float(cc.ratio.max()) if len(cc) else None
                for v, n in g.verdict.value_counts().items():
                    row["v_" + v.replace(" ", "_")] = int(n)
        for ref in ("wave4", "wave1_3"):
            o = l3o[(l3o.run == k) & (l3o.reference == ref)]
            if not len(o):
                continue
            pre = "l3_" if ref == "wave4" else "l3_ref_wave1_3_"
            row[pre + "outcomes"] = len(o)
            row[pre + "pass_05sd"] = int(o.pass_05sd.sum())
            row[pre + "pass_02sd"] = int(o.pass_02sd.sum())
            row[pre + "outcomes_with_failing_group"] = int(o.any_group_fails_05sd.sum())
            a = l3[(l3.run == k) & (l3.reference == ref)]
            row[pre + "group_rows_pass_05sd"] = f"{int(a.pass_05sd.sum())}/{len(a)}"
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    byc = (h.groupby(["run", "contrast", "label"]).size().unstack(fill_value=0).reset_index())
    med = h[h.label != "not read"].groupby(["run", "contrast"]).ratio.median().rename("median_ratio_read").reset_index()
    byc.merge(med, on=["run", "contrast"], how="left").to_csv(os.path.join(OUT, "level2_by_contrast.csv"), index=False)
    byo = (h.groupby(["run", "outcome", "label"]).size().unstack(fill_value=0).reset_index())
    byo["outcome_label"] = byo.outcome.map(outcome_label)
    byo.to_csv(os.path.join(OUT, "level2_by_outcome.csv"), index=False)

    rungs = pd.DataFrame([
        dict(rung="gate", status="not applicable", reason="one answer per twin per specification (temperature 0, one run at 0.7); no repeated draws under one prompt"),
        dict(rung="level 1", status="not applicable", reason="wave 4 holds no multi-item scale of one construct: single items, between-subject conditions, 40 pricing decisions at random prices, 10 separate policy opinions"),
        dict(rung="level 2", status="run", reason="persona attributes = waves 1-3 demographics; reference = same respondents' wave-4 answers"),
        dict(rung="level 3", status="run", reason="same respondents; tolerance 0.5 reference SD (default) and 0.2 SD"),
        dict(rung="level 4", status="not applicable", reason="as level 1"),
    ])
    rungs.to_csv(os.path.join(OUT, "rungs.csv"), index=False)

    ex = pd.DataFrame(EXCL_COLS)
    f0 = read_formatted(os.path.join(files[HEADLINE], "responses_llm_imputed_formatted.csv"))
    unm = [c for c in f0.columns if c not in F2C]
    ex = pd.concat([ex, pd.DataFrame(dict(kind="column", formatted_column=unm, catalog_column="", status="excluded",
                                          reason="not in the authors' wave-4 mapping: survey metadata, attention or "
                                                 "logistics questions, or randomisation-order variables (mapping README)")),
                    pd.DataFrame(skips)], ignore_index=True)
    ex.to_csv(os.path.join(OUT, "exclusions.csv"), index=False)
    pd.DataFrame(dict(outcome=OUTCOMES, label=[outcome_label(o) for o in OUTCOMES],
                      n_wave4=[int(O4[o].notna().sum()) for o in OUTCOMES],
                      sd_wave4=[float(O4[o].std()) for o in OUTCOMES])).to_csv(os.path.join(OUT, "outcomes.csv"), index=False)

    cols = ["run", "kept", "not_kept", "unresolved", "not_read", "contrasts", "point_ratio_in_kept", "median_ratio_read",
            "l2_model_verdict", "certifiable_own", "cert_kept", "cert_not_kept", "cert_unresolved", "cert_not_read",
            "cert_ratio_min", "cert_ratio_max", "family_outcome_kept", "family_outcome_not_kept", "family_outcome_unresolved",
            "family_outcome_not_read", "ref_wave1_3_kept", "ref_wave1_3_not_kept", "l3_pass_05sd", "l3_pass_02sd",
            "l3_outcomes_with_failing_group", "paper_accuracy"]
    print(summ[[c for c in cols if c in summ]].to_string(index=False))
    print(gd.describe().round(3).to_string())


if __name__ == "__main__":
    main()
