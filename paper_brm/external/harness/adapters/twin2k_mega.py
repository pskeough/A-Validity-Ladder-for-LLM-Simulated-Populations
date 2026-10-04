"""Twin-2K-500 Mega Study (Peng, Gui, Brucks, Merlau, Fan, ... and Toubia, 2026; arXiv 2509.19088;
Hugging Face dataset LLM-Digital-Twin/Twin-2K-500-Mega-Study, Apache-2.0; code
github.com/TianyiPeng/Twin-2K-500-Mega-Study). Raw files: raw/twin2k_mega/ (SOURCE.txt holds URLs and
SHA-256).

Design. 19 survey studies, each answered by human respondents drawn from the Twin-2K-500 panel and,
for the same respondents (key TWIN_ID = PID of Twin-2K-500), by LLM digital twins under up to 23
twin specifications (results.zip: results/<study>/<specification>_<date>/consolidated_original_answers_
values.csv = the human answers, consolidated_llm_values.csv = the twin answers, same column order,
same two Qualtrics header rows). Demographic attributes come from the Twin-2K-500 profiles that
scripts/93 reads (raw/twin2k/wave1_3_response.csv: QID12 sex, QID13 age band, QID14 education, QID15
race or origin, QID20 party, QID21 family income), joined on PID.

Frozen before any gap was computed (3 Oct 2026):
  contrasts  the seven of scripts/93: Female - Male, Black - White, Hispanic - White, High school or
             less - College graduate or more, 18-29 - 65+, Republican - Democrat, income under
             $30,000 - $100,000 or more. The list is CONTRASTS below and was not changed.
  gaps       raw outcome-scale mean differences g (twin) and gamma (human) on the respondents who
             have both a human and a twin answer to the outcome; a contrast is formed when both groups
             hold at least MIN_N = 30 such pairs.
  se, cov    respondent bootstrap, B = 2000, one resample of persons per group that keeps each twin
             with its human, se = SD over resamples, cov = covariance of the twin and human gaps
             (as scripts/93), df infinite.
  family     one twin specification across all studies, outcomes and contrasts; family_size = number
             of contrasts in that family (Bonferroni, harness default).
  control    family "human_split_half": the human respondents of each study are split in half
             (seed 20261003, stratified by party); half A is the simulator, half B the reference;
             independent samples, cov = 0, each half bootstrapped separately; a contrast is formed when
             both groups hold at least MIN_N respondents in each half.
Outcomes (mechanical rule; applied to the QSF and the data, not to any gap):
  - the column maps to a QSF question (DataExportTag) of type Matrix/Likert (single answer), Slider,
    text entry that is numeric, or single-answer multiple choice with two options or ordinal option
    text (scale words or numeric bins; see ordinal_choices). Nominal multiple choice (product chosen out
    of six, quiz answers), multi-answer, ranking, constant-sum, free text, timing, browser, display-order,
    randomiser, embedded-data columns are excluded, with reasons in exclusions.csv.
  - numeric (at least 90% of the non-empty cells) in the human and the twin file, at least two
    distinct human values, and a label that is not a demographic or party item (those are the
    grouping variables, DEMOG below).
  - columns that repeat one item over a loop (<n>_<tag>, 10 or more iterations) are pooled to the
    respondent's mean over the iterations both the human and the twin answered (idea_evaluation).
  - experiments with between-subject conditions: no condition is modelled; every outcome column is
    read on the respondents who saw it (pooled over conditions); the share of outcomes seen by under
    90% of respondents is reported per study in inventory_studies.csv.
"""
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

NAME = "twin2k_mega"
KIND = "digital twins (LLM, 23 specifications) of the same respondents, 19 studies, and an ideal-simulator control"
SOURCE = ("Peng et al. (2026), Digital Twins as Funhouse Mirrors, arXiv 2509.19088; "
          "https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study; "
          "https://github.com/TianyiPeng/Twin-2K-500-Mega-Study")

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(EXT, "raw")
MEGA = os.path.join(RAW, "twin2k_mega")
RES = os.path.join(MEGA, "results_unzipped", "results")
DAT = os.path.join(MEGA, ".dat")
PROFILE = os.path.join(RAW, "twin2k", "wave1_3_response.csv")
OUT = os.path.join(EXT, "harness", "results", NAME)

SEED = 20261003
B_BOOT = 2000
MIN_N = 30
# Frozen contrast list: attribute, level a, level b (a - b); identical to scripts/93_twin2k_ladder.py
CONTRASTS = [("sex", "Female", "Male"), ("race", "Black", "White"), ("race", "Hispanic", "White"),
             ("educ", "High school or less", "College graduate or more"), ("age", "18-29", "65+"),
             ("party", "Republican", "Democrat"), ("income", "Less than $30,000", "$100,000 or more")]

STUDIES = ["accuracy_nudges", "affective_priming", "consumer_minimalism", "context_effects", "default_eric",
           "digital_certification", "hiring_algorithms", "idea_evaluation", "idea_generation", "infotainment",
           "junk_fees", "obedient_twins", "preference_redistribution", "privacy", "promiscuous_donors",
           "quantitative_intuition", "recommendation_algorithms", "story_beliefs", "targeting_fairness"]
# reference specification: fixes the outcome list of a study and, for the control, the human pool
REF_SPECS = ["full_persona_without_reasoning", "temperature_zero"]
H_FILE = "consolidated_original_answers_values.csv"
T_FILE = "consolidated_llm_values.csv"

# survey items that are the grouping variables themselves (demographics, party, ideology)
DEMOG = re.compile(
    r"(what is your (gender|sex|age|race|ethnic|education|household|annual|total|highest|marital|zip|state)"
    r"|your (gender|sex|age|race|ethnicity|education level|household income|annual income|political party|"
    r"party identification|marital status|employment status|zip)"
    r"|year (were you )?born|how old are you|political (views|ideology|affiliation|party|orientation)"
    r"|republican|democrat|liberal.{0,20}conservative|conservative.{0,20}liberal"
    r"|race or ethnic|ethnic background|which of the following (best )?describes your (race|gender|ethnic)"
    r"|do you identify as|gender identity|sexual orientation|highest level of (school|education)"
    r"|household income|total income|annual (household )?income)", re.I)

ATTN = re.compile(r"(please type .{0,12} here|to ensure you.re paying attention|attention check|as your answer to this question)", re.I)
NAME_DEMOG = re.compile(r"^(age|gender|sex|race|ethnicity|hispanic|education|income|hhi|party|political_party|region|zip|"
                        r"state|ideology|conservative|oonservative|lean|strong_rep|strong_dem)$", re.I)

# scale words that make a single-answer choice set ordinal
SCALE = re.compile(
    r"(agree|disagree|likely|unlikely|important|not at all|slightly|somewhat|moderate|very |extremely|strongly|"
    r"a little|a lot|a great deal|none at all|never|rarely|sometimes|often|always|frequent|good|poor|fair|"
    r"excellent|satisf|happy|unhappy|true|false|certain|confident|familiar|much|less|more|worse|better|"
    r"prefer|support|oppose|favor|concern|worried|comfortable|trust|positive|negative|quite|daily|weekly|monthly|"
    r"definitely|probably|possibly|completely|entirely|neutral|average|low|high|rate|appropriate|acceptable|"
    r"offensive|safe|risky|sure|difficult|easy|interest|bored|funny|serious|accurate|inaccurate|appeal|"
    r"fairly|extent|degree|\d+\s*(-|–|to|\+|%|hours?|times?|years?|min))", re.I)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_html(t):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(t))).strip()


# ============================================================================== demographics
def demographics():
    h = pd.read_csv(PROFILE, low_memory=False).set_index("pid")
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


# ============================================================================== specifications
def spec_key(dirname):
    k = re.sub(r"_?\d{4}-\d{2}-\d{2}.*$", "", dirname)
    return "full_persona_without_reasoning" if k == "full_persona_without_resoning" else k


def spec_dirs(study):
    """{specification key: directory} for the specifications that have both consolidated files; a
    duplicate key keeps the later-dated directory (digital_certification fine-tuned, temperature 0)."""
    out, notes = {}, []
    base = os.path.join(RES, study)
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if not os.path.isdir(p):
            continue
        if not (os.path.exists(os.path.join(p, H_FILE)) and os.path.exists(os.path.join(p, T_FILE))):
            notes.append(f"{d}: no consolidated values files")
            continue
        k = spec_key(d)
        if k in out:
            notes.append(f"{k}: {os.path.basename(out[k])} and {d} both present, kept {d} (later date)")
        out[k] = p
    return out, notes


def ref_spec(specs):
    for k in REF_SPECS:
        if k in specs:
            return k
    return sorted(specs)[0]


def read_values(path, cols=None):
    """Consolidated Qualtrics CSV -> DataFrame of strings indexed by integer TWIN_ID (the two header
    rows, the label row and the import-id row, are dropped; labels returned separately)."""
    f = pd.read_csv(path, dtype=str, low_memory=False,
                    usecols=None if cols is None else (lambda c: c == "TWIN_ID" or c in cols))
    labels = f.iloc[0]
    tid = pd.to_numeric(f["TWIN_ID"], errors="coerce")
    f = f[tid.notna()].copy()
    f.index = tid[tid.notna()].astype(int).values
    f = f.drop(columns=["TWIN_ID"])
    if f.index.duplicated().any():
        raise ValueError(f"duplicate TWIN_ID in {path}")
    return f, labels


def num(s):
    return pd.to_numeric(s, errors="coerce")


# ============================================================================== QSF
def qsf_questions(study):
    q = json.load(open(os.path.join(DAT, study, "raw_data", "survey.qsf"), encoding="utf-8"))
    out = {}
    for e in q["SurveyElements"]:
        if e.get("Element") != "SQ":
            continue
        p = e["Payload"]

        def texts(x):
            if isinstance(x, dict):
                return [strip_html(v.get("Display", "")) for v in x.values() if isinstance(v, dict)]
            if isinstance(x, list):
                return [strip_html(v.get("Display", "")) if isinstance(v, dict) else strip_html(v) for v in x]
            return []
        out[p["DataExportTag"]] = dict(qtype=p.get("QuestionType"), selector=p.get("Selector"),
                                       sub=p.get("SubSelector"), choices=texts(p.get("Choices")),
                                       answers=texts(p.get("Answers")))
    return out


def match_tag(col, tags):
    """QSF DataExportTag of a data column and the loop prefix, or (None, None)."""
    c = re.sub(r"\.\d+$", "", col)
    base = [(c, None)]
    m = re.match(r"^(\d+)_(.+)$", c)
    if m:
        base.append((m.group(2), m.group(1)))
    for b, loop in list(base):
        parts = b.split("_")
        for k in range(len(parts) - 1, 0, -1):
            base.append(("_".join(parts[:k]), loop))
    for b, loop in base:
        if b in tags:
            return b, loop
    return None, None


NOT_ITEM = re.compile(r"(^|_)DO(_|$)|^FL_\d+|_TEXT$|First Click|Last Click|Page Submit|Click Count|(^|_)RT_|"
                      r"_(Browser|Version|Operating System|Resolution)$|^Q_|^(Status|ResponseId|Recipient|External|"
                      r"DistributionChannel|UserLanguage|Location|IPAddress|Start|End|Recorded|Duration|Finished|Progress)")


NONORD = re.compile(r"(don.?t know|do not know|not sure|unsure|prefer not|not applicable|n/a|none of the|other|"
                    r"no opinion|don.?t use|refuse|can.?t remember|depends)", re.I)


def ordinal_choices(texts):
    """(ok, why): a single-answer choice set that can be read as an ordinal rating or a yes/no.
    Rejected: placeholder text, a don't-know / other / not-sure / not-applicable option (its code would
    sit on the scale), more than 12 options, nominal options. Accepted: two options; a numbered scale
    (leading integers on at least 70% of the options, consecutive from 0 or 1); scale words on at least
    60% of the options."""
    t = [x for x in texts if x]
    if not t:
        return False, "no option text"
    if all(re.match(r"^Click to write", x) for x in t):
        return False, "placeholder options"
    if any(NONORD.search(x) for x in t):
        return False, "has a don't-know, other, not-sure or not-applicable option"
    if len(t) == 2:
        return True, "two options"
    if len(t) > 12:
        return False, f"{len(t)} options, nominal or lookup"
    ints = []
    for x in t:
        m = re.match(r"^\s*(\d+)", x) or re.search(r"(\d+)\s*$", x)
        ints.append(int(m.group(1)) if m else None)
    got = [i for i in ints if i is not None]
    if len(got) >= 0.7 * len(t) and len(got) >= 3 and got[0] in (0, 1) and got == list(range(got[0], got[0] + len(got))):
        return True, "numbered scale"
    hit = sum(bool(SCALE.search(x)) for x in t)
    if hit / len(t) >= 0.6:
        return True, "ordinal option text"
    return False, "nominal options"


def classify_column(col, tags):
    """(include, qtype, why) from the QSF alone."""
    if NOT_ITEM.search(col):
        return False, "", "survey metadata, timing, display-order, randomiser or other-text column"
    tag, loop = match_tag(col, tags)
    if tag is None:
        return False, "", "no QSF question (embedded data or derived column)"
    q = tags[tag]
    t, sel = q["qtype"], q["selector"]
    if t == "Matrix" and sel in ("Likert", "Bipolar") and q["sub"] in ("SingleAnswer", None):
        return True, "Matrix/Likert", "rating scale"
    if t == "Slider":
        return True, "Slider", "slider"
    if t == "TE":
        return True, "TextEntry", "text entry, kept only if numeric"
    if t == "MC" and sel in ("SAVR", "SAHR", "SACOL", "DL", "SAHR"):
        ok, why = ordinal_choices(q["choices"])
        return ok, "MC single answer", why
    return False, str(t), f"question type {t}/{sel} not read as a rating"


# ============================================================================== outcomes
def numeric_share(s):
    nn = s.notna() & (s.astype(str).str.strip() != "")
    if nn.sum() == 0:
        return 0.0, 0
    x = num(s)
    return float(x.notna().sum() / nn.sum()), int(x.notna().sum())


def build_outcomes(study, specs):
    """Outcome list of a study from the reference specification: list of dict(name, cols, pooled,
    label, qtype, levels) and the exclusion rows."""
    rs = ref_spec(specs)
    p = specs[rs]
    hs, labels = read_values(os.path.join(p, H_FILE))
    ts, _ = read_values(os.path.join(p, T_FILE))
    tags = qsf_questions(study)
    excl, cand = [], []
    for col in hs.columns:
        inc, qtype, why = classify_column(col, tags)
        lab = strip_html(labels.get(col, ""))[:120]
        if (not inc) and why == "placeholder options" and col in hs.columns:
            x = num(hs[col]).dropna()
            lv = sorted(x.unique())
            if len(lv) >= 5 and (x % 1 == 0).all() and lv == list(range(int(lv[0]), int(lv[0]) + len(lv))) and lv[0] in (0, 1):
                inc, why = True, "placeholder option text; numbered 1..k scale in the data"
        if inc and (DEMOG.search(strip_html(labels.get(col, ""))) or NAME_DEMOG.match(col)):
            inc, why = False, "demographic or party item (a grouping variable)"
        if inc and ATTN.search(strip_html(labels.get(col, ""))):
            inc, why = False, "attention check"
        if inc:
            sh, nh = numeric_share(hs[col])
            st, nt = numeric_share(ts[col]) if col in ts.columns else (0.0, 0)
            if nh == 0:
                inc, why = False, "empty in the human file"
            elif sh < 0.9 or st < 0.9:
                inc, why = False, f"not numeric (human {sh:.2f}, twin {st:.2f} of non-empty cells)"
            elif num(hs[col]).nunique() < 2:
                inc, why = False, "constant in the human file"
        if not inc:
            excl.append(dict(study=study, column=col, label=lab, status="excluded", reason=why))
        else:
            cand.append(dict(col=col, label=lab, qtype=qtype, why=why))
    # loop pooling
    groups = {}
    for c in cand:
        m = re.match(r"^(\d+)_(.+)$", re.sub(r"\.\d+$", "", c["col"]))
        if m:
            groups.setdefault(m.group(2), []).append(c)
    pooled_cols = {c["col"] for suf, g in groups.items() if len(g) >= 10 for c in g}
    outs = []
    for c in cand:
        if c["col"] in pooled_cols:
            continue
        x = num(hs[c["col"]])
        outs.append(dict(name=c["col"], cols=[c["col"]], pooled=False, label=c["label"], qtype=c["qtype"],
                         levels=int(x.nunique())))
    for suf, g in groups.items():
        if len(g) < 10:
            continue
        outs.append(dict(name=f"loopmean:{suf}", cols=[c["col"] for c in g], pooled=True,
                         label=f"{g[0]['label']} [mean over {len(g)} loop iterations]", qtype=g[0]["qtype"],
                         levels=int(max(num(hs[c["col"]]).nunique() for c in g))))
        for c in g:
            excl.append(dict(study=study, column=c["col"], label=c["label"], status="pooled",
                             reason=f"pooled into loopmean:{suf}"))
    return outs, excl, rs


def pair_frames(study, spec_path, cols):
    """Numeric human and twin frames on their common TWIN_IDs, columns cols (those present)."""
    hs, _ = read_values(os.path.join(spec_path, H_FILE), set(cols))
    ts, _ = read_values(os.path.join(spec_path, T_FILE), set(cols))
    idx = hs.index.intersection(ts.index)
    present = [c for c in cols if c in hs.columns and c in ts.columns]
    H = hs.loc[idx, present].apply(num)
    T = ts.loc[idx, present].apply(num)
    raw_t = ts.loc[idx, present]
    nonnum = int((raw_t.notna() & T.isna()).to_numpy().sum())
    lo, hi = H.min(), H.max()
    oor = int((T.notna() & ((T < lo) | (T > hi))).to_numpy().sum())
    return H, T, dict(n_h_rows=len(hs), n_t_rows=len(ts), n_matched=len(idx), cells=int(T.notna().to_numpy().sum()),
                      non_numeric=nonnum, outside_human_range=oor)


def outcome_series(out, H, T=None):
    """Human (and twin) series of one outcome; pooled outcomes are the respondent mean over the loop
    iterations that both answered."""
    if not out["pooled"]:
        c = out["cols"][0]
        if c not in H.columns:
            return None, None
        return H[c], (T[c] if T is not None else None)
    cols = [c for c in out["cols"] if c in H.columns]
    if not cols:
        return None, None
    h = H[cols].to_numpy(float)
    if T is None:
        m = ~np.isnan(h)
        k = m.sum(1)
        y = np.where(k > 0, np.nansum(h, 1) / np.maximum(k, 1), np.nan)
        return pd.Series(y, index=H.index), None
    t = T[cols].to_numpy(float)
    m = ~np.isnan(h) & ~np.isnan(t)
    k = m.sum(1)
    y = np.where(k > 0, np.where(m, h, 0).sum(1) / np.maximum(k, 1), np.nan)
    s = np.where(k > 0, np.where(m, t, 0).sum(1) / np.maximum(k, 1), np.nan)
    return pd.Series(y, index=H.index), pd.Series(s, index=H.index)


# ============================================================================== bootstrap gaps
def paired_gap(s, y, rng):
    """Twin gap, human gap, their SEs and covariance from one respondent bootstrap; groups are the
    two index sets already split: s = (sa, sb), y = (ya, yb)."""
    sa, sb = s
    ya, yb = y
    ia = rng.integers(0, len(sa), (B_BOOT, len(sa)))
    ib = rng.integers(0, len(sb), (B_BOOT, len(sb)))
    bg = sa[ia].mean(1) - sb[ib].mean(1)
    bp = ya[ia].mean(1) - yb[ib].mean(1)
    return (sa.mean() - sb.mean(), bg.std(ddof=1), ya.mean() - yb.mean(), bp.std(ddof=1),
            np.cov(bg, bp, ddof=1)[0, 1])


def twin_gaps(study, si, specs_used, outs, demo, spec_name, spec_path, skips):
    H, T, diag = pair_frames(study, spec_path, [c for o in outs for c in o["cols"]])
    d = demo.reindex(H.index)
    rows = []
    for oi, o in enumerate(outs):
        y, s = outcome_series(o, H, T)
        if y is None:
            continue
        ok = y.notna().to_numpy() & s.notna().to_numpy()
        yv, sv = y.to_numpy(float), s.to_numpy(float)
        for ci, (attr, a, b) in enumerate(CONTRASTS):
            ma = ok & (d[attr] == a).to_numpy()
            mb = ok & (d[attr] == b).to_numpy()
            if ma.sum() < MIN_N or mb.sum() < MIN_N:
                skips.append(dict(family=spec_name, study=study, outcome=o["name"], contrast=f"{a} - {b}",
                                  reason=f"group below {MIN_N} pairs (n_a={int(ma.sum())}, n_b={int(mb.sum())})"))
                continue
            rng = np.random.default_rng([SEED, si, oi, ci])
            g, se_g, p, se_p, cov = paired_gap((sv[ma], sv[mb]), (yv[ma], yv[mb]), rng)
            rows.append(dict(family=spec_name, contrast=f"{a} - {b}", g=g, se_g=se_g, gamma=p, se_gamma=se_p, cov=cov,
                             study=study, outcome=o["name"], outcome_label=o["label"], attribute=attr,
                             n_a=int(ma.sum()), n_b=int(mb.sum())))
    return rows, diag


def split_halves(pids, party, si):
    """Boolean array: True = half A. Stratified by party; seed 20261003 (and the study index)."""
    rng = np.random.default_rng([SEED, si, 99])
    a = np.zeros(len(pids), bool)
    strata = pd.Series(party.reindex(pids).fillna("missing").values)
    for k, (lev, idx) in enumerate(strata.groupby(strata).groups.items()):
        idx = np.array(list(idx))
        idx = idx[rng.permutation(len(idx))]
        n_a = len(idx) // 2 + (len(idx) % 2 if k % 2 == 0 else 0)
        a[idx[:n_a]] = True
    return a


def split_half_gaps(study, si, outs, demo, pool_path, skips):
    hs, _ = read_values(os.path.join(pool_path, H_FILE), {c for o in outs for c in o["cols"]})
    H = hs.apply(num)
    d = demo.reindex(H.index)
    isA = split_halves(H.index, demo["party"], si)
    rows = []
    for oi, o in enumerate(outs):
        y, _ = outcome_series(o, H)
        if y is None:
            continue
        yv = y.to_numpy(float)
        ok = ~np.isnan(yv)
        for ci, (attr, a, b) in enumerate(CONTRASTS):
            m = {}
            for half, hm in (("A", isA), ("B", ~isA)):
                m[half] = (ok & hm & (d[attr] == a).to_numpy(), ok & hm & (d[attr] == b).to_numpy())
            n = [int(x.sum()) for half in "AB" for x in m[half]]
            if min(n) < MIN_N:
                skips.append(dict(family="human_split_half", study=study, outcome=o["name"], contrast=f"{a} - {b}",
                                  reason=f"group below {MIN_N} respondents in a half (A: {n[0]}, {n[1]}; B: {n[2]}, {n[3]})"))
                continue
            rng = np.random.default_rng([SEED, si, oi, ci, 7])
            res = {}
            for half in "AB":
                xa, xb = yv[m[half][0]], yv[m[half][1]]
                ia = rng.integers(0, len(xa), (B_BOOT, len(xa)))
                ib = rng.integers(0, len(xb), (B_BOOT, len(xb)))
                bg = xa[ia].mean(1) - xb[ib].mean(1)
                res[half] = (xa.mean() - xb.mean(), bg.std(ddof=1))
            rows.append(dict(family="human_split_half", contrast=f"{a} - {b}", g=res["A"][0], se_g=res["A"][1],
                             gamma=res["B"][0], se_gamma=res["B"][1], cov=0.0, study=study, outcome=o["name"],
                             outcome_label=o["label"], attribute=attr, n_a=n[0] + n[1], n_b=n[2] + n[3]))
    return rows


# ============================================================================== load
def load():
    os.makedirs(OUT, exist_ok=True)
    demo = demographics()
    skips, excl_all, inv, studies, parse, rows, inputs = [], [], [], [], [], [], []
    for si, study in enumerate(STUDIES):
        specs, notes = spec_dirs(study)
        if not specs:
            studies.append(dict(study=study, n_human=0, n_specs=0, specifications="", outcomes=0,
                                join_on_pid="", note="no consolidated values files; " + "; ".join(notes)))
            continue
        outs, excl, rs = build_outcomes(study, specs)
        excl_all += excl
        pool = specs[rs]
        for o in outs:
            pass
        # inventory over specifications
        per_spec_n = {}
        for k, p in sorted(specs.items()):
            H, T, diag = pair_frames(study, p, [c for o in outs for c in o["cols"]])
            diag.update(study=study, family=k)
            parse.append(diag)
            per_spec_n[k] = (H, T)
        Hr, Tr = per_spec_n[rs]
        seen_partial = 0
        for o in outs:
            y, s = outcome_series(o, Hr, Tr)
            if y is None:
                continue
            ok_h = int(y.notna().sum())
            ok_p = int((y.notna() & s.notna()).sum())
            specs_with = []
            for k, (H, T) in per_spec_n.items():
                yy, ss = outcome_series(o, H, T)
                if yy is not None and int((yy.notna() & ss.notna()).sum()) >= 1:
                    specs_with.append(k)
            seen_partial += ok_h < 0.9 * len(Hr)
            inv.append(dict(study=study, outcome=o["name"], label=o["label"], qtype=o["qtype"], levels=o["levels"],
                            pooled=o["pooled"], n_human=ok_h, n_twin_matched=ok_p, n_specifications=len(specs_with),
                            specifications=";".join(specs_with)))
        pool_ids = Hr.index
        studies.append(dict(study=study, n_human=len(pool_ids), n_specs=len(specs), specifications=";".join(sorted(specs)),
                            outcomes=len(outs), outcomes_seen_by_under_90pct=int(seen_partial),
                            join_on_pid=f"human and twin rows match on TWIN_ID in all {len(specs)} specifications"
                            if all(len(a.index) == len(b.index) for a, b in per_spec_n.values()) else "see parse_receipts",
                            reference_spec=rs, profile_match=int(pool_ids.isin(demo.index).sum()),
                            note="; ".join(notes)))
        for k, p in sorted(specs.items()):
            r, _ = twin_gaps(study, si, specs, outs, demo, k, p, skips)
            rows += r
            for f in (H_FILE, T_FILE):
                inputs.append(os.path.join(p, f))
        rows += split_half_gaps(study, si, outs, demo, pool, skips)
        print(f"{study}: {len(specs)} specs, {len(outs)} outcomes, {len(rows)} gap rows so far", flush=True)
    gaps = pd.DataFrame(rows)
    gaps["family_size"] = gaps.groupby("family")["contrast"].transform("size")
    pd.DataFrame(inv).to_csv(os.path.join(OUT, "inventory.csv"), index=False)
    pd.DataFrame(studies).to_csv(os.path.join(OUT, "inventory_studies.csv"), index=False)
    pd.DataFrame(excl_all).to_csv(os.path.join(OUT, "exclusions.csv"), index=False)
    pd.DataFrame(skips).to_csv(os.path.join(OUT, "contrasts_not_formed.csv"), index=False)
    pd.DataFrame(parse).to_csv(os.path.join(OUT, "parse_receipts.csv"), index=False)
    inputs = sorted(set(inputs + [PROFILE]))
    return dict(mode="gaps", gaps=gaps, inputs=inputs,
                notes="family = twin specification across all studies, outcomes and contrasts; human_split_half is the "
                      "ideal-simulator control (half A simulator, half B reference, cov 0)")


if __name__ == "__main__":
    # inventory only (no gaps): python adapters/twin2k_mega.py
    demo = demographics()
    for si, study in enumerate(STUDIES):
        specs, notes = spec_dirs(study)
        if not specs:
            print(study, "no files", notes)
            continue
        outs, excl, rs = build_outcomes(study, specs)
        print(f"== {study}: {len(specs)} specs, ref {rs}, {len(outs)} outcomes", notes)
        for o in outs:
            print("   ", o["name"][:30].ljust(30), o["qtype"].ljust(16), o["levels"], o["label"][:90].replace("\n", " "))
