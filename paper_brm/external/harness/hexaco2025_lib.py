"""Shared code for the Mercer, Martin & Swatton (2025) HEXACO adapter of the Validity Ladder.

Simulated: item-level HEXACO-PI-R-100 answers of 310 generated agents (PopCensus) from four models,
one answer per agent and item (github.com/alan-turing-institute/hexaco-rep-public, MIT).
Reference: Open Psychometrics HEXACO.zip, which is the 240-item IPIP HEXACO-equivalent inventory
(Ashton/Lee; ipip.ori.org/newHEXACO_PI_key.htm), NOT the HEXACO-PI-R. The two instruments share no item
wording, so no item-by-item mapping exists. They share facet labels (23 of 24), so level 4 is read on
facet scores (see hexaco2025_l4.py). The only PI-R-100 human reference is the published self-report
descriptive table of hexaco.org (N = 1126 students; means, SDs, alphas), used for level 3.

Decisions fixed on 3 Oct 2026 before any ladder number was computed (not to be edited after results):
  * Level 4 unit: a domain; indicators are its facet scores, scored as the mean of the keyed items.
  * Primary domain set: Honesty-Humility, Emotionality, Agreeableness, Conscientiousness, Openness with
    their four shared facets; Extraversion with the THREE shared facets (Social Boldness, Sociability,
    Liveliness), because PI-R Social Self-Esteem and IPIP Expressiveness are different constructs.
    Extraversion with the four positional facets (SSE paired with Expressiveness) is a sensitivity row.
  * Primary reference: respondents with V1 >= 4 and V2 >= 4 (the two validity items, 7-point), all
    countries, facet = mean of its ten keyed items.
  * Sensitivities: (S1) reference facets built from random four-item subsets of each IPIP facet (matches
    the PI-R-100's four-item facets); (S2) no validity filter; (S3) respondents from GB only.
  * Fit: Pearson correlations of facet scores (composites of 4 to 10 items; polychorics apply to item
    answers), package one_factor_uls, congruence, r2_verdict, general_factor_vs_ref, cluster_boot_mult.
"""
import html
import json
import os
import re
from html.parser import HTMLParser

import numpy as np
import pandas as pd

EXT = r"C:\Research\PsychBench\UpdatedRun\paper_brm\external"
RAW = os.path.join(EXT, "raw", "hexaco2025")
OUT = os.path.join(EXT, "harness", "results", "hexaco2025")
WORK = os.path.join(EXT, "harness", "_work_hexaco2025")
MODELS = {"GPT-4": "hexaco-pi-r-responses-GPT-4.csv", "llama3.2": "hexaco-pi-r-responses-llama3.2.csv",
          "phi4": "hexaco-pi-r-responses-phi4.csv", "sonnet": "hexaco-pi-r-responses-sonnet.csv"}
SEED = 20261003

# Official HEXACO-PI-R 100-item key, hexaco.org/downloads/ScoringKeys_100.pdf (item numbers 1-100; R = reverse).
PIR_KEY = {
    ("Honesty-Humility", "Sincerity"): "6R 30 54R 78",
    ("Honesty-Humility", "Fairness"): "12R 36R 60 84R",
    ("Honesty-Humility", "Greed Avoidance"): "18 42R 66R 90R",
    ("Honesty-Humility", "Modesty"): "24 48 72R 96R",
    ("Emotionality", "Fearfulness"): "5 29R 53 77R",
    ("Emotionality", "Anxiety"): "11 35R 59R 83",
    ("Emotionality", "Dependence"): "17 41R 65 89R",
    ("Emotionality", "Sentimentality"): "23 47 71 95R",
    ("Extraversion", "Social Self-Esteem"): "4 28 52R 76R",
    ("Extraversion", "Social Boldness"): "10R 34 58 82R",
    ("Extraversion", "Sociability"): "16R 40 64 88",
    ("Extraversion", "Liveliness"): "22 46 70R 94R",
    ("Agreeableness", "Forgiveness"): "3 27 51R 75R",
    ("Agreeableness", "Gentleness"): "9R 33 57 81",
    ("Agreeableness", "Flexibility"): "15R 39 63R 87R",
    ("Agreeableness", "Patience"): "21R 45 69 93R",
    ("Conscientiousness", "Organization"): "2 26 50R 74R",
    ("Conscientiousness", "Diligence"): "8 32 56R 80R",
    ("Conscientiousness", "Perfectionism"): "14 38R 62 86",
    ("Conscientiousness", "Prudence"): "20R 44R 68 92R",
    ("Openness", "Aesthetic Appreciation"): "1R 25R 49 73",
    ("Openness", "Inquisitiveness"): "7 31 55R 79R",
    ("Openness", "Creativity"): "13R 37 61 85R",
    ("Openness", "Unconventionality"): "19R 43 67 91R",
}
ALTRUISM = "97 98 99R 100R"
PIR_KEY_DOMAIN = {f: d for (d, f) in PIR_KEY}
DOMAINS = ["Honesty-Humility", "Emotionality", "Extraversion", "Agreeableness", "Conscientiousness", "Openness"]

# PI-R facet -> IPIP facet code (Open Psychometrics column prefix). None = no counterpart.
IPIP_OF = {
    "Sincerity": "HSinc", "Fairness": "HFair", "Greed Avoidance": "HGree", "Modesty": "HMode",
    "Fearfulness": "EFear", "Anxiety": "EAnxi", "Dependence": "EDepe", "Sentimentality": "ESent",
    "Social Self-Esteem": "XExpr",  # positional pairing only; not the same construct (sensitivity row)
    "Social Boldness": "XSocB", "Sociability": "XSoci", "Liveliness": "XLive",
    "Forgiveness": "AForg", "Gentleness": "AGent", "Flexibility": "AFlex", "Patience": "APati",
    "Organization": "COrga", "Diligence": "CDili", "Perfectionism": "CPerf", "Prudence": "CPrud",
    "Aesthetic Appreciation": "OAesA", "Inquisitiveness": "OInqu", "Creativity": "OCrea", "Unconventionality": "OUnco",
}
# domain read -> (domain, facets, note)
DOMAIN_SETS = {}
for d in DOMAINS:
    fac = [f for (dd, f) in PIR_KEY if dd == d]
    if d == "Extraversion":
        DOMAIN_SETS["Extraversion"] = (d, [f for f in fac if f != "Social Self-Esteem"], "3 shared facets (SSE excluded)")
        DOMAIN_SETS["Extraversion_4pos"] = (d, fac, "4 positional facets (SSE paired with Expressiveness)")
    else:
        DOMAIN_SETS[d] = (d, fac, "4 shared facets")

# Published human HEXACO-PI-R-100 self-report, college sample N = 1126 (hexaco.org/downloads/descriptives_100.pdf)
HUMAN_N = 1126
HUMAN_DESC = {  # name: (alpha, mean, sd)
    "Honesty-Humility": (.83, 3.19, .62), "Sincerity": (.65, 3.20, .78), "Fairness": (.76, 3.34, .98),
    "Greed Avoidance": (.84, 2.72, .98), "Modesty": (.69, 3.49, .78),
    "Emotionality": (.84, 3.43, .62), "Fearfulness": (.73, 3.06, .89), "Anxiety": (.68, 3.69, .81),
    "Dependence": (.79, 3.38, .87), "Sentimentality": (.71, 3.58, .80),
    "Extraversion": (.85, 3.50, .57), "Social Self-Esteem": (.68, 3.85, .68), "Social Boldness": (.74, 3.03, .87),
    "Sociability": (.69, 3.59, .75), "Liveliness": (.77, 3.52, .77),
    "Agreeableness": (.84, 2.94, .58), "Forgiveness": (.75, 2.75, .83), "Gentleness": (.66, 3.17, .73),
    "Flexibility": (.59, 2.74, .72), "Patience": (.77, 3.11, .86),
    "Conscientiousness": (.82, 3.44, .56), "Organization": (.72, 3.26, .91), "Diligence": (.67, 3.79, .68),
    "Perfectionism": (.68, 3.50, .78), "Prudence": (.69, 3.18, .75),
    "Openness": (.81, 3.41, .60), "Aesthetic Appreciation": (.65, 3.34, .88), "Inquisitiveness": (.63, 3.19, .88),
    "Creativity": (.74, 3.63, .85), "Unconventionality": (.50, 3.46, .64),
}


def parse_key(s):
    """'6R 30 54R 78' -> [(6, True), (30, False), ...] (item number 1-based, reverse flag)."""
    return [(int(t.rstrip("R")), t.endswith("R")) for t in s.split()]


def pir_item_keys():
    """dict facet -> list of (item0, reverse) with zero-based item index."""
    return {f: [(i - 1, r) for i, r in parse_key(k)] for (_, f), k in PIR_KEY.items()}


# ----------------------------------------------------------------------------- Mercer et al. agents
def parse_response(s):
    """Likert value 1-5 from the start of a text answer; NaN when none of the five labels leads it.
    Same rule as support/hexaco_pi_r.py responses_to_scores (strongly disagree 1, disagree 2, neutral 3,
    agree 4, strongly agree 5), applied to the leading label."""
    t = str(s).strip().lower()
    for lab, v in (("strongly disagree", 1), ("disagree", 2), ("neutral", 3), ("strongly agree", 5), ("agree", 4)):
        if t.startswith(lab):
            return v
    return np.nan


def load_agents(model):
    """(310, 100) raw 1-5 answers (NaN where unparsed), index = agent name."""
    df = pd.read_csv(os.path.join(RAW, "data", MODELS[model]), index_col=0)
    df.columns = range(len(df.columns))
    return df.map(parse_response).astype(float)


def keyed(raw, keys=None):
    """Reverse-keyed answers (6 - x for R items), columns 0..99 unchanged order."""
    x = raw.copy()
    rev = sorted({i for lst in pir_item_keys().values() for (i, r) in lst if r} | {98, 99})
    x[rev] = 6 - x[rev]
    return x


def facet_scores_agents(raw):
    k = keyed(raw)
    return pd.DataFrame({f: k[[i for i, _ in lst]].mean(axis=1, skipna=False) for f, lst in pir_item_keys().items()})


# ----------------------------------------------------------------------------- IPIP key parsing
class _Txt(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []

    def handle_data(self, d):
        self.out.append(d)


def _norm(s):
    s = html.unescape(s).replace("\xa0", " ").replace("\u2019", "'").lower()
    s = re.sub(r"[^a-z' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_ipip_key():
    """List of (facet_code like 'H:Sinc', keyed +1/-1, item text) from the IPIP key page."""
    raw = open(os.path.join(RAW, "ipip_key", "newHEXACO_PI_key.htm"), encoding="cp1252", errors="replace").read()
    p = _Txt()
    p.feed(raw)
    txt = html.unescape("".join(p.out)).replace("\xa0", " ")
    txt = txt[txt.find("The Items in the Preliminary"):]
    paras = [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n\s*\n", txt)]
    paras = [b for b in paras if b]
    facet, sign, out = None, None, []
    for b in paras:
        m = re.search(r"\(([HEXACO]):(\w+)\)", b)
        if m:
            facet, sign = f"{m.group(1)}:{m.group(2)}", None
            continue
        if b in ("+ keyed", "- keyed"):
            sign = 1 if b[0] == "+" else -1
            continue
        if facet and sign and "Alpha" not in b and not re.fullmatch(r"[\[\]\d\. =]*", b):
            out.append((facet, sign, b))
    return out


def ipip_keying():
    """DataFrame item, facet (Open Psychometrics code), sign (+1/-1), key_text, codebook_text, ratio.

    Item text on the IPIP page is stated without the leading 'I'; matched to the Open Psychometrics
    codebook text within the facet by best difflib ratio, one-to-one."""
    import difflib
    cb = {}
    for line in open(os.path.join(RAW, "human", "HEXACO", "codebook.txt"), encoding="utf-8", errors="replace"):
        m = re.match(r"^([HEXACO][A-Za-z]+\d+)\s+(.*)$", line.rstrip("\n"))
        if m:
            cb[m.group(1)] = m.group(2)
    key = parse_ipip_key()
    code_of = {"H:Sinc": "HSinc", "H:Fair": "HFair", "H:Gree": "HGree", "H:Mode": "HMode", "E:Fear": "EFear",
               "E:Anxi": "EAnxi", "E:Depe": "EDepe", "E:Sent": "ESent", "X:Expr": "XExpr", "X:SocB": "XSocB",
               "X:Soci": "XSoci", "X:Live": "XLive", "A:Forg": "AForg", "A:Gent": "AGent", "A:Flex": "AFlex",
               "A:Pati": "APati", "C:Orga": "COrga", "C:Dili": "CDili", "C:Perf": "CPerf", "C:Prud": "CPrud",
               "O:AesA": "OAesA", "O:Inqu": "OInqu", "O:Crea": "OCrea", "O:Unco": "OUnco"}
    rows = []
    for fc, pref in code_of.items():
        ks = [(sg, t) for f, sg, t in key if f == fc]
        items = {c: t for c, t in cb.items() if re.fullmatch(pref + r"\d+", c)}
        # greedy best-pair assignment
        pairs = sorted(((difflib.SequenceMatcher(None, _norm(t), _norm(ct)).ratio(), i, c)
                        for i, (sg, t) in enumerate(ks) for c, ct in items.items()), reverse=True)
        used_k, used_c = set(), set()
        for r, i, c in pairs:
            if i in used_k or c in used_c:
                continue
            used_k.add(i)
            used_c.add(c)
            rows.append(dict(item=c, facet=pref, sign=ks[i][0], key_text=ks[i][1], codebook_text=items[c], ratio=round(r, 3)))
    return pd.DataFrame(rows).sort_values("item").reset_index(drop=True)


# ----------------------------------------------------------------------------- Open Psychometrics humans
def load_human_raw():
    df = pd.read_csv(os.path.join(RAW, "human", "HEXACO", "data.csv"), sep="\t", keep_default_na=False, low_memory=False)
    df["elapse"] = pd.to_numeric(df["elapse"], errors="coerce")  # some cells are UTF-7 junk such as +AC0-1810
    return df


def human_facet_scores(df, keying, items_per_facet=None, rng=None):
    """Facet scores of humans: mean of keyed items (8 - x for negatively keyed on the 7-point scale).
    items_per_facet = 4: a random subset of that many items per facet (rng)."""
    out = {}
    for pref, g in keying.groupby("facet"):
        its = list(g.item)
        if items_per_facet is not None:
            its = list(rng.choice(its, items_per_facet, replace=False))
        sign = dict(zip(g.item, g.sign))
        cols = [df[i].to_numpy(float) if sign[i] > 0 else 8.0 - df[i].to_numpy(float) for i in its]
        out[pref] = np.mean(cols, axis=0)
    return pd.DataFrame(out, index=df.index)
