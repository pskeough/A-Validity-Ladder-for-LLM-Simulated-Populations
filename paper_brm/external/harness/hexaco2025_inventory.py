"""Inventory of the HEXACO 2025 files: agents, runs, attributes, item coding, key checks, human file.
Writes results/hexaco2025/inventory.csv (section, key, value, note) and ipip_keying.csv."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hexaco2025_lib as L

os.makedirs(L.OUT, exist_ok=True)
rows = []


def add(section, key, value, note=""):
    rows.append(dict(section=section, key=key, value=value, note=note))


# ---- agents
pop = json.load(open(os.path.join(L.RAW, "data", "pop_census.json"), encoding="utf-8"))
ages = np.array([p["Age"] for p in pop], float)
add("agents", "population", "PopCensus (pop_census.json)", "310 census-style biographies, England and Wales occupation codes")
add("agents", "n_agents", len(pop))
add("agents", "attributes recorded", "Full Name, Age, Occupation, OC (occupation code), Hobbies/Interests, Personality Facts",
    "no sex/gender field, no education, no region; sex could only be guessed from first names (not used)")
add("agents", "age mean/sd/min/max", f"{ages.mean():.2f} / {ages.std(ddof=1):.2f} / {ages.min():.0f} / {ages.max():.0f}")
add("agents", "distinct occupations", len({p['Occupation'] for p in pop}))
add("agents", "PopProfessional (pop_professional.json)", "present",
    "used by the paper only for the lexical (adjective) survey; no HEXACO-PI-R-100 answers exist for it")

names = [p["Full Name"].replace(" ", "").split(".")[0] for p in pop]
for m in L.MODELS:
    raw_txt = pd.read_csv(os.path.join(L.RAW, "data", L.MODELS[m]), index_col=0)
    raw = L.load_agents(m)
    add("model", f"{m}: rows x items", f"{raw.shape[0]} x {raw.shape[1]}", "one text answer per agent and item")
    add("model", f"{m}: distinct agent ids", raw.index.nunique(), "no agent appears twice, so no repeated draws of one persona")
    add("model", f"{m}: agents in pop_census.json order-matched", int(len(set(raw.index) & set(names))), "of 310 after fix_name()")
    nan = int(raw.isna().to_numpy().sum())
    add("model", f"{m}: unparsed answers (any item)", nan,
        "answers whose leading label is not one of the five Likert labels (e.g. [content-filtered])")
    bad = raw.isna().sum()
    bad = bad[bad > 0]
    add("model", f"{m}: unparsed by item (1-based)", json.dumps({int(k) + 1: int(v) for k, v in bad.items()}), "")
    add("model", f"{m}: agents with an unparsed answer in items 1-96", int(raw.loc[:, :95].isna().any(axis=1).sum()), "")
    # answers that carry text after the label (free-text reasoning)
    add("model", f"{m}: answers with text beyond the label", int((raw_txt.map(lambda s: len(str(s)) > 22)).to_numpy().sum()),
        "length > 22 characters")

add("runs", "repeated runs of one agent under one prompt", "none",
    "each model file holds one answer per agent and item; popc_responses_1..6 are six 50-agent row batches of the lexical adjective survey (data_prep.ipynb concatenates them: pd.concat(parts)), not six runs")
add("runs", "gate / level 1 / R3", "not applicable", "need repeated draws of one persona")
add("runs", "R4", "not applicable",
    "agents carry Age and Occupation; the Open Psychometrics reference records neither (only country, elapse)")

# ---- item coding and keys
add("coding", "answer scale", "Strongly disagree=1 ... Strongly agree=5 (text labels leading each answer)")
add("coding", "item order", "columns 0-99 = HEXACO-PI-R-100 items 1-100 (support/hexaco_pi_r.py questions list)")
# the repo's scoring dict vs the official key
sys.path.insert(0, L.RAW)
import importlib.util
spec = importlib.util.spec_from_file_location("hexaco_pi_r", os.path.join(L.RAW, "support", "hexaco_pi_r.py"))
src = open(os.path.join(L.RAW, "support", "hexaco_pi_r.py"), encoding="utf-8").read()
ns = {}
exec(compile(src.split("def load_responses")[0], "hexaco_pi_r_head", "exec"), ns)
repo = ns["scoring"]
off_dom = {}
for (d, f), k in L.PIR_KEY.items():
    for i, r in L.parse_key(k):
        off_dom.setdefault(d, {})[i] = r
repo_dom = {}
for d, (pos, neg) in repo.items():
    dd = "Openness" if d == "Openness" else d
    repo_dom[dd] = {**{i: False for i in pos}, **{i: True for i in neg}}
same = all(off_dom[d] == repo_dom[d] for d in L.DOMAINS)
add("coding", "repo scoring dict == hexaco.org ScoringKeys_100 (96 items, keys and reverse flags)", str(same))
add("coding", "questions in repo list", len(ns["questions"]))
add("coding", "facet structure", "from hexaco.org ScoringKeys_100.pdf: 24 facets x 4 items + Altruism 97, 98, 99R, 100R (excluded from domains)")
add("coding", "domain score", "mean of the domain's 16 keyed items (4 facets x 4)", "repo and hexaco.org rule; Altruism excluded")

# ---- human file
hum = L.load_human_raw()
add("human", "file", "HEXACO.zip -> HEXACO/data.csv (tab-separated)", "openpsychometrics.org/_rawdata/")
add("human", "instrument", "IPIP HEXACO-equivalent scales, 240 items (24 facets x 10)",
    "listing text: 'Answers to the IPIP HEXACO equivalent scales'; codebook item codes e.g. HSinc1; NOT the HEXACO-PI-R")
add("human", "rows", len(hum))
add("human", "scale", "1-7 (strongly disagree ... strongly agree); codebook labels 5 as 'slightly disagree' (typo for slightly agree)")
add("human", "other columns", "V1, V2 (validity items), country, elapse (seconds)", "no age, sex, education or occupation")
items = [c for c in hum.columns if c[0] in "HEXACO" and c[1:2].isupper() and c not in ("V1", "V2")]
add("human", "item columns", len(items))
add("human", "item values outside 1-7 (any)", int(((hum[items] < 1) | (hum[items] > 7)).to_numpy().sum()))
add("human", "V1/V2 value counts", json.dumps({"V1": hum.V1.value_counts().sort_index().to_dict(), "V2": hum.V2.value_counts().sort_index().to_dict()}, default=int))
add("human", "country: n distinct / GB n / US n", f"{hum.country.nunique()} / {(hum.country == 'GB').sum()} / {(hum.country == 'US').sum()}")
add("human", "elapse median / p5 / p95 (s)", f"{hum.elapse.median():.0f} / {hum.elapse.quantile(.05):.0f} / {hum.elapse.quantile(.95):.0f}")

kd = L.ipip_keying()
kd.to_csv(os.path.join(L.OUT, "ipip_keying.csv"), index=False)
cnt = kd.groupby("facet").size()
add("human", "IPIP key parsed: items matched", len(kd), "of 240 (item text on ipip.ori.org/newHEXACO_PI_key.htm matched to the codebook text)")
add("human", "IPIP key: facets with exactly 10 items", int((cnt == 10).sum()), "of 24")
add("human", "IPIP key: min / median match ratio", f"{kd.ratio.min():.3f} / {kd.ratio.median():.3f}")
add("human", "IPIP key: items with ratio < 0.6", ", ".join(kd[kd.ratio < 0.6].item))
# verification: after keying, every item should correlate positively with the rest of its facet
neg = []
sign = dict(zip(kd.item, kd.sign))
for pref, g in kd.groupby("facet"):
    X = np.column_stack([hum[i].to_numpy(float) if sign[i] > 0 else 8 - hum[i].to_numpy(float) for i in g.item])
    for j, it in enumerate(g.item):
        r = np.corrcoef(X[:, j], np.delete(X, j, 1).mean(1))[0, 1]
        if r <= 0.15:
            neg.append((it, round(r, 3)))
add("human", "IPIP key check: items with corrected item-facet r <= .15 after keying", json.dumps(neg),
    "empty list = every parsed keying direction is consistent with the data")

pd.DataFrame(rows).to_csv(os.path.join(L.OUT, "inventory.csv"), index=False)
print(pd.DataFrame(rows).to_string(max_colwidth=110))
