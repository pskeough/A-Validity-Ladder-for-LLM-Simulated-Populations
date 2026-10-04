"""Gate (not applicable), level 3 on domain means, and the receipt for the HEXACO 2025 adapter.

gate.csv     one row per model and domain: gate, level 1 and R3 need repeated draws of one persona; the
             files hold one answer per agent and item, so every row is 'not applicable'.
l3.csv       level 3 on domain means (and, as description, facet means) of each model's 310 agents against the
             published HEXACO-PI-R-100 self-report means of the hexaco.org college sample (N = 1126;
             descriptives_100.pdf), TOST at alpha .05 with the package's level3_tost, tolerance 0.5 reference SD
             (0.2 SD beside it). One 'Overall' row per domain: the reference records no attribute the agents
             carry, so no post-stratification is possible.
receipt.csv  the mean, SD and Cronbach alpha per model and domain that hexaco_pi_r.ipynb prints (stored output
             of process_pir), recomputed here from the response files.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hexaco2025_lib as L
from validity_ladder.level3 import level3_model_verdict, level3_tost, read_tost

os.makedirs(L.OUT, exist_ok=True)
keys = L.pir_item_keys()
agents_raw = {m: L.load_agents(m) for m in L.MODELS}

# ---------------------------------------------------------------------------------------------- gate
g = []
for m in L.MODELS:
    for d in L.DOMAINS:
        g.append(dict(model=m, domain=d, rung="gate", verdict="not applicable", se_k=np.nan, phi_k=np.nan,
                      reason="one answer per agent and item in the released file (310 distinct agents, no repeated run); "
                             "SE(k) of an agent's domain score cannot be estimated without repeated draws of the same agent"))
pd.DataFrame(g).to_csv(os.path.join(L.OUT, "gate.csv"), index=False)

# ---------------------------------------------------------------------------------------------- level 3
rows = []
for m in L.MODELS:
    k = L.keyed(agents_raw[m])
    fac_scores = {f: k[[i for i, _ in lst]].mean(axis=1) for f, lst in keys.items()}
    dom_scores = {}
    for d in L.DOMAINS:
        cols = sorted(i for f, lst in keys.items() if L.PIR_KEY_DOMAIN[f] == d for i, _ in lst)
        dom_scores[d] = k[cols].mean(axis=1)  # the instrument's rule: mean of the 16 keyed items
    for level, scores in (("domain", dom_scores), ("facet", fac_scores)):
        for name, x in scores.items():
            alpha_h, mean_h, sd_h = L.HUMAN_DESC[name]
            x = x.to_numpy(float)
            n = len(x)
            resid = float(x.mean() - mean_h)
            var_sim = float(x.var(ddof=1) / n)
            var_ref = sd_h ** 2 / L.HUMAN_N
            tol = 0.5 * sd_h
            t = level3_tost(resid, var_sim, var_ref, L.HUMAN_N - 1, tol)
            rows.append(dict(model=m, level=level, domain=L.PIR_KEY_DOMAIN.get(name, name), scale=name,
                             n_agents=n, sim_mean=float(x.mean()), sim_sd=float(x.std(ddof=1)), ref_mean=mean_h, ref_sd=sd_h,
                             ref_n=L.HUMAN_N, resid=resid, resid_sd_units=resid / sd_h, sd_ratio=float(x.std(ddof=1)) / sd_h,
                             se=t["se"], df=t["df"], ci90_lo=t["ci90_lo"], ci90_hi=t["ci90_hi"], tolerance=tol,
                             verdict=t["verdict"], verdict_0p2sd=read_tost(t["ci90_lo"], t["ci90_hi"], 0.2 * sd_h)))
l3 = pd.DataFrame(rows)
mv = []
for m, gm in l3[l3.level == "domain"].groupby("model", sort=False):
    mv.append(dict(model=m, level="model verdict (6 domains)", domain="all", scale="all", verdict=level3_model_verdict(gm.verdict.tolist()),
                   verdict_0p2sd=level3_model_verdict(gm.verdict_0p2sd.tolist())))
l3 = pd.concat([l3, pd.DataFrame(mv)], ignore_index=True)
l3.to_csv(os.path.join(L.OUT, "l3.csv"), index=False)

# ---------------------------------------------------------------------------------------------- receipt
nb = json.load(open(os.path.join(L.RAW, "hexaco_pi_r.ipynb"), encoding="utf-8"))
pub = {}
order = list(L.MODELS)  # GPT-4, llama3.2, phi4, sonnet = the notebook's GPT4, llama3.2, phi4, sonnet
outs = []
for ci, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    for o in c.get("outputs", []):
        txt = "".join(o.get("text", [])) or "".join(o.get("data", {}).get("text/plain", []))
        if txt.lstrip().startswith("mean") and "alpha" in txt.split("\n")[0]:
            outs.append((ci, txt))
assert len(outs) == 4, len(outs)
rec = []
for m, (ci, txt) in zip(order, outs):
    k = L.keyed(agents_raw[m])
    for line in txt.strip().split("\n")[1:]:
        mm = re.match(r"^(\S+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)$", line.strip())
        d = mm.group(1)
        pm, ps, pa = (float(mm.group(i)) for i in (2, 3, 4))
        cols = sorted(i for f, lst in keys.items() if L.PIR_KEY_DOMAIN[f] == d for i, _ in lst)
        X = k[cols]
        mean = float(X.mean().mean())
        sd = float(X.std().mean())  # mean of the item SDs (support/hexaco_pi_r.py calc_mean_sd_alpha)
        kk = X.shape[1]
        alpha = float(kk / (kk - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1)))
        for stat, ours, p in (("mean", mean, pm), ("item-mean sd", sd, ps), ("Cronbach alpha", alpha, pa)):
            rec.append(dict(item=f"{m} | {d} | {stat}", ours=round(ours, 4), published=p,
                            abs_diff=round(abs(round(ours, 4) - p), 4),
                            source_location=f"hexaco-rep-public hexaco_pi_r.ipynb cell {ci} stored output (process_pir {m}); alpha via pingouin.cronbach_alpha"))
r = pd.DataFrame(rec)
r.to_csv(os.path.join(L.OUT, "receipt.csv"), index=False)
print(r.abs_diff.describe())
print(r.sort_values("abs_diff", ascending=False).head(6).to_string())
print(l3[l3.level.str.startswith("model")].to_string())
print(l3[l3.level == "domain"][["model", "scale", "sim_mean", "ref_mean", "ref_sd", "resid_sd_units", "ci90_lo", "ci90_hi", "tolerance", "verdict", "verdict_0p2sd", "sd_ratio"]].to_string())
