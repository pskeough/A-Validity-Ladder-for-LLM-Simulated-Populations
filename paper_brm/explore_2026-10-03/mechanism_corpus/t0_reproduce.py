"""Reproduction checks: the pipeline here must return the audit's published numbers before any new
quantity is trusted. Writes reproduction_checks.csv and prints a summary.

  L3  80c 'PS' rows (S1 primary, corpus all, combined framing, mean) for 4 models + pooled x 10 groups:
      residual and 90% limits.
  L2  78c headline rows (analysis == headline) for 4 models x 2 estimands x 7 contrasts: ratio and the
      frozen verdict, and the DeepSeek-V3 standardised Low minus High ratio 1.91 of l2_headline.csv.
  L1  pattern reuse ratio (DeepSeek-V3 clinical 34.83) and item profile at matched totals
      (DeepSeek-V3 item 1 minus NHANES +0.1754) from l1_pattern_reuse.csv, l1_personfit_item_profile.csv.
"""
import os

import numpy as np
import pandas as pd

from common import (BRM, CELLS, DPQ, FULLNAME, ITEMS, MODELS, OUT, CellRef, cis_matched, l2_reference,
                    SHORT, l2_sim_gaps, l3_rows, load_corpus, load_nhanes, r3_verdict, sim_cells)

rows = []


def add(check, item, got, want, tol, text=False):
    ok = (got == want) if text else bool(abs(got - want) <= tol)
    rows.append(dict(check=check, item=item, got=got if text else round(float(got), 6),
                     want=want if text else round(float(want), 6), tol=tol, ok=bool(ok)))


# ------------------------------------------------------------------------------------ L3
nh = load_nhanes()
cref = CellRef(nh)
corpus = load_corpus()
x = cis_matched(corpus)
sims = sim_cells(x)
ref = pd.read_csv(os.path.join(BRM, "80c_l3_results.csv"), low_memory=False)
ref = ref[(ref.spec == "S1 primary") & (ref.corpus == "all") & (ref.outcome == "mean") &
          (ref.framing == "combined") & (ref.estimand == "PS")].set_index(["model", "group"])
for mdl in MODELS + ["pooled"]:
    t = sims[(mdl, "combined")]
    for r in l3_rows(cref, t["s"], t["v"]):
        w = ref.loc[(mdl, r["group"])]
        add("L3 resid", f"{mdl} | {r['group']}", r["resid"], w.resid, 2e-4)
        add("L3 ci90_lo", f"{mdl} | {r['group']}", r["ci90_lo"], w.ci90_lo, 2e-4)
        add("L3 ci90_hi", f"{mdl} | {r['group']}", r["ci90_hi"], w.ci90_hi, 2e-4)

# ------------------------------------------------------------------------------------ L2
gaps = l2_sim_gaps()
rf = l2_reference()
vd = pd.read_csv(os.path.join(BRM, "l2_verdicts.csv"), low_memory=False)
vd = vd[vd.analysis == "headline"].set_index(["estimand", "contrast", "scope"])
for est in ("standardised", "marginal"):
    for (e, c) in [k for k in rf.index if k[0] == est]:
        for mdl in MODELS:
            g, se, df = gaps[(FULLNAME[mdl], c)]
            out = r3_verdict(g, se, df, rf.loc[(est, c)])
            w = vd.loc[(est, c, FULLNAME[mdl])]
            add("L2 ratio", f"{est} | {c} | {mdl}", out["ratio"], w.ratio, 6e-5)
            add("L2 verdict", f"{est} | {c} | {mdl}", out["verdict"], w.verdict, 0, text=True)
g, se, df = gaps[(FULLNAME["deepseek-chat-v3"], "Low minus High SES")]
o = r3_verdict(g, se, df, rf.loc[("standardised", "Low minus High SES")])
add("L2 headline", "DeepSeek-V3 standardised Low minus High ratio (published 1.91)", round(o["ratio"], 2),
    1.91, 0.0)

# ------------------------------------------------------------------------------------ L1
nscore = pd.read_csv(os.path.join(BRM, "l1_scores_nhanes.csv"))
n0 = nscore[nscore.reference == "2005_2018"].reset_index(drop=True)
key_n = (n0[DPQ].to_numpy() * 4 ** np.arange(8)).sum(axis=1)
key_c = (corpus[ITEMS].to_numpy() * 4 ** np.arange(8)).sum(axis=1)


def collision(keys, persona, totals):
    out = {}
    df = pd.DataFrame(dict(k=keys, p=persona, t=totals))
    for t, g in df.groupby("t"):
        n = len(g)
        if n < 2:
            continue
        nk = g.k.value_counts().to_numpy(float)
        npk = g.groupby(["p", "k"]).size().to_numpy(float)
        np_ = g.p.value_counts().to_numpy(float)
        denom = n ** 2 - (np_ ** 2).sum()
        if denom <= 0:
            continue
        out[t] = ((nk ** 2).sum() - (npk ** 2).sum()) / denom, n
    return out


nh_col = collision(key_n, np.arange(len(n0)), n0.total.to_numpy())
ru = pd.read_csv(os.path.join(BRM, "l1_pattern_reuse.csv"))
for mdl in MODELS:
    for fr in ("clinical", "narrative"):
        m = ((corpus.short == mdl) & (corpus.framing == fr)).to_numpy()
        cc = collision(key_c[m], corpus.profile_id.to_numpy()[m], corpus.total.to_numpy()[m])
        ts = [t for t in cc if t in nh_col]
        wts = np.array([cc[t][1] for t in ts], float)
        pm = np.average([cc[t][0] for t in ts], weights=wts)
        pn = np.average([nh_col[t][0] for t in ts], weights=wts)
        w = ru[(ru.model == SHORT[mdl]) & (ru.framing == fr)].iloc[0]
        add("L1 reuse ratio", f"{mdl} | {fr}", pm / pn, w.ratio, 5e-4)

ip = pd.read_csv(os.path.join(BRM, "l1_personfit_item_profile.csv")).set_index("source")
Xn = n0[DPQ].to_numpy(float)
num = np.zeros((25, 8))
den = np.zeros(25)
np.add.at(num, n0.total.to_numpy(), Xn * n0.w.to_numpy()[:, None])
np.add.at(den, n0.total.to_numpy(), n0.w.to_numpy())
cmean = num / np.where(den > 0, den, np.nan)[:, None]
short = SHORT
for mdl in MODELS:
    mm = (corpus.short == mdl).to_numpy()
    Xc = corpus.loc[mm, ITEMS].to_numpy(float)
    ri = cmean[corpus.total.to_numpy()[mm]]
    ok = ~np.isnan(ri).any(axis=1)
    diff = (Xc[ok] - ri[ok]).mean(axis=0)
    for j in range(8):
        add("L1 item profile", f"{mdl} | item {j + 1}", diff[j], ip.loc[short[mdl], f"item{j + 1}_minus_nhanes"], 5e-5)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "reproduction_checks.csv"), index=False)
print(res.groupby("check").agg(n=("ok", "size"), n_ok=("ok", "sum")).to_string())
print("ALL OK:", bool(res.ok.all()))
bad = res[~res.ok]
if len(bad):
    print(bad.to_string())
