"""Test D. Item profile: which items carry the distortion.

D1  Item means at matched total. For each model (and framing) the mean over draws of
    x_j - m_NHANES,j(total), where m is the NHANES weighted item mean among respondents with the same
    total (79c's construction). The eight differences sum to zero by construction, so the distortion is a
    reallocation of load between items; half the sum of absolute differences is the points of load moved
    per draw. Shares of the sum of squares are reported per item.
D2  Item share of the level-3 residual. The post-stratified overall residual of the total is the sum of
    eight item residuals, sum_c p_c (s_cj - y_cj), computed for Overall, Low and High income.
D3  Item share of a group gap. For Low minus High income, Women minus Men and Middle minus High, the
    equal-weight standardised gap (matched on the other three attributes, the level-2 estimand) is
    split by item, simulated against NHANES, with the item ratio.

Reproduction checks printed at the end: item sums equal the total-score quantities that the audit
already prints (L2 standardised NHANES Low minus High 1.2974, and the simulated gaps of 78b).

Outputs: D_item_profile_matched_total.csv, D_item_profile_summary.csv, D_item_level_residual.csv,
D_item_gaps.csv, reproduction_checks_D.csv
"""
import os

import numpy as np
import pandas as pd

from common import (BRM, DPQ, FRAMINGS, FULLNAME, ITEM_NAMES, ITEMS, MODELS, OUT, SHORT, CellRef, attr, cis_matched,
                    l2_reference, l2_sim_gaps, load_corpus, load_nhanes, sim_cells)

nh = load_nhanes()
corpus = load_corpus()
sims = sim_cells(cis_matched(corpus))
nscore = nh  # nh already carries items and w7

# ---------------------------------------------------------------- D1 matched-total item means
# 79c uses l1_scores_nhanes weights ('w' of that file = w7) on 36,274 adults
Xn = nh[DPQ].to_numpy(float)
num = np.zeros((25, 8))
den = np.zeros(25)
np.add.at(num, nh.phq8.to_numpy(int), Xn * nh.w7.to_numpy()[:, None])
np.add.at(den, nh.phq8.to_numpy(int), nh.w7.to_numpy())
cmean = num / np.where(den > 0, den, np.nan)[:, None]

rows = []
for mdl in MODELS + ["pooled"]:
    for fr in FRAMINGS + ["both"]:
        m = np.ones(len(corpus), bool) if mdl == "pooled" else (corpus.short == mdl).to_numpy()
        if fr != "both":
            m = m & (corpus.framing == fr).to_numpy()
        Xc = corpus.loc[m, ITEMS].to_numpy(float)
        ri = cmean[corpus.total.to_numpy()[m]]
        ok = ~np.isnan(ri).any(axis=1)
        diff = (Xc[ok] - ri[ok]).mean(axis=0)
        sq = diff ** 2
        for j in range(8):
            rows.append(dict(model=SHORT.get(mdl, mdl), framing=fr, item=ITEM_NAMES[j], diff=diff[j],
                             share_of_squares=sq[j] / sq.sum(), load_moved_per_draw=np.abs(diff).sum() / 2,
                             n_draws=int(ok.sum())))
prof = pd.DataFrame(rows)
prof.to_csv(os.path.join(OUT, "D_item_profile_matched_total.csv"), index=False)
wide = prof[prof.framing == "both"].pivot(index="item", columns="model", values="diff")
sq = prof[prof.framing == "both"].pivot(index="item", columns="model", values="share_of_squares")
mv = prof[prof.framing == "both"].groupby("model").load_moved_per_draw.first()
wide.to_csv(os.path.join(OUT, "D_item_profile_summary.csv"))

# ---------------------------------------------------------------- D2 item share of the level-3 residual
crs = [CellRef(nh, ycol=c) for c in DPQ]
Nf = crs[0].Nf
y_items = np.column_stack([c.mf for c in crs])          # 48 x 8 NHANES cell item means
res_rows = []
for mdl in MODELS + ["pooled"]:
    S = sims[(mdl, "combined")]["items"]                # 48 x 8
    for gname, dim, val in [("Overall", None, None), ("Low", 2, "Low"), ("Middle", 2, "Middle"),
                            ("High", 2, "High")]:
        g = np.ones(48, bool) if dim is None else attr(dim) == val
        p = Nf[g] / Nf[g].sum()
        item_res = p @ (S[g] - y_items[g])
        for j in range(8):
            res_rows.append(dict(model=SHORT.get(mdl, mdl), group=gname, item=ITEM_NAMES[j],
                                 item_residual=item_res[j], share_of_total_residual=item_res[j] / item_res.sum(),
                                 total_residual=item_res.sum()))
pd.DataFrame(res_rows).to_csv(os.path.join(OUT, "D_item_level_residual.csv"), index=False)

# ---------------------------------------------------------------- D3 item share of group gaps
def gap_vec(M, dim, hi, lo):
    """Equal-weight standardised gap per column of M (48 x k): mean over strata of matched differences."""
    a = attr(dim)
    others = [k for k in range(4) if k != dim]
    key = np.array(["|".join(CELLS_[c][k] for k in others) for c in range(48)])
    out = []
    for st in np.unique(key[(a == hi) | (a == lo)]):
        ih = np.flatnonzero((key == st) & (a == hi))
        il = np.flatnonzero((key == st) & (a == lo))
        assert len(ih) == 1 and len(il) == 1
        out.append(M[ih[0]] - M[il[0]])
    return np.mean(out, axis=0)


from common import CELLS as CELLS_  # noqa: E402

CONTR = [("Low minus High SES", 2, "Low", "High"), ("Women minus Men", 1, "Women", "Men"),
         ("Middle minus High SES", 2, "Middle", "High"), ("Black minus White", 0, "Black", "White")]
gap_rows = []
nh_gap = {}
for name, dim, hi, lo in CONTR:
    nh_gap[name] = gap_vec(y_items, dim, hi, lo)
    for j in range(8):
        gap_rows.append(dict(source="NHANES", contrast=name, item=ITEM_NAMES[j], gap=nh_gap[name][j],
                             share_of_total_gap=nh_gap[name][j] / nh_gap[name].sum(), ratio_to_nhanes=1.0,
                             total_gap=nh_gap[name].sum()))
sim_gap = {}
for mdl in MODELS:
    S = sims[(mdl, "combined")]["items"]
    for name, dim, hi, lo in CONTR:
        gv = gap_vec(S, dim, hi, lo)
        sim_gap[(mdl, name)] = gv
        for j in range(8):
            gap_rows.append(dict(source=SHORT[mdl], contrast=name, item=ITEM_NAMES[j], gap=gv[j],
                                 share_of_total_gap=gv[j] / gv.sum(), ratio_to_nhanes=gv[j] / nh_gap[name][j],
                                 total_gap=gv.sum()))
pd.DataFrame(gap_rows).to_csv(os.path.join(OUT, "D_item_gaps.csv"), index=False)

# ---------------------------------------------------------------- reproduction checks
chk = []
rf = l2_reference()
chk.append(dict(check="NHANES Low minus High, sum of item gaps vs l2_reference standardised",
                got=nh_gap["Low minus High SES"].sum(), want=rf.loc[("standardised", "Low minus High SES")].estimate))
chk.append(dict(check="NHANES Women minus Men, sum of item gaps vs l2_reference standardised",
                got=nh_gap["Women minus Men"].sum(), want=rf.loc[("standardised", "Women minus Men")].estimate))
gaps = l2_sim_gaps()
for mdl in MODELS:
    for name in ("Low minus High SES", "Women minus Men"):
        chk.append(dict(check=f"{SHORT[mdl]} {name}, sum of item gaps vs 78b persona-pair gap",
                        got=sim_gap[(mdl, name)].sum(), want=gaps[(FULLNAME[mdl], name)][0]))
    S = sims[(mdl, "combined")]["items"]
    p = Nf / Nf.sum()
    chk.append(dict(check=f"{SHORT[mdl]} overall PS residual, sum of item residuals vs 80c",
                    got=float((p @ (S - y_items)).sum()),
                    want=float(pd.read_csv(os.path.join(BRM, 
                                                         "80c_l3_results.csv"), low_memory=False)
                               .query("spec == 'S1 primary' and corpus == 'all' and outcome == 'mean' and "
                                      "framing == 'combined' and estimand == 'PS' and group == 'Overall' and "
                                      "model == @mdl").resid.iloc[0])))
chk = pd.DataFrame(chk)
chk["ok"] = (chk.got - chk.want).abs() < 2e-4
chk.to_csv(os.path.join(OUT, "reproduction_checks_D.csv"), index=False)

pd.set_option("display.width", 250)
print(chk.round(4).to_string(index=False))
print("\nD1 item minus NHANES at matched total (both framings)")
print(wide.round(3).to_string())
print("share of squares"); print(sq.round(3).to_string())
print("load moved per draw (points)"); print(mv.round(3).to_string())
r2 = pd.DataFrame(res_rows)
print("\nD2 share of overall residual")
print(r2[r2.group == "Overall"].pivot(index="item", columns="model", values="share_of_total_residual").round(3).to_string())
print(r2[r2.group == "Overall"].groupby("model").total_residual.first().round(3).to_string())
g = pd.DataFrame(gap_rows)
for name in [c[0] for c in CONTR]:
    q = g[g.contrast == name]
    print(f"\nD3 {name}: gap per item")
    print(q.pivot(index="item", columns="source", values="gap").round(3).to_string())
    print(q.pivot(index="item", columns="source", values="share_of_total_gap").round(3).to_string())
