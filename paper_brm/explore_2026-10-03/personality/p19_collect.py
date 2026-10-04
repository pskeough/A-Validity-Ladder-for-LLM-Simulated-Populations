"""Merge the per-scale outputs of p10, read the outputs of p12 and p20, and write the summary tables
that REPORT.md quotes: out/p19_summary.md (markdown tables) and the merged out/a10_*.csv.
No computation beyond tallies and reading verdicts; every number is read from a CSV written by
p10_audit_ipip.py, p12_human_control.py or p20_petrov.py.
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SCALES = ["Extraversion", "Emotional_Stability", "Agreeableness", "Conscientiousness", "Openness"]
ABBR = {"Extraversion": "E", "Emotional_Stability": "ES", "Agreeableness": "A", "Conscientiousness": "C", "Openness": "O"}


def merge(kind):
    fs = [os.path.join(OUT, f"a10_{kind}__{s}.csv") for s in SCALES]
    d = pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)], ignore_index=True)
    d.to_csv(os.path.join(OUT, f"a10_{kind}.csv"), index=False)
    return d


def md(df, floatfmt=2):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def short(v):
    v = str(v)
    if v.startswith("fail"):
        return "fail"
    return v


def l4_overall(r1, r2):
    if r1 is False or r2 == "fail":
        return "fail"
    if r1 and r2 == "pass":
        return "pass (R4 n/a)"
    return "unresolved"


def main():
    gate, l1, l4, desc, tau = (merge(k) for k in ("gate", "l1", "l4", "desc", "l1_tau"))
    parts = []

    # ---------------------------------------------------------------- A: verdict grid
    grid = []
    for _, r in l1.iterrows():
        q = l4[(l4.scale == r.scale) & (l4.model == r.model)].iloc[0]
        g = gate[(gate.scale == r.scale) & (gate.model == r.model)].iloc[0]
        grid.append(dict(model=r.model, scale=ABBR[r.scale], gate_single_draw=bool(g.pass_single),
                         L1_tau_frozen=short(r.verdict), L1_tau_1p5=short(r.verdict_tau1p5), R1=bool(q.R1), R2=q.R2,
                         L4=l4_overall(bool(q.R1), q.R2)))
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(OUT, "a10_verdict_grid.csv"), index=False)
    tally = []
    for rung, col in [("gate (single draw within 0.25 SD)", "gate_single_draw"), ("L1 at frozen tau", "L1_tau_frozen"),
                      ("L1 at tau 1.5", "L1_tau_1p5"), ("R1", "R1"), ("R2", "R2"), ("L4 (R1 and R2; R4 n/a)", "L4")]:
        vc = grid[col].astype(str).value_counts().to_dict()
        tally.append(dict(rung=rung, **{k: vc.get(k, 0) for k in sorted(vc)}))
    parts.append("## A. IPIP-50, 8 LLMs, one persona, 10,000 draws each: tally over 40 model x scale cells\n")
    parts.append(pd.DataFrame(tally).fillna(0).to_string(index=False) + "\n")
    wide = grid.assign(cell=lambda d: d.L1_tau_frozen.str[:4] + "/" + d.R1.map({True: "R1p", False: "R1f"}) + "/R2" + d.R2.str[:1])
    parts.append("Verdict grid (L1 at frozen tau / R1 / R2): \n" + wide.pivot(index="model", columns="scale", values="cell")[[ABBR[s] for s in SCALES]].to_string() + "\n")

    # tau
    tt = tau.sort_values("resolved_departure", ascending=False).groupby("scale").head(1)[["scale", "group", "n", "resolved_departure", "point_departure"]]
    taus = l1.groupby("scale").tau.first().reset_index()
    parts.append("## A. tau per scale and the subgroup that sets it\n" + md(taus.merge(tt, on="scale"), 2) + "\n")

    # R2 numbers
    r2 = l4[["scale", "model", "R1", "ev_ratio", "load_min", "phi", "phi_lo90", "loading_rmsd", "loading_rmsd_lo90", "loading_rmsd_hi90", "R2"]]
    r2.to_csv(os.path.join(OUT, "a10_r2_table.csv"), index=False)
    best = r2.sort_values("loading_rmsd_lo90").head(6)
    parts.append("## A. Closest cells to the R2 size limit (RMSD lower 90% limit must be <= 0.10)\n" + md(best, 3) + "\n")
    parts.append("RMSD point range over 40 cells: %.3f to %.3f; median %.3f. Congruence phi range %.3f to %.3f.\n" % (
        r2.loading_rmsd.min(), r2.loading_rmsd.max(), r2.loading_rmsd.median(), r2.phi.min(), r2.phi.max()))

    # why: per model summary
    why = desc.groupby("model").agg(sd_ratio_mean=("sd_ratio", "mean"), sd_ratio_min=("sd_ratio", "min"), sd_ratio_max=("sd_ratio", "max"),
                                    mean_diff_sd=("mean_diff_sd", "mean"), item_sd_ratio=("item_sd_ratio_mean", "mean"),
                                    inter_item_r=("sim_mean_inter_item_r", "mean"), hum_inter_item_r=("hum_mean_inter_item_r", "mean"),
                                    item_profile_corr=("item_profile_corr", "mean")).reset_index()
    parts.append("## A. Per model, averaged over the five scales\n" + md(why, 2) + "\n")
    l1s = l1.groupby("model").agg(misfit_ratio=("misfit_ratio", "mean"), overfit_ratio=("overfit_ratio", "mean")).reset_index()
    parts.append("Mean L1 ratios per model (point values; band is [1/tau, tau]):\n" + md(l1s, 2) + "\n")
    # unique profiles, over 50 items (full) are in p13; here per scale (10 items)
    up = desc.groupby("model").agg(unique_profiles_10items=("sim_unique_profiles", "mean"), modal_profile_share=("sim_modal_profile_share", "mean"),
                                   share_within_two_adjacent_cats=("sim_share_all_items_in_two_adjacent_cats", "mean"),
                                   hum_share_within_two_adjacent_cats=("hum_share_all_items_in_two_adjacent_cats", "mean")).reset_index()
    parts.append("Profile concentration (10-item scale profiles, 10,000 draws per model; humans: 20,000 reference sample, "
                 "unique profiles %d to %d):\n" % (desc.hum_unique_profiles_in_ref.min(), desc.hum_unique_profiles_in_ref.max()) + md(up, 3) + "\n")

    # gate
    gs = gate.groupby("model").agg(single_draw_sd_over_ref_sd=("within_sd_over_ref_sd", "mean"), k_min=("k_min", "max"),
                                   k_rec=("k_rec", "max"), single_pass=("pass_single", "sum")).reset_index()
    parts.append("## A. Gate (one persona; draw variance only): single-draw SD over human SD, draws k needed for SE <= 0.25 SD (max over scales)\n" + md(gs, 2) + "\n")

    # level 3 overall
    l3 = desc[["scale", "model", "mean_diff_sd", "l3_overall_ci90_lo_sd", "l3_overall_ci90_hi_sd", "l3_overall_verdict_0p5sd", "l3_overall_verdict_0p2sd"]]
    vc = l3.l3_overall_verdict_0p5sd.value_counts().to_dict()
    vc2 = l3.l3_overall_verdict_0p2sd.value_counts().to_dict()
    parts.append(f"## A. Pooled level shift (not a persona grid): TOST at 0.5 SD: {vc}; at 0.2 SD: {vc2}\n")

    # control
    for f, name in (("p12_control_l1.csv", "L1"), ("p12_control_l4.csv", "L4")):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            c = pd.read_csv(p)
            if name == "L1":
                c = c[["scale", "tau_rule", "tau", "misfit_ratio", "misfit_ratio_ci90_lo", "misfit_ratio_ci90_hi", "overfit_ratio",
                       "overfit_ratio_ci90_lo", "overfit_ratio_ci90_hi", "verdict"]]
            else:
                c = c[["scale", "R1", "phi", "phi_lo90", "loading_rmsd", "loading_rmsd_lo90", "loading_rmsd_hi90", "R2"]]
            parts.append(f"## A. Positive control, 10,000 held-out humans through {name}\n" + md(c, 3) + "\n")

    # ---------------------------------------------------------------- B: Petrov
    if os.path.exists(os.path.join(OUT, "p20_l1.csv")):
        pl1 = pd.read_csv(os.path.join(OUT, "p20_l1.csv"))
        pl4 = pd.read_csv(os.path.join(OUT, "p20_l4.csv"))
        pd_ = pd.read_csv(os.path.join(OUT, "p20_desc.csv"))
        g = pl1[["scale", "population", "model", "n", "tau", "misfit_ratio", "misfit_ratio_ci90_lo", "misfit_ratio_ci90_hi", "overfit_ratio",
                 "overfit_ratio_ci90_lo", "overfit_ratio_ci90_hi", "verdict", "verdict_tau1p5"]]
        g = g.assign(verdict=g.verdict.map(short), verdict_tau1p5=g.verdict_tau1p5.map(short))
        parts.append("## B. Petrov et al.: level 1\n" + md(g, 2) + "\n")
        q = pl4[["scale", "population", "model", "n", "R1", "ev_ratio", "load_min", "phi", "phi_lo90", "loading_rmsd", "loading_rmsd_lo90",
                 "loading_rmsd_hi90", "R2", "R3", "human_ev_ratio", "human_load_min"]]
        parts.append("## B. Petrov et al.: level 4\n" + md(q, 3) + "\n")
        parts.append("## B. Petrov et al.: description\n" + md(pd_[["scale", "population", "model", "n", "sd_ratio", "mean_diff_sd", "item_sd_ratio_mean",
                                                                  "item_profile_corr", "sim_mean_inter_item_r", "hum_mean_inter_item_r", "sim_unique_profiles", "hum_unique_profiles"]], 3) + "\n")
        for k, nm in (("p20_l3.csv", "level 3 (twin-matched residual)"), ("p20_l2.csv", "level 2 (twin-matched gaps)")):
            p = os.path.join(OUT, k)
            if os.path.exists(p):
                d = pd.read_csv(p)
                parts.append(f"## B. Petrov et al.: {nm}\n" + md(d, 2) + "\n")
    # ---------------------------------------------------------------- B: tallies of levels 3 and 2, profiles
    p3 = os.path.join(OUT, "p20_l3.csv")
    if os.path.exists(p3):
        d3 = pd.read_csv(p3)
        rows = []
        for (model, scale), g in d3.groupby(["model", "scale"]):
            row = dict(model=model, scale=scale)
            for col, nm in (("verdict_0p5sd", "L3_0.5SD"), ("verdict_0p2sd", "L3_0.2SD")):
                v = g[col]
                row[nm] = "pass" if (v == "pass").all() else ("fail" if (v == "fail").any() else "unresolved")
                row[nm + "_rows_pass"] = f"{int((v == 'pass').sum())}/{len(v)}"
            rows.append(row)
        t3 = pd.DataFrame(rows)
        t3.to_csv(os.path.join(OUT, "p20_l3_domain_verdicts.csv"), index=False)
        parts.append("## B. Silicon level 3 by domain (pass = every group row inside the band)\n" + md(t3, 2) + "\n")
    p2 = os.path.join(OUT, "p20_l2.csv")
    if os.path.exists(p2):
        d2 = pd.read_csv(p2)
        t2 = d2.groupby(["model", "label"]).size().unstack(fill_value=0).reset_index()
        parts.append("## B. Silicon level 2 labels per model (10 contrasts each)\n" + md(t2, 0) + "\n")
        t2b = d2.groupby(["model", "verdict"]).size().unstack(fill_value=0).reset_index()
        parts.append("Verdicts:\n" + md(t2b, 0) + "\n")
    pp = os.path.join(OUT, "p13_profiles.csv")
    if os.path.exists(pp):
        parts.append("## A. Whole-vector concentration (50 items)\n" + md(pd.read_csv(pp), 3) + "\n")
    text = "\n".join(parts)
    open(os.path.join(OUT, "p19_summary.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
