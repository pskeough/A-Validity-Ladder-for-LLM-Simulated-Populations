"""Per-model tallies of the ladder rungs for the ipip_audit release and per-population tallies for the
Petrov release, read from the merged CSVs. Output: out/p15_model_tables.md (printed too)."""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    g = pd.read_csv(os.path.join(OUT, "a10_gate.csv"))
    l1 = pd.read_csv(os.path.join(OUT, "a10_l1.csv"))
    l4 = pd.read_csv(os.path.join(OUT, "a10_l4.csv"))
    d = pd.read_csv(os.path.join(OUT, "a10_desc.csv"))
    rows = []
    for m in sorted(l1.model.unique()):
        a, b, c, e = g[g.model == m], l1[l1.model == m], l4[l4.model == m], d[d.model == m]
        cat = b.verdict.str.replace("fail: ", "").value_counts().to_dict()
        rows.append({
            "model": m,
            "gate k=10 min / rec": f"{int(a.pass_min.sum())}/5, {int(a.pass_rec.sum())}/5",
            "k for 0.25 SD (max)": int(a.k_min.max()),
            "SD ratio range": f"{e.sd_ratio.min():.2f} to {e.sd_ratio.max():.2f}",
            "L1 pass": f"{int((b.verdict == 'pass').sum())}/5",
            "misfit ratio max": round(b.misfit_ratio.max(), 2),
            "overfit ratio range": f"{b.overfit_ratio.min():.2f} to {b.overfit_ratio.max():.2f}",
            "R1 pass": f"{int(c.R1.sum())}/5",
            "R2 pass": f"{int((c.R2 == 'pass').sum())}/5",
            "phi lower limit range": f"{c.phi_lo90.min():.3f} to {c.phi_lo90.max():.3f}",
            "RMSD range": f"{c.loading_rmsd.min():.3f} to {c.loading_rmsd.max():.3f}",
        })
    t = pd.DataFrame(rows)
    out = ["## A per model\n", md(t), ""]
    # L1 verdict category counts
    cat = l1.assign(cat=l1.verdict.str.replace("fail: ", "")).groupby(["model", "cat"]).size().unstack(fill_value=0).reset_index()
    out += ["\n## A L1 verdict categories per model (counts of 5 scales)\n", md(cat), ""]
    # B
    p1 = pd.read_csv(os.path.join(OUT, "p20_l1.csv"))
    p4 = pd.read_csv(os.path.join(OUT, "p20_l4.csv"))
    pdsc = pd.read_csv(os.path.join(OUT, "p20_desc.csv"))
    rows = []
    for (pop, m), a in p1.groupby(["population", "model"]):
        c = p4[(p4.population == pop) & (p4.model == m)]
        e = pdsc[(pdsc.population == pop) & (pdsc.model == m)]
        rows.append({"population": pop, "model": m, "n range": f"{a.n.min()} to {a.n.max()}",
                     "L1 pass / undetermined / fail": f"{int((a.verdict == 'pass').sum())}/{int((a.verdict == 'undetermined').sum())}/{int(a.verdict.str.startswith('fail').sum())}",
                     "R1 pass": f"{int(c.R1.sum())}/5", "R2 pass / unresolved / fail": f"{int((c.R2 == 'pass').sum())}/{int((c.R2 == 'unresolved').sum())}/{int((c.R2 == 'fail').sum())}",
                     "SD ratio range": f"{e.sd_ratio.min():.2f} to {e.sd_ratio.max():.2f}",
                     "RMSD range": f"{c.loading_rmsd.min():.3f} to {c.loading_rmsd.max():.3f}",
                     "mean inter-item r (sim vs hum)": f"{e.sim_mean_inter_item_r.mean():.2f} vs {e.hum_mean_inter_item_r.mean():.2f}"})
    out += ["\n## B per population and model\n", md(pd.DataFrame(rows)), ""]
    text = "\n".join(out)
    open(os.path.join(OUT, "p15_model_tables.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
