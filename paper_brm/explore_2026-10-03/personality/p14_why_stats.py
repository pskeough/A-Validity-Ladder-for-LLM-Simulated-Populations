"""Descriptive statistics for the "why" section, computed from the stored loadings (a10_l4.csv,
p20_l4.csv: columns lam and human_lam), the description tables (a10_desc.csv, p20_desc.csv) and the
level-1 tables. No new inference; every quantity is a function of numbers already written by p10 and p20.

For each cell:
  mean_lam_sim, mean_lam_hum   mean one-factor loading of the simulated and human data
  rmsd                         RMS loading difference (as stored)
  bias_share                   (mean difference)^2 / rmsd^2: share of the squared RMSD that is a uniform shift
  profile_r                    correlation of the item loadings across simulated and human data
  sim_larger                   True when the mean simulated loading exceeds the human one
Then rank correlations of sd_ratio with rmsd and with overfit ratio across the cells.

Output: out/p14_why.csv and a printed summary (out/p14_run.log).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def parse(s):
    return np.array([float(x) for x in s.split()])


def cell_rows(l4, desc, l1, keys):
    rows = []
    for _, r in l4.iterrows():
        a, b = parse(r.lam), parse(r.human_lam)
        d = a - b
        rm = float(np.sqrt(np.mean(d ** 2)))
        mask = (desc[keys[0]] == r[keys[0]])
        for k in keys[1:]:
            mask &= desc[k] == r[k]
        ds = desc[mask].iloc[0]
        m1 = (l1[keys[0]] == r[keys[0]])
        for k in keys[1:]:
            m1 &= l1[k] == r[k]
        o = l1[m1].iloc[0]
        rows.append(dict(**{k: r[k] for k in keys}, mean_lam_sim=a.mean(), mean_lam_hum=b.mean(), rmsd=rm,
                         bias_share=float(d.mean() ** 2 / rm ** 2) if rm > 0 else np.nan,
                         profile_r=float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan,
                         sim_larger=bool(a.mean() > b.mean()), R1=r.R1, R2=r.R2, sd_ratio=ds.sd_ratio,
                         overfit_ratio=o.overfit_ratio, misfit_ratio=o.misfit_ratio, mean_diff_sd=ds.mean_diff_sd))
    return pd.DataFrame(rows)


def report(name, d):
    ok = d.dropna(subset=["profile_r"])
    r1 = d[d.R1 == True]
    print(f"== {name}: {len(d)} cells; R1-pass cells {len(r1)}")
    print(f"  mean simulated loading larger than humans' in {int(d.sim_larger.sum())} of {len(d)} cells; "
          f"in R1-pass cells {int(r1.sim_larger.sum())} of {len(r1)}")
    print(f"  mean loading sim {d.mean_lam_sim.median():.2f} (median) vs human {d.mean_lam_hum.median():.2f}")
    print(f"  bias share of squared RMSD: median {d.bias_share.median():.2f} (R1-pass cells {r1.bias_share.median():.2f})")
    print(f"  profile correlation of item loadings: median {ok.profile_r.median():.2f}; "
          f"R1-pass cells {r1.profile_r.median():.2f}")
    for a, b in (("sd_ratio", "rmsd"), ("sd_ratio", "overfit_ratio"), ("mean_diff_sd", "rmsd")):
        x = d[[a, b]].dropna()
        rho, p = stats.spearmanr(x[a], x[b])
        print(f"  Spearman {a} vs {b}: rho {rho:.2f} (p {p:.3g}, n {len(x)})")


def main():
    l4a = pd.read_csv(os.path.join(OUT, "a10_l4.csv"))
    da = pd.read_csv(os.path.join(OUT, "a10_desc.csv"))
    l1a = pd.read_csv(os.path.join(OUT, "a10_l1.csv"))
    A = cell_rows(l4a, da, l1a, ["scale", "model"])
    A.insert(0, "release", "ipip_audit")
    l4b = pd.read_csv(os.path.join(OUT, "p20_l4.csv"))
    db = pd.read_csv(os.path.join(OUT, "p20_desc.csv"))
    l1b = pd.read_csv(os.path.join(OUT, "p20_l1.csv"))
    B = cell_rows(l4b, db, l1b, ["scale", "population", "model"])
    B.insert(0, "release", "petrov2024")
    pd.concat([A, B], ignore_index=True).to_csv(os.path.join(OUT, "p14_why.csv"), index=False)
    report("A ipip_audit", A)
    report("B petrov2024 (all four populations)", B)
    for pop in ("generic", "silicon"):
        report(f"B petrov2024 {pop}", B[B.population == pop])
    # by model A
    print(A.groupby("model").agg(sd_ratio=("sd_ratio", "mean"), rmsd=("rmsd", "mean"), mean_lam_sim=("mean_lam_sim", "mean"),
                                 overfit=("overfit_ratio", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
