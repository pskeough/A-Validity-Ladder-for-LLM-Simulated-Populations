"""Reproduction check for the Petrov, Serapio-Garcia and Rentfrow (2024) release: recompute from the
released item-level files the numbers the authors released as aggregates.

  combined_cronbach_alpha_df.csv   Cronbach's alpha per BFI domain, 4 populations (20 numbers)
  combined_validity_df.csv         convergent / discriminant correlations of domain means (10 pairs x 4)

Inputs: paper_brm/external/raw/petrov2024/*.csv. Nothing else is used. Alpha is computed on the
complete cases (listwise) of the domain's items, as the authors' per-uid frame allows; the
pairwise variant is printed as a sensitivity.
Output: out/p21_alpha_repro.csv, out/p21_validity_repro.csv, printed summary.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "C:/Research/PsychBench/UpdatedRun/paper_brm/external/raw/petrov2024"
OUT = os.path.join(HERE, "out")
POPS = ["generic_gpt35", "generic_gpt4", "silicon_gpt35", "silicon_gpt4"]


def alpha(X):
    X = np.asarray(X, float)
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(0, ddof=1).sum() / X.sum(1).var(ddof=1))


def main():
    os.makedirs(OUT, exist_ok=True)
    au = pd.read_csv(os.path.join(RAW, "combined_cronbach_alpha_df.csv"))
    va = pd.read_csv(os.path.join(RAW, "combined_validity_df.csv"))
    rows = []
    domain_means = {}
    for pop in POPS:
        d = pd.read_csv(os.path.join(RAW, f"{pop}_df.csv"))
        for (scale, dim), g in d.groupby(["scale", "dimension"]):
            wide = g.pivot(index="uid", columns="item_index", values="response_reversed")
            complete = wide.dropna()
            a_list = alpha(complete.to_numpy())
            # pairwise-deletion alpha from the covariance matrix
            C = wide.cov(min_periods=2).to_numpy()
            k = C.shape[0]
            a_pair = k / (k - 1) * (1 - np.diag(C).sum() / C.sum())
            ref = au.loc[(au.scale == scale) & (au.dimension == dim), f"{pop}_alpha"].iloc[0]
            rows.append(dict(population=pop, scale=scale, dimension=dim, n_items=k, n_complete=len(complete),
                             n_total=len(wide), alpha_listwise=a_list, alpha_pairwise=a_pair, authors_alpha=ref,
                             abs_diff_listwise=abs(a_list - ref), abs_diff_pairwise=abs(a_pair - ref)))
            domain_means[(pop, f"{scale}_{dim}")] = wide.mean(axis=1, skipna=True)
    ra = pd.DataFrame(rows)
    ra.to_csv(os.path.join(OUT, "p21_alpha_repro.csv"), index=False)
    print("ALPHA: max abs diff listwise", ra.abs_diff_listwise.max(), "pairwise", ra.abs_diff_pairwise.max())
    print("ALPHA rows matching within 0.005: listwise", int((ra.abs_diff_listwise < 0.005).sum()), "of", len(ra),
          "| pairwise", int((ra.abs_diff_pairwise < 0.005).sum()))
    print(ra[["population", "dimension", "n_complete", "n_total", "alpha_listwise", "alpha_pairwise", "authors_alpha"]].to_string())

    vrows = []
    for _, r in va.iterrows():
        for pop in POPS:
            a, b = domain_means[(pop, r.var1)], domain_means[(pop, r.var2)]
            both = pd.concat([a, b], axis=1).dropna()
            ours = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
            theirs = float(r[f"{pop}_r"])
            vrows.append(dict(population=pop, var1=r.var1, var2=r.var2, n=len(both), ours=ours, authors=theirs,
                              abs_diff=abs(ours - theirs)))
    rv = pd.DataFrame(vrows)
    rv.to_csv(os.path.join(OUT, "p21_validity_repro.csv"), index=False)
    print("VALIDITY r: max abs diff", rv.abs_diff.max(), "; rows within 0.005:", int((rv.abs_diff < 0.005).sum()),
          "of", len(rv))
    print(rv.groupby("population").abs_diff.agg(["max", "median"]))


if __name__ == "__main__":
    main()
