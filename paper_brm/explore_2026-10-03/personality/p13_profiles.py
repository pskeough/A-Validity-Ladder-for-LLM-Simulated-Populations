"""Concentration of whole 50-item answer vectors and of single items, per model, against the human sample.
Reads big5_llm.csv and a 20,000-person human sample (same seed as p10). Raw scores 1-5 (no keying needed:
uniqueness and modal share do not depend on it).

Output: out/p13_profiles.csv
  unique_vectors      distinct 50-item vectors among the draws
  modal_vector_share  share of draws that equal the single most common vector
  mean_item_modal_share  mean over items of the share of the item's most common response
  mean_item_entropy_bits mean Shannon entropy of an item's response distribution (max log2 5 = 2.32)
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "C:/Research/PsychBench/UpdatedRun/paper_brm/external/raw/ipip_audit"
OUT = os.path.join(HERE, "out")
SEED = 20261003
N_REF = 20000
PREFIX = ["EXT", "EST", "AGR", "CSN", "OPN"]


def stats(X):
    n = len(X)
    rows, counts = np.unique(X, axis=0, return_counts=True)
    ent, modal = [], []
    for j in range(X.shape[1]):
        p = np.bincount(X[:, j], minlength=6)[1:] / n
        modal.append(p.max())
        q = p[p > 0]
        ent.append(float(-(q * np.log2(q)).sum()))
    return dict(n=n, unique_vectors=len(rows), unique_share=len(rows) / n, modal_vector_share=counts.max() / n,
                mean_item_modal_share=float(np.mean(modal)), mean_item_entropy_bits=float(np.mean(ent)))


def main():
    os.makedirs(OUT, exist_ok=True)
    item_cols = [f"{p}{i}" for p in PREFIX for i in range(1, 11)]
    hum = pd.read_csv(os.path.join(RAW, "human_data_cleaned.csv"), usecols=item_cols)
    hum = hum[hum[item_cols].notna().all(axis=1)].reset_index(drop=True)
    idx = np.random.default_rng(SEED).choice(len(hum), N_REF, replace=False)
    sim = pd.read_csv(os.path.join(RAW, "big5_llm.csv"), encoding="utf-8-sig")
    rows = [dict(group="humans (20,000 sample)", **stats(hum.loc[idx, item_cols].to_numpy(int)))]
    for m, g in sim.groupby("model"):
        rows.append(dict(group=m, **stats(g[item_cols].to_numpy(int))))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "p13_profiles.csv"), index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
