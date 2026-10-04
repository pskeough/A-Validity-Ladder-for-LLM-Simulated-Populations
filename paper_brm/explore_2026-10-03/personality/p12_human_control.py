"""Positive control for p10: 10,000 real respondents, disjoint from the 20,000-person reference sample,
are passed through level 1 and level 4 exactly as a simulated sample is (same reference, same tau, same
bootstrap sizes, each person her own cluster). A rung that fails these real people cannot support a
verdict about a model. The reference steps replicate p10 line for line so the reference is identical
(same seed sequence); tau uses the same country groups.

Output: out/p12_control_l1.csv, out/p12_control_l4.csv
"""
import importlib.util
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "C:/Research/PsychBench/UpdatedRun"
RAW = os.path.join(BASE, "paper_brm", "external", "raw", "ipip_audit")
OUT = os.path.join(HERE, "out")
SEED = 20261003
N_REF, N_CTRL = 20000, 10000
B_L1, B_L4 = 2000, 500
PREFIX = {"Extraversion": "EXT", "Emotional_Stability": "EST", "Agreeableness": "AGR",
          "Conscientiousness": "CSN", "Openness": "OPN"}
REVERSE = {"Extraversion": [2, 4, 6, 8, 10], "Emotional_Stability": [1, 3, 5, 6, 7, 8, 9, 10],
           "Agreeableness": [1, 3, 5, 7], "Conscientiousness": [2, 4, 6, 8], "Openness": [2, 4, 6]}


def load_core():
    spec = importlib.util.spec_from_file_location("core91", os.path.join(BASE, "scripts", "91_ladder_core.py"))
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = [old[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


V = load_core()


def keyed(df, scale):
    out = []
    for i in range(1, 11):
        x = df[f"{PREFIX[scale]}{i}"].to_numpy(int)
        out.append(5 - x if i in REVERSE[scale] else x - 1)
    return np.column_stack(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    item_cols = [f"{p}{i}" for p in PREFIX.values() for i in range(1, 11)]
    hum = pd.read_csv(os.path.join(RAW, "human_data_cleaned.csv"), usecols=item_cols + ["country"])
    hum = hum[hum[item_cols].notna().all(axis=1) & hum[item_cols].isin([1, 2, 3, 4, 5]).all(axis=1)].reset_index(drop=True)
    hum[item_cols] = hum[item_cols].astype(int)
    idx = np.random.default_rng(SEED).choice(len(hum), N_REF, replace=False)
    rest = np.setdiff1d(np.arange(len(hum)), idx)
    cidx = np.random.default_rng(SEED + 7).choice(rest, N_CTRL, replace=False)
    l1_rows, l4_rows = [], []
    for si, scale in enumerate(PREFIX):
        t0 = time.time()
        rng = np.random.default_rng(SEED + 100 * si)
        J, M = 10, 5
        Xall = keyed(hum, scale)
        Xh, Xc = Xall[idx], Xall[cidx]
        th, tc = Xh.sum(1), Xc.sum(1)
        country = hum.country.to_numpy()[idx]
        groups = {f"country={v}": country == v for v in pd.Series(country).value_counts().index
                  if isinstance(v, str) and (country == v).sum() >= 100}
        grm = V.GRM(J, M)
        fit = grm.fit(Xh)
        _, _, lz_h = grm.score(fit["a"], fit["b"], Xh)
        ref1 = V.L1Ref(th, lz_h, None, J * (M - 1), B_L1, rng)
        tau, worst, _ = V.tau_from_subgroups(ref1, th, lz_h, groups)
        ref4 = V.L4Ref(Xh, None, M, B_L4, rng)
        rng_c = np.random.default_rng(SEED + 100 * si + 50)
        _, _, lz_c = grm.score(fit["a"], fit["b"], Xc)
        per = np.arange(len(Xc))
        for tau_use, label in ((tau, "frozen"), (1.5, "tau1.5")):
            r1 = V.l1_eval(ref1, tc, lz_c, per, np.random.default_rng(SEED + 100 * si + 51), tau=tau_use)
            l1_rows.append(dict(scale=scale, tau_rule=label, **r1))
        r4 = V.l4_eval(Xc, per, ref4, rng_c)
        lam = r4.pop("lam")
        l4_rows.append(dict(scale=scale, n=len(Xc), **r4, lam=" ".join(f"{x:.2f}" for x in lam),
                            human_lam=" ".join(f"{x:.2f}" for x in ref4.lam)))
        r1f = l1_rows[-2]
        print(f"[{scale}] control: tau {tau} L1 mis {r1f['misfit_ratio']:.2f} [{r1f['misfit_ratio_ci90_lo']:.2f},"
              f"{r1f['misfit_ratio_ci90_hi']:.2f}] ov {r1f['overfit_ratio']:.2f} [{r1f['overfit_ratio_ci90_lo']:.2f},"
              f"{r1f['overfit_ratio_ci90_hi']:.2f}] {r1f['verdict']} | at 1.5: {l1_rows[-1]['verdict']} | "
              f"R1 {r4['R1']} R2 {r4['R2']} (phi lo {r4['phi_lo90']:.3f}, rmsd hi {r4['loading_rmsd_hi90']:.3f}) "
              f"({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(l1_rows).to_csv(os.path.join(OUT, "p12_control_l1.csv"), index=False)
    pd.DataFrame(l4_rows).to_csv(os.path.join(OUT, "p12_control_l4.csv"), index=False)


if __name__ == "__main__":
    main()
