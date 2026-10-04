"""Positive control at the size of Petrov's generic persona sets (about 145 persons).

Question: can level 1 and R2 return "pass" at all for a sample of about 145 real respondents? The
Twin-2K reference (2,058) is split: N_CTRL = 145 respondents are held out as the "simulated sample"
(each person her own cluster, as in p20) and the remaining respondents are the reference; the same GRM,
L1Ref, tau rule and L4Ref are built on the reference alone. R = 5 random hold-out draws. If real people
at this size come back "unresolved" the generic sets cannot pass, whatever the model does.

Output: out/p22_small_n_control.csv, printed table.  Seed 20261022.  B_L1 = 1,000, B_L4 = 300 (smaller
than p20 to keep the run short; the verdict rule is unchanged).
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
OUT = os.path.join(HERE, "out")
RAWT = os.path.join(BASE, "paper_brm", "external", "raw", "twin2k")
SEED = 20261022
N_CTRL, R, B_L1, B_L4 = 145, 5, 1000, 300
HUMAN_KEY = {
    "Extraversion": [1, -6, 11, 16, -21, 26, -31, 36],
    "Agreeableness": [-2, 7, -12, 17, 22, -27, 32, -37, 42],
    "Conscientiousness": [3, -8, 13, -18, -23, 28, 33, 38, -43],
    "Neuroticism": [4, -9, 14, 19, -24, 29, -34, 39],
    "Openness": [5, 10, 15, 20, 25, 30, -35, 40, -41, 44],
}


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


def keyed(df, items):
    return np.column_stack([(df[f"i{abs(i)}"].to_numpy(int) - 1) if i > 0 else (5 - df[f"i{abs(i)}"].to_numpy(int))
                            for i in items])


def main():
    d = pd.read_csv(os.path.join(RAWT, "wave1_3_response.csv"), low_memory=False)
    h = d[["QID12", "QID13", "QID15"] + [f"QID25_{i}" for i in range(1, 45)]].dropna()
    h = h.rename(columns={f"QID25_{i}": f"i{i}" for i in range(1, 45)}).reset_index(drop=True)
    rows = []
    for si, (scale, key) in enumerate(HUMAN_KEY.items()):
        t0 = time.time()
        X = keyed(h, key)
        J, M = X.shape[1], 5
        for rep in range(R):
            rng = np.random.default_rng(SEED + 1000 * si + rep)
            hold = rng.choice(len(X), N_CTRL, replace=False)
            mask = np.ones(len(X), bool)
            mask[hold] = False
            Xr, Xc = X[mask], X[hold]
            tr, tc = Xr.sum(1), Xc.sum(1)
            sex, age, race = h.QID12.to_numpy()[mask], h.QID13.to_numpy()[mask], h.QID15.to_numpy()[mask]
            groups = {f"sex={v}": sex == v for v in np.unique(sex)}
            groups.update({f"age={v}": age == v for v in np.unique(age)})
            groups.update({f"race={v}": race == v for v in np.unique(race)})
            groups = {k: m for k, m in groups.items() if m.sum() >= 100}
            grm = V.GRM(J, M)
            fit = grm.fit(Xr)
            _, _, lz_r = grm.score(fit["a"], fit["b"], Xr)
            ref1 = V.L1Ref(tr, lz_r, None, J * (M - 1), B_L1, rng)
            tau, worst, _ = V.tau_from_subgroups(ref1, tr, lz_r, groups)
            ref4 = V.L4Ref(Xr, None, M, B_L4, rng)
            _, _, lz_c = grm.score(fit["a"], fit["b"], Xc)
            per = np.arange(len(Xc))
            r1 = V.l1_eval(ref1, tc, lz_c, per, rng, tau=tau)
            r4 = V.l4_eval(Xc, per, ref4, rng)
            rows.append(dict(scale=scale, rep=rep, n=len(Xc), tau=tau, misfit_ratio=r1["misfit_ratio"],
                             misfit_lo=r1["misfit_ratio_ci90_lo"], misfit_hi=r1["misfit_ratio_ci90_hi"],
                             overfit_ratio=r1["overfit_ratio"], overfit_lo=r1["overfit_ratio_ci90_lo"],
                             overfit_hi=r1["overfit_ratio_ci90_hi"], L1=r1["verdict"].split(":")[0], R1=r4["R1"],
                             phi_lo90=r4["phi_lo90"], rmsd=r4["loading_rmsd"], rmsd_lo90=r4["loading_rmsd_lo90"],
                             rmsd_hi90=r4["loading_rmsd_hi90"], R2=r4["R2"]))
            print(f"[{scale} rep {rep}] L1 {rows[-1]['L1']} (mis {r1['misfit_ratio']:.2f} [{r1['misfit_ratio_ci90_lo']:.2f},"
                  f"{r1['misfit_ratio_ci90_hi']:.2f}] ov {r1['overfit_ratio']:.2f} [{r1['overfit_ratio_ci90_lo']:.2f},"
                  f"{r1['overfit_ratio_ci90_hi']:.2f}]) R1 {r4['R1']} R2 {r4['R2']} (rmsd {r4['loading_rmsd']:.3f} "
                  f"[{r4['loading_rmsd_lo90']:.3f},{r4['loading_rmsd_hi90']:.3f}])", flush=True)
        print(f"[{scale}] {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "p22_small_n_control.csv"), index=False)
    print(out.groupby("scale").agg(L1_pass=("L1", lambda s: int((s == "pass").sum())), R2_pass=("R2", lambda s: int((s == "pass").sum())),
                                   rmsd_hi_median=("rmsd_hi90", "median")).to_string())
    print("L1 verdicts:", out.L1.value_counts().to_dict(), "| R2 verdicts:", out.R2.value_counts().to_dict())


if __name__ == "__main__":
    main()
