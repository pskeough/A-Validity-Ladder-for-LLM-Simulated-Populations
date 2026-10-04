"""Validity ladder on the "LLM Psychometric Fidelity Audit" release: IPIP Big Five markers (50 items,
5 categories) answered 10,000 times by each of 8 current LLMs, against the Open Psychometrics IPIP-FFM
human sample (cleaned by the release authors).

Data (downloaded 2026-10-03, raw files in paper_brm/external/raw/ipip_audit, SOURCE.txt there):
  big5_llm.csv            80,000 rows = 8 models x 10,000 draws, items EXT1..OPN10, scored 1-5
  human_data_cleaned.csv  500,703 human respondents, same items, plus country and timing columns

Design facts that fix how the ladder is applied (read the REPORT):
  * Every draw is one call with the SAME prompt ("You are a human participant completing an online
    personality survey ...", temperature 1.0). There is one persona. Each draw is therefore treated as
    its own cluster (persona = draw index) in the level-1 and level-4 bootstraps, which is the same as
    an iid-person bootstrap of the simulated sample.
  * The gate needs between-persona variance and a persona mean. With one persona only the draw
    variance s2_r is estimable. The gate row reports s2_r, the single-draw SD against the human SD, and
    the number of draws k needed for SE(k) <= 0.25 / 0.125 reference SD. phi(k) is not defined.
  * The personas carry no attribute, so level 2 and R4 are not applicable. Level 3 has one group (all);
    its TOST on the pooled mean is reported as a description of the level shift, not as a persona grid.

Reference: a random sample of N_REF humans (seed below) is used for the GRM, the within-total
l_z* reference distribution and the loadings; all 500,703 are used for descriptive means and SDs
(reproduction check against the authors' notebook). Reference subgroups for tau: country (the only
demographic column), levels with n >= 100 in the sample.

Keying: the release authors' MANUAL_REVERSE_KEY (every item scored so that high = the trait named in
the authors' notebook: Extraversion, Emotional_Stability, Agreeableness, Conscientiousness, Openness).
Scored 0-4.

Run:  python p10_audit_ipip.py --scales Extraversion Emotional_Stability ...   (one process)
Output: out/a10_<kind>__<scale>.csv, merged by p19_collect.py. Seed base 20261003.
"""
import argparse
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
N_REF = 20000
B_L1, B_L4 = 2000, 500

PREFIX = {"Extraversion": "EXT", "Emotional_Stability": "EST", "Agreeableness": "AGR",
          "Conscientiousness": "CSN", "Openness": "OPN"}
REVERSE = {
    "Extraversion": [2, 4, 6, 8, 10],
    "Emotional_Stability": [1, 3, 5, 6, 7, 8, 9, 10],
    "Agreeableness": [1, 3, 5, 7],
    "Conscientiousness": [2, 4, 6, 8],
    "Openness": [2, 4, 6],
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


def cols(scale):
    return [f"{PREFIX[scale]}{i}" for i in range(1, 11)]


def keyed(df, scale):
    """Items scored 0-4, high = the trait named in the authors' key."""
    out = []
    for i in range(1, 11):
        x = df[f"{PREFIX[scale]}{i}"].to_numpy(int)
        out.append(5 - x if i in REVERSE[scale] else x - 1)
    return np.column_stack(out)


def load_data():
    item_cols = [f"{p}{i}" for p in PREFIX.values() for i in range(1, 11)]
    t0 = time.time()
    hum = pd.read_csv(os.path.join(RAW, "human_data_cleaned.csv"), usecols=item_cols + ["country"])
    ok = hum[item_cols].notna().all(axis=1) & hum[item_cols].isin([1, 2, 3, 4, 5]).all(axis=1)
    print(f"humans read {len(hum)} rows in {time.time() - t0:.0f}s; complete and in range {int(ok.sum())}", flush=True)
    hum = hum[ok].reset_index(drop=True)
    hum[item_cols] = hum[item_cols].astype(int)
    sim = pd.read_csv(os.path.join(RAW, "big5_llm.csv"), encoding="utf-8-sig")
    assert sim[item_cols].notna().all().all() and sim[item_cols].isin([1, 2, 3, 4, 5]).all().all()
    sim[item_cols] = sim[item_cols].astype(int)
    return hum, sim


def gate_one_cell(totals, sd_ref, k_report):
    """Single persona: only the draw variance s2_r is estimable (82_gate_lib.pos, k_for_se reused)."""
    s2r = float(np.var(totals, ddof=1))
    t_min, t_rec = V.gate_tolerances(sd_ref)
    return dict(n_personas=1, n_draws=len(totals), k=k_report, s2_r=s2r, within_sd=float(np.sqrt(s2r)),
                sd_ref=sd_ref, within_sd_over_ref_sd=float(np.sqrt(s2r) / sd_ref),
                se_k=float(np.sqrt(s2r / k_report)), tol_min=t_min, tol_rec=t_rec,
                k_min=V.GL.k_for_se(s2r, t_min), k_rec=V.GL.k_for_se(s2r, t_rec),
                pass_min=bool(np.sqrt(s2r / k_report) <= t_min), pass_rec=bool(np.sqrt(s2r / k_report) <= t_rec),
                pass_single=bool(np.sqrt(s2r) <= t_min), phi_k=np.nan)


def tost_overall(diff, sd_ref, n_sim, n_ref, var_sim_total, var_ref_total, delta_sd):
    """90% interval of (simulated mean - human mean) in score points, and the TOST reading."""
    from scipy import stats
    var = var_sim_total / n_sim + var_ref_total / n_ref
    se = float(np.sqrt(var))
    df = var ** 2 / ((var_sim_total / n_sim) ** 2 / (n_sim - 1) + (var_ref_total / n_ref) ** 2 / (n_ref - 1))
    tc = float(stats.t.ppf(0.95, df))
    lo, hi = diff - tc * se, diff + tc * se
    tol = delta_sd * sd_ref
    verdict = "pass" if (lo > -tol and hi < tol) else ("fail" if (lo > tol or hi < -tol) else "unresolved")
    return lo, hi, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=list(PREFIX))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--b1", type=int, default=B_L1)
    ap.add_argument("--b4", type=int, default=B_L4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    hum, sim = load_data()
    models = args.models or sorted(sim.model.unique())
    scale_index = {s: i for i, s in enumerate(PREFIX)}

    for scale in args.scales:
        t_scale = time.time()
        rng = np.random.default_rng(SEED + 100 * scale_index[scale])
        J, M = 10, 5
        # --- human reference
        Xall = keyed(hum, scale)
        tot_all = Xall.sum(1)
        sd_all = float(tot_all.std(ddof=1))
        idx = np.random.default_rng(SEED).choice(len(hum), N_REF, replace=False)   # same sample for every scale
        Xh = Xall[idx]
        th = Xh.sum(1)
        sd_ref = float(th.std(ddof=1))
        country = hum.country.to_numpy()[idx]
        groups = {}
        for v in pd.Series(country).value_counts().index:
            if (country == v).sum() >= 100 and isinstance(v, str):
                groups[f"country={v}"] = country == v
        grm = V.GRM(J, M)
        t0 = time.time()
        fit = grm.fit(Xh)
        print(f"[{scale}] GRM converged {fit['converged']} nit {fit['nit']} patterns {fit['n_patterns']} "
              f"({time.time() - t0:.0f}s); human SD (sample) {sd_ref:.3f}, all {sd_all:.3f}; "
              f"tau groups {len(groups)}", flush=True)
        _, _, lz_h = grm.score(fit["a"], fit["b"], Xh)
        ref1 = V.L1Ref(th, lz_h, None, J * (M - 1), args.b1, rng)
        t0 = time.time()
        tau, worst, trows = V.tau_from_subgroups(ref1, th, lz_h, groups)
        print(f"[{scale}] tau {tau} (worst resolved subgroup departure {worst:.2f}; worst point "
              f"{max(r['point_departure'] for r in trows):.2f}) ({time.time() - t0:.0f}s)", flush=True)
        t0 = time.time()
        ref4 = V.L4Ref(Xh, None, M, args.b4, rng)
        print(f"[{scale}] human R1 absolute {ref4.gf}, ev ratio {ref4.ev[0] / ref4.ev[1]:.2f}, loadings "
              f"{' '.join(f'{x:.2f}' for x in ref4.lam)} ({time.time() - t0:.0f}s)", flush=True)
        assert (ref4.lam > 0).all(), "keying error: negative human loading"
        hum_item_sd = Xh.std(0, ddof=1)
        hum_item_mean = Xh.mean(0)
        hum_rbar = float(np.corrcoef(Xh.T)[np.triu_indices(J, 1)].mean())
        hum_unique = len(np.unique(Xh, axis=0))

        g_rows, l1_rows, l4_rows, d_rows = [], [], [], []
        for model in models:
            s = sim[sim.model == model]
            Xs = keyed(s, scale)
            ts = Xs.sum(1)
            n = len(Xs)
            per = np.arange(n)
            t0 = time.time()
            g = gate_one_cell(ts, sd_ref, k_report=10)
            g_rows.append(dict(scale=scale, model=model, **g))
            _, _, lz_s = grm.score(fit["a"], fit["b"], Xs)
            rng_m = np.random.default_rng(SEED + 100 * scale_index[scale] + 1 + models.index(model))
            r1 = V.l1_eval(ref1, ts, lz_s, per, rng_m, tau=tau)
            _, _, v15 = V.PF.read_verdict(r1, 1.5)
            l1_rows.append(dict(scale=scale, model=model, n=n, verdict_tau1p5=v15, **r1))
            r4 = V.l4_eval(Xs, per, ref4, rng_m)
            lam = r4.pop("lam")
            l4_rows.append(dict(scale=scale, model=model, n=n, **r4, lam=" ".join(f"{x:.2f}" for x in lam),
                                human_lam=" ".join(f"{x:.2f}" for x in ref4.lam),
                                human_ev_ratio=ref4.ev[0] / ref4.ev[1], human_load_min=ref4.lam.min()))
            # description
            diff = float(ts.mean() - th.mean())
            lo5, hi5, v5 = tost_overall(diff, sd_ref, n, len(th), ts.var(ddof=1), th.var(ddof=1), 0.5)
            lo2, hi2, v2 = tost_overall(diff, sd_ref, n, len(th), ts.var(ddof=1), th.var(ddof=1), 0.2)
            sim_item_sd = Xs.std(0, ddof=1)
            sim_rbar = float(np.corrcoef(Xs.T)[np.triu_indices(J, 1)].mean()) if (sim_item_sd > 0).all() else np.nan
            d_rows.append(dict(
                scale=scale, model=model, n=n, sim_mean_total=float(ts.mean()), hum_mean_total=float(th.mean()),
                hum_mean_total_all=float(tot_all.mean()), sim_sd_total=float(ts.std(ddof=1)), hum_sd_total=sd_ref,
                hum_sd_total_all=sd_all, sd_ratio=float(ts.std(ddof=1) / sd_ref), mean_diff_sd=diff / sd_ref,
                l3_overall_ci90_lo_sd=lo5 / sd_ref, l3_overall_ci90_hi_sd=hi5 / sd_ref,
                l3_overall_verdict_0p5sd=v5, l3_overall_verdict_0p2sd=v2,
                item_mean_abs_diff_over_item_sd=float(np.mean(np.abs(Xs.mean(0) - hum_item_mean) / hum_item_sd)),
                item_sd_ratio_mean=float(np.mean(sim_item_sd / hum_item_sd)),
                item_profile_corr=float(np.corrcoef(Xs.mean(0), hum_item_mean)[0, 1]),
                sim_mean_inter_item_r=sim_rbar, hum_mean_inter_item_r=hum_rbar,
                sim_unique_profiles=int(len(np.unique(Xs, axis=0))), hum_unique_profiles_in_ref=int(hum_unique),
                sim_modal_profile_share=float(pd.Series(map(tuple, Xs)).value_counts().iloc[0] / n),
                sim_share_all_items_in_two_adjacent_cats=float(
                    np.mean([(np.ptp(row) <= 1) for row in Xs])),
                hum_share_all_items_in_two_adjacent_cats=float(np.mean(np.ptp(Xh, axis=1) <= 1))))
            print(f"[{scale}] {model:30s} sdratio {d_rows[-1]['sd_ratio']:.2f} dmean {d_rows[-1]['mean_diff_sd']:+.2f}SD | "
                  f"gate kmin {g['k_min']:.0f} single {g['pass_single']} | L1 mis {r1['misfit_ratio']:.2f} "
                  f"[{r1['misfit_ratio_ci90_lo']:.2f},{r1['misfit_ratio_ci90_hi']:.2f}] ov {r1['overfit_ratio']:.2f} "
                  f"[{r1['overfit_ratio_ci90_lo']:.2f},{r1['overfit_ratio_ci90_hi']:.2f}] {r1['verdict']} | "
                  f"R1 {r4['R1']} (ev {r4['ev_ratio']:.1f}, min {r4['load_min']:.2f}) R2 {r4['R2']} "
                  f"(phi {r4['phi']:.3f} lo {r4['phi_lo90']:.3f}; rmsd {r4['loading_rmsd']:.3f} lo {r4['loading_rmsd_lo90']:.3f}) "
                  f"({time.time() - t0:.0f}s)", flush=True)
        tag = args.tag
        pd.DataFrame(g_rows).to_csv(os.path.join(OUT, f"a10_gate__{scale}{tag}.csv"), index=False)
        pd.DataFrame(l1_rows).to_csv(os.path.join(OUT, f"a10_l1__{scale}{tag}.csv"), index=False)
        pd.DataFrame(l4_rows).to_csv(os.path.join(OUT, f"a10_l4__{scale}{tag}.csv"), index=False)
        pd.DataFrame(d_rows).to_csv(os.path.join(OUT, f"a10_desc__{scale}{tag}.csv"), index=False)
        pd.DataFrame([dict(scale=scale, **r) for r in trows]).to_csv(os.path.join(OUT, f"a10_l1_tau__{scale}{tag}.csv"), index=False)
        print(f"[{scale}] done in {time.time() - t_scale:.0f}s", flush=True)


if __name__ == "__main__":
    main()
