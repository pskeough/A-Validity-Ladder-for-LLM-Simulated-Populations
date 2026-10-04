"""Validity ladder on the Petrov, Serapio-Garcia and Rentfrow (2024) release: GPT-3.5 and GPT-4
answering the 44-item BFI as personas ("Limited ability of LLMs to simulate human psychological
behaviours: a psychometric analysis", arXiv 2405.07248; github.com/nikbpetrov/LLMs-Simulate-Humans).

Simulated populations (raw files in paper_brm/external/raw/petrov2024, SOURCE.txt there)
  generic  150 PersonaChat-style personas, no attributes, one answer set each
  silicon  1,000 real BBC Big Personality Test respondents turned into personas by their demographics
           (age, sex, country, schooling, occupation ...), one answer set each. The BBC human answers
           are NOT released item by item; the release holds each respondent's five domain means
           (bbc_summary_scores_df.pickle), so the human twin of every silicon persona is known at the
           domain level only.
Each model x population has ONE draw per persona. The gate and level 1's persona clustering are
therefore not defined as in the frozen spec; each persona is its own cluster.

Reference for levels 1 and 4: Twin-2K-500 BFI-44 (US panel, N = 2,058 complete), the same file the
PersonaLLM shakedown used (paper_brm/external/raw/twin2k/wave1_3_response.csv), because the BBC item
answers are not public. Items are scored 0-4 after reverse keying. The release stores answers already
reverse-keyed (column response_reversed, 1-5), which are used as stored (x - 1).

Rungs
  gate      not applicable (one draw per persona)
  level 1   frozen rule, tau from Twin-2K subgroups (sex, age band, race), B = 2,000
  level 4   R1 and R2 against Twin-2K loadings, B = 500; R3 diagnostic; R4 not run (package lacks it;
            level 4 therefore cannot pass here)
  level 3   silicon only, twin-matched: residual of persona domain mean minus its BBC human's domain
            mean, per group (all, sex, age band, UK), paired SE, TOST at 90% against 0.5 and 0.2 BBC
            SD. This is the post-stratified residual with one twin per cell, as in the Twin-2K run
            (scripts/93_twin2k_ladder.py).
  level 2   silicon only, twin-matched: simulated gap g and human gap gamma between two groups on
            the same 1,000 people, respondent bootstrap (B = 2,000) keeping each persona with its
            human, level2_contrast (Fieller, closed form, df infinite, stops 1 and 2), Bonferroni over
            the model's family (5 domains x contrasts). Marginal gaps only (no standardisation).

Seed 20261020. One process; B_L1 = 2,000, B_L4 = 500.
Output: out/p20_*.csv
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
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "C:/Research/PsychBench/UpdatedRun"
sys.path.insert(0, os.path.join(BASE, "package", "src"))
from validity_ladder.level2 import level2_contrast  # noqa: E402
from validity_ladder.level3 import read_tost  # noqa: E402

RAWP = os.path.join(BASE, "paper_brm", "external", "raw", "petrov2024")
RAWT = os.path.join(BASE, "paper_brm", "external", "raw", "twin2k")
OUT = os.path.join(HERE, "out")
SEED = 20261020
B_L1, B_L4, B_L2 = 2000, 500, 2000

# BFI-44 scales, 1-based item numbers; Petrov's release is already reverse-keyed
SCALES = {
    "Extraversion": [1, 6, 11, 16, 21, 26, 31, 36],
    "Agreeableness": [2, 7, 12, 17, 22, 27, 32, 37, 42],
    "Conscientiousness": [3, 8, 13, 18, 23, 28, 33, 38, 43],
    "Neuroticism": [4, 9, 14, 19, 24, 29, 34, 39],
    "Openness": [5, 10, 15, 20, 25, 30, 35, 40, 41, 44],
}
# scripts/92_personallm_shakedown.py keys for the Twin-2K raw answers (negative = reverse-keyed)
HUMAN_KEY = {
    "Extraversion": [1, -6, 11, 16, -21, 26, -31, 36],
    "Agreeableness": [-2, 7, -12, 17, 22, -27, 32, -37, 42],
    "Conscientiousness": [3, -8, 13, -18, -23, 28, 33, 38, -43],
    "Neuroticism": [4, -9, 14, 19, -24, 29, -34, 39],
    "Openness": [5, 10, 15, 20, 25, 30, -35, 40, -41, 44],
}
POPS = [("generic", "gpt35"), ("generic", "gpt4"), ("silicon", "gpt35"), ("silicon", "gpt4")]
MODEL_NAME = {"gpt35": "GPT-3.5", "gpt4": "GPT-4"}


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


def load_twin_humans():
    d = pd.read_csv(os.path.join(RAWT, "wave1_3_response.csv"), low_memory=False)
    h = d[["QID12", "QID13", "QID15"] + [f"QID25_{i}" for i in range(1, 45)]].dropna()
    h = h.rename(columns={f"QID25_{i}": f"i{i}" for i in range(1, 45)})
    return h.reset_index(drop=True)


def keyed_human(df, items):
    return np.column_stack([(df[f"i{abs(i)}"].to_numpy(int) - 1) if i > 0 else (5 - df[f"i{abs(i)}"].to_numpy(int))
                            for i in items])


def load_pop(pop, model):
    d = pd.read_csv(os.path.join(RAWP, f"{pop}_{model}_df.csv"))
    b = d[d.scale == "BFI"]
    return b.pivot(index="uid", columns="item_index", values="response_reversed")   # uid x item (1-5, keyed)


def tost_paired(r, sd_ref, delta_sd):
    n = len(r)
    m, se = float(r.mean()), float(r.std(ddof=1) / np.sqrt(n))
    tc = float(stats.t.ppf(0.95, n - 1))
    lo, hi = m - tc * se, m + tc * se
    tol = delta_sd * sd_ref
    return m, se, lo, hi, read_tost(lo, hi, tol)


def main():
    os.makedirs(OUT, exist_ok=True)
    hum = load_twin_humans()
    sex, age, race = hum.QID12.to_numpy(), hum.QID13.to_numpy(), hum.QID15.to_numpy()
    groups = {f"sex={v}": sex == v for v in np.unique(sex)}
    groups.update({f"age={v}": age == v for v in np.unique(age)})
    groups.update({f"race={v}": race == v for v in np.unique(race) if (race == v).sum() >= 100})
    groups = {k: m for k, m in groups.items() if m.sum() >= 100}
    print(f"Twin-2K reference {len(hum)} respondents; tau groups {len(groups)}", flush=True)

    wide = {p: load_pop(*p) for p in POPS}
    for p, w in wide.items():
        print(p, w.shape, "complete rows", int(w.notna().all(axis=1).sum()), flush=True)
    attrs = pd.read_pickle(os.path.join(RAWP, "bbc_silicon_samples_df.pickle")).set_index("uid")
    bbc = pd.read_pickle(os.path.join(RAWP, "bbc_summary_scores_df.pickle")).pivot(
        index="uid", columns="dimension", values="response_reversed")

    l1_rows, l4_rows, d_rows, tau_rows, l3_rows, l2_rows, na_rows = [], [], [], [], [], [], []
    sim_dom = {}            # (model, scale) -> Series of silicon domain means (1-5) indexed by uid

    for si, (scale, items) in enumerate(SCALES.items()):
        t_scale = time.time()
        rng = np.random.default_rng(SEED + 100 * si)
        J, M = len(items), 5
        Xh = keyed_human(hum, HUMAN_KEY[scale])
        th = Xh.sum(1)
        sd = float(th.std(ddof=1))
        grm = V.GRM(J, M)
        fit = grm.fit(Xh)
        _, _, lz_h = grm.score(fit["a"], fit["b"], Xh)
        ref1 = V.L1Ref(th, lz_h, None, J * (M - 1), B_L1, rng)
        tau, worst, trows = V.tau_from_subgroups(ref1, th, lz_h, groups)
        tau_rows += [dict(scale=scale, **r) for r in trows]
        ref4 = V.L4Ref(Xh, None, M, B_L4, rng)
        assert (ref4.lam > 0).all()
        hum_item_sd, hum_item_mean = Xh.std(0, ddof=1), Xh.mean(0)
        hum_rbar = float(np.corrcoef(Xh.T)[np.triu_indices(J, 1)].mean())
        print(f"[{scale}] GRM ok {fit['converged']}, human SD {sd:.2f}, tau {tau} (worst resolved {worst:.2f}), "
              f"human R1 abs {ref4.gf} ev {ref4.ev[0] / ref4.ev[1]:.1f} loadings "
              f"{' '.join(f'{x:.2f}' for x in ref4.lam)}", flush=True)
        for (pop, model) in POPS:
            w = wide[(pop, model)]
            cols = [c for c in items]
            sub = w[cols]
            ok = sub.notna().all(axis=1)
            Xs = (sub[ok].to_numpy() - 1).astype(int)
            uids = sub.index[ok]
            n = len(Xs)
            ts = Xs.sum(1)
            per = np.arange(n)
            na_rows.append(dict(scale=scale, population=pop, model=MODEL_NAME[model], n_personas=len(w),
                                n_complete=n, gate="not applicable: one draw per persona"))
            rng_m = np.random.default_rng(SEED + 100 * si + 1 + POPS.index((pop, model)))
            _, _, lz_s = grm.score(fit["a"], fit["b"], Xs)
            r1 = V.l1_eval(ref1, ts, lz_s, per, rng_m, tau=tau)
            _, _, v15 = V.PF.read_verdict(r1, 1.5)
            l1_rows.append(dict(scale=scale, population=pop, model=MODEL_NAME[model], n=n, verdict_tau1p5=v15, **r1))
            r4 = V.l4_eval(Xs, per, ref4, rng_m)
            lam = r4.pop("lam")
            l4_rows.append(dict(scale=scale, population=pop, model=MODEL_NAME[model], n=n, **r4,
                                lam=" ".join(f"{x:.2f}" for x in lam), human_lam=" ".join(f"{x:.2f}" for x in ref4.lam),
                                human_ev_ratio=ref4.ev[0] / ref4.ev[1], human_load_min=ref4.lam.min()))
            sim_sd = Xs.std(0, ddof=1)
            d_rows.append(dict(
                scale=scale, population=pop, model=MODEL_NAME[model], n=n, sim_mean_total=float(ts.mean()),
                hum_mean_total=float(th.mean()), sim_sd_total=float(ts.std(ddof=1)), hum_sd_total=sd,
                sd_ratio=float(ts.std(ddof=1) / sd), mean_diff_sd=float((ts.mean() - th.mean()) / sd),
                item_mean_abs_diff_over_item_sd=float(np.mean(np.abs(Xs.mean(0) - hum_item_mean) / hum_item_sd)),
                item_sd_ratio_mean=float(np.mean(sim_sd / hum_item_sd)),
                item_profile_corr=float(np.corrcoef(Xs.mean(0), hum_item_mean)[0, 1]),
                sim_mean_inter_item_r=float(np.corrcoef(Xs.T)[np.triu_indices(J, 1)].mean()),
                hum_mean_inter_item_r=hum_rbar, sim_unique_profiles=int(len(np.unique(Xs, axis=0))),
                hum_unique_profiles=int(len(np.unique(Xh, axis=0)))))
            print(f"[{scale}] {pop:7s} {MODEL_NAME[model]:7s} n {n:4d} sdratio {d_rows[-1]['sd_ratio']:.2f} "
                  f"dmean {d_rows[-1]['mean_diff_sd']:+.2f}SD | L1 mis {r1['misfit_ratio']:.2f} "
                  f"[{r1['misfit_ratio_ci90_lo']:.2f},{r1['misfit_ratio_ci90_hi']:.2f}] ov {r1['overfit_ratio']:.2f} "
                  f"[{r1['overfit_ratio_ci90_lo']:.2f},{r1['overfit_ratio_ci90_hi']:.2f}] {r1['verdict']} | "
                  f"R1 {r4['R1']} (ev {r4['ev_ratio']:.1f}, min {r4['load_min']:.2f}) R2 {r4['R2']} "
                  f"(phi {r4['phi']:.3f} lo {r4['phi_lo90']:.3f}; rmsd {r4['loading_rmsd']:.3f} lo {r4['loading_rmsd_lo90']:.3f}) "
                  f"R3 {r4['R3']}", flush=True)
            if pop == "silicon":
                # domain mean on the 1-5 scale, as the release's own BBC summary scores
                sim_dom[(model, scale)] = pd.Series(sub[ok].mean(axis=1).to_numpy(), index=uids)
        print(f"[{scale}] done in {time.time() - t_scale:.0f}s", flush=True)

    # ------------------------------------------------------------------ levels 3 and 2, silicon twins
    rng = np.random.default_rng(SEED + 9000)
    age_ = attrs["age"]
    group_defs = {
        "all": lambda a: np.ones(len(a), bool),
        "sex=0": lambda a: a["sex"].to_numpy() == 0,
        "sex=1": lambda a: a["sex"].to_numpy() == 1,
        "age<30": lambda a: a["age"].to_numpy() < 30,
        "age30-44": lambda a: (a["age"].to_numpy() >= 30) & (a["age"].to_numpy() < 45),
        "age45+": lambda a: a["age"].to_numpy() >= 45,
        "country=GB": lambda a: a["country"].to_numpy() == "GB",
        "country!=GB": lambda a: a["country"].to_numpy() != "GB",
    }
    contrasts = [("sex=1 minus sex=0", "sex=1", "sex=0"), ("age<30 minus age45+", "age<30", "age45+")]
    for model in ("gpt35", "gpt4"):
        fam = len(SCALES) * len(contrasts)
        for scale in SCALES:
            s = sim_dom[(model, scale)]
            h = bbc[scale]
            common = s.index.intersection(h.index).intersection(attrs.index)
            s, h, a = s.loc[common].to_numpy(), h.loc[common].to_numpy(), attrs.loc[common]
            sd_h = float(h.std(ddof=1))
            n = len(common)
            r = s - h
            for gname, gf in group_defs.items():
                m = gf(a)
                if m.sum() < 30:
                    continue
                row = dict(scale=scale, model=MODEL_NAME[model], group=gname, n=int(m.sum()), hum_sd=sd_h,
                           sim_mean=float(s[m].mean()), hum_mean=float(h[m].mean()),
                           twin_r=float(np.corrcoef(s[m], h[m])[0, 1]))
                for dsd in (0.5, 0.2):
                    mean_r, se, lo, hi, v = tost_paired(r[m], sd_h, dsd)
                    row.update({f"resid": mean_r, f"resid_sd": mean_r / sd_h, f"ci90_lo_sd": lo / sd_h,
                                f"ci90_hi_sd": hi / sd_h, f"verdict_{str(dsd).replace('.', 'p')}sd": v})
                l3_rows.append(row)
            # level 2: respondent bootstrap keeping persona and human together
            counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B_L2)]).astype(float)
            for cname, ga, gb in contrasts:
                ma, mb = group_defs[ga](a), group_defs[gb](a)
                if ma.sum() < 30 or mb.sum() < 30:
                    continue

                def gap(y):
                    wa, wb = counts * ma, counts * mb
                    return (wa @ y) / wa.sum(1) - (wb @ y) / wb.sum(1)
                g_b, p_b = gap(s), gap(h)
                g0 = float(s[ma].mean() - s[mb].mean())
                p0 = float(h[ma].mean() - h[mb].mean())
                se_g, se_p = float(g_b.std(ddof=1)), float(p_b.std(ddof=1))
                cov = float(np.cov(g_b, p_b)[0, 1])
                out = level2_contrast(g0, se_g, p0, se_p, cov=cov, family_size=fam)
                l2_rows.append(dict(scale=scale, model=MODEL_NAME[model], contrast=cname, n_a=int(ma.sum()),
                                    n_b=int(mb.sum()), g=g0, se_g=se_g, gamma=p0, se_gamma=se_p, cov=cov,
                                    ratio=out["ratio"], ci_lo=out["ci_lo"], ci_hi=out["ci_hi"],
                                    reading=out["reading"], verdict=out["verdict"], label=out["label"],
                                    k_rel_halfwidth=out.get("k_rel_halfwidth"), family_size=fam))

    pd.DataFrame(l1_rows).to_csv(os.path.join(OUT, "p20_l1.csv"), index=False)
    pd.DataFrame(l4_rows).to_csv(os.path.join(OUT, "p20_l4.csv"), index=False)
    pd.DataFrame(d_rows).to_csv(os.path.join(OUT, "p20_desc.csv"), index=False)
    pd.DataFrame(tau_rows).to_csv(os.path.join(OUT, "p20_l1_tau.csv"), index=False)
    pd.DataFrame(na_rows).to_csv(os.path.join(OUT, "p20_gate_na.csv"), index=False)
    pd.DataFrame(l3_rows).to_csv(os.path.join(OUT, "p20_l3.csv"), index=False)
    pd.DataFrame(l2_rows).to_csv(os.path.join(OUT, "p20_l2.csv"), index=False)
    l3 = pd.DataFrame(l3_rows)
    print(l3[l3.group == "all"][["scale", "model", "n", "twin_r", "resid_sd", "ci90_lo_sd", "ci90_hi_sd",
                                  "verdict_0p5sd", "verdict_0p2sd"]].to_string(), flush=True)
    print(pd.DataFrame(l2_rows)[["scale", "model", "contrast", "g", "gamma", "ratio", "ci_lo", "ci_hi", "verdict",
                                 "label"]].to_string(), flush=True)


if __name__ == "__main__":
    main()
