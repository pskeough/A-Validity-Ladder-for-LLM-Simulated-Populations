"""Level 1 (person fit) on the decoding control (reviewer request; SUPPORTING ANALYSIS, frozen
level-1 rule of paper_brm/LADDER_SPEC.md applied unchanged).

Question. The temperature-0 decoding control (scripts/55, 56, 60, 82d) addressed draw variance.
Does decoding also change answer-pattern regularity, which is what level 1 reads?

Data. analysis/decoding_control_raw.jsonl through decoding_control_io (same loader, dedup and
pairing as 56, 57, 60, 82d): 4 endpoints x 2 arms (provider default, temperature 0 / top_p 1) x 12
cohorts x 30 draws, clinical framing. The released corpus's clinical draws of the same 12
personas per model are carried as a third column ("corpus_clinical", as in 82d) for comparison.
Gemini-3-Flash and GLM-4.7 ran on undated routes (the dated snapshots were retired), so their
control arms are a different build from the corpus (decoding_control_design.json).

Level 1, exactly as 79c (frozen rule):
  GRM       NHANES 2005-2018 weighted item parameters (analysis/brm/l1_grm_params.csv, fit
            "weighted"); WLE theta and lz* for every draw by 79_l1_lib.score_vectors.
  reference NHANES 2005-2018 adults, analysis/brm/l1_scores_nhanes.csv (reference 2005_2018,
            lzstar_weighted, MEC weight w); total strata of >= 100 respondents
            (79_l1_lib.total_strata); within-stratum CDF (TailRef); randomised-PIT tail shares
            (tail_probs).
  ratios    misfit = share below the within-total 5th percentile / .05; overfit = share above
            the 95th / .05.
  intervals persona-clustered bootstrap of the draws (12 personas) jointly with the Rao-Wu PSU
            bootstrap of NHANES (79_l1_lib.raowu_mult), paired by replicate, B = 2000, 90%
            percentile intervals. Within a model the same persona resample is used for every
            arm, so the arm differences (temp0 minus default) are paired; their 90% intervals are
            reported as a descriptive extra.
  verdict   79c.read_verdict at the frozen tau = 1.5: pass when both ratios' 90% intervals lie in
            [1/1.5, 1.5]; fail when either tail is wholly outside; both tails below = compressed.
Checks: lz* recomputed here for the corpus rows equals analysis/brm/l1_scores_corpus.csv, and the
point ratios of each model's full clinical corpus (120 personas) equal l1_personfit_main.csv.

Output: analysis/brm/96_decoding_l1.csv. Seeded (20261003). Single process.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUTD = os.path.join(BASE, "analysis", "brm")
sys.path.insert(0, HERE)
from decoding_control_io import balanced_cells, both_arms_only, load  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = [old[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


L = _load("l1lib", "79_l1_lib.py")
PF = _load("l1pf", "79c_l1_personfit.py")

B = 2000
SEED = 20261003
TAU = 1.5
CTRL2CORPUS = {"gpt-4o-mini": "openai/gpt-4o-mini", "deepseek-chat": "deepseek/deepseek-chat-v3",
               "gemini-3-flash-preview": "google/gemini-3-flash-preview", "glm-4.7": "z-ai/glm-4.7"}
ARMS = ["default", "temp0", "corpus_clinical"]


def grm_params():
    par = pd.read_csv(os.path.join(OUTD, "l1_grm_params.csv"))
    par = par[par.fit == "weighted"].set_index("item").loc[L.DPQ]
    return par.a.to_numpy(float), par[["b1", "b2", "b3"]].to_numpy(float)


def main():
    rng = np.random.default_rng(SEED)
    a, b = grm_params()

    # ------------------------------------------------------------------ reference (as 79c)
    nall = pd.read_csv(os.path.join(OUTD, "l1_scores_nhanes.csv"))
    n0 = nall[nall.reference == "2005_2018"].reset_index(drop=True)
    smap = L.total_strata(n0.total.to_numpy(), 100)
    cond = L.TailRef(smap[n0.total.to_numpy()], n0.lzstar_weighted.to_numpy())
    w0 = n0.w.to_numpy()
    mult, pidx = L.raowu_mult(n0, B, rng)

    # ------------------------------------------------------------------ simulated draws
    dc, rep = load(verbose=True)
    dc = balanced_cells(both_arms_only(dc, verbose=True), verbose=True)
    X = np.array(dc["items"].tolist(), dtype=int)
    assert X.shape[1] == 8 and np.isin(X, [0, 1, 2, 3]).all(), "item outside 0..3"
    assert (X.sum(axis=1) == dc.total.to_numpy()).all()
    _, _, lzs = L.score_vectors(a, b, X)
    dc = dc.assign(lz=lzs, model_full=dc.model.map(CTRL2CORPUS))
    assert dc.model_full.notna().all()

    corpus = L.load_corpus()
    sc = pd.read_csv(os.path.join(OUTD, "l1_scores_corpus.csv"))
    assert (sc.total.to_numpy() == corpus.total.to_numpy()).all()
    Xc = corpus[L.ITEMS].to_numpy(int)
    _, _, lzc = L.score_vectors(a, b, Xc)
    d_lz = float(np.max(np.abs(lzc - sc.lzstar_weighted.to_numpy())))
    print(f"check: lz* recomputed for {len(corpus)} corpus rows vs l1_scores_corpus.csv, "
          f"max |diff| {d_lz:.2e}")
    assert d_lz < 1e-6
    corpus["lz"] = lzc

    # check: point ratios of each model's full clinical corpus vs the published 79c row
    pub = pd.read_csv(os.path.join(OUTD, "l1_personfit_main.csv"))
    pub = pub[(pub.reference == "2005_2018 weighted GRM") & (pub["sample"] == "full") &
              (pub.framing == "clinical") & (pub.subset == "all")].set_index("model")
    for full in CTRL2CORPUS.values():
        m = (corpus.model == full) & (corpus.framing == "clinical")
        loc = cond.locate(smap[corpus.total[m].to_numpy()], corpus.lz[m].to_numpy())
        Fm, F = cond.cdf(loc, w0)
        cl, ch, _ = L.tail_probs(Fm, F)
        p = pub.loc[L.SHORT[full]]
        dm, do = abs(cl.mean() / .05 - p.misfit_ratio), abs(ch.mean() / .05 - p.overfit_ratio)
        print(f"check: {L.SHORT[full]:15s} full clinical corpus point ratios vs 79c: "
              f"misfit diff {dm:.1e}, overfit diff {do:.1e}")
        assert dm < 1e-9 and do < 1e-9

    # ------------------------------------------------------------------ groups
    parts = []
    for cm, full in CTRL2CORPUS.items():
        g = dc[dc.model == cm]
        cohorts = sorted(g.profile_id.unique())
        assert len(cohorts) == 12, (cm, len(cohorts))
        for arm in ("default", "temp0"):
            x = g[g.arm == arm]
            parts.append(pd.DataFrame(dict(model=full, arm=arm, profile_id=x.profile_id,
                                           total=x.total, lz=x.lz)))
        cc = corpus[(corpus.model == full) & (corpus.framing == "clinical") &
                    corpus.profile_id.isin(cohorts)]
        assert cc.profile_id.nunique() == 12
        parts.append(pd.DataFrame(dict(model=full, arm="corpus_clinical",
                                       profile_id=cc.profile_id, total=cc.total, lz=cc.lz)))
    q = pd.concat(parts, ignore_index=True)
    tot, val = q.total.to_numpy(int), q.lz.to_numpy(float)
    loc = cond.locate(smap[tot], val)
    Fm, F = cond.cdf(loc, w0)
    cl_pt, ch_pt, mid_pt = L.tail_probs(Fm, F)

    # persona resample per model, shared by the model's arms (paired differences)
    models = list(CTRL2CORPUS.values())
    pers_list = {m: sorted(q[q.model == m].profile_id.unique()) for m in models}
    ridx = {m: rng.integers(0, 12, size=(B, 12)) for m in models}
    keys = [(m, arm) for m in models for arm in ARMS]
    gidx = {}
    for m, arm in keys:
        mask = ((q.model == m) & (q.arm == arm)).to_numpy()
        pos = {p: i for i, p in enumerate(pers_list[m])}
        pinv = np.array([pos[p] for p in q.profile_id[mask]])
        gidx[(m, arm)] = (mask, pinv, np.bincount(pinv, minlength=12).astype(float))
    bl = {k: np.empty(B) for k in keys}
    bh = {k: np.empty(B) for k in keys}
    for bb in range(B):
        wb = mult[bb, pidx] * w0
        Fm, F = cond.cdf(loc, wb)
        l, h, _ = L.tail_probs(Fm, F)
        for k in keys:
            mask, pinv, cnt = gidx[k]
            sl = np.bincount(pinv, weights=l[mask], minlength=12)
            sh = np.bincount(pinv, weights=h[mask], minlength=12)
            ix = ridx[k[0]][bb]
            den = cnt[ix].sum()
            bl[k][bb], bh[k][bb] = sl[ix].sum() / den, sh[ix].sum() / den

    rows = []
    for m, arm in keys:
        mask, pinv, cnt = gidx[(m, arm)]
        r = dict(model=L.SHORT[m], arm=arm, n_personas=12, n_draws=int(mask.sum()),
                 mean_lzstar=float(val[mask].mean()), mean_mid_pit=float(mid_pt[mask].mean()),
                 misfit_ratio=cl_pt[mask].mean() / .05,
                 misfit_ratio_ci90_lo=np.quantile(bl[(m, arm)], .05) / .05,
                 misfit_ratio_ci90_hi=np.quantile(bl[(m, arm)], .95) / .05,
                 overfit_ratio=ch_pt[mask].mean() / .05,
                 overfit_ratio_ci90_lo=np.quantile(bh[(m, arm)], .05) / .05,
                 overfit_ratio_ci90_hi=np.quantile(bh[(m, arm)], .95) / .05)
        vm, vo, ov = PF.read_verdict(r, TAU)
        r.update(misfit_reading=vm, overfit_reading=vo, verdict=ov, tau=TAU)
        if arm == "temp0":
            dmis = (bl[(m, "temp0")] - bl[(m, "default")]) / .05
            dove = (bh[(m, "temp0")] - bh[(m, "default")]) / .05
            mt = gidx[(m, "default")][0]
            r.update(misfit_diff_vs_default=r["misfit_ratio"] - cl_pt[mt].mean() / .05,
                     misfit_diff_ci90_lo=np.quantile(dmis, .05),
                     misfit_diff_ci90_hi=np.quantile(dmis, .95),
                     overfit_diff_vs_default=r["overfit_ratio"] - ch_pt[mt].mean() / .05,
                     overfit_diff_ci90_lo=np.quantile(dove, .05),
                     overfit_diff_ci90_hi=np.quantile(dove, .95))
        p = pub.loc[L.SHORT[m]]
        r.update(full_grid_clinical_verdict_79c=p.verdict,
                 full_grid_misfit_ci90=f"{p.misfit_ratio_ci90_lo:.3f}-{p.misfit_ratio_ci90_hi:.3f}",
                 full_grid_overfit_ci90=f"{p.overfit_ratio_ci90_lo:.3f}-{p.overfit_ratio_ci90_hi:.3f}")
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTD, "96_decoding_l1.csv"), index=False)

    pd.set_option("display.width", 250)
    cols = ["model", "arm", "n_draws", "misfit_ratio", "misfit_ratio_ci90_lo",
            "misfit_ratio_ci90_hi", "overfit_ratio", "overfit_ratio_ci90_lo",
            "overfit_ratio_ci90_hi", "verdict"]
    print()
    print(res[cols].round(3).to_string(index=False))
    print("\npaired arm difference, temp0 minus default (ratio units, 90% interval):")
    print(res[res.arm == "temp0"][["model", "misfit_diff_vs_default", "misfit_diff_ci90_lo",
                                   "misfit_diff_ci90_hi", "overfit_diff_vs_default",
                                   "overfit_diff_ci90_lo", "overfit_diff_ci90_hi"]]
          .round(3).to_string(index=False))
    print("\nwritten analysis/brm/96_decoding_l1.csv")


if __name__ == "__main__":
    main()
