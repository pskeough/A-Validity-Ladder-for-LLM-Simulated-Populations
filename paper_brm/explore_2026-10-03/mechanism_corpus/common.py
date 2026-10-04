"""Shared code for the mechanism_corpus diagnostics (tests A to D).

Read-only with respect to the audit: it imports the frozen engines (78b pairing, 78c r3, 80_l3_lib
Design) by file path and reads their outputs. Nothing here edits an existing file. All new files are
written under this folder.

Single process, one BLAS thread (the machine hard-resets under sustained load).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util  # noqa: E402
import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCRIPTS = os.path.join(BASE, "scripts")
BRM = os.path.join(BASE, "analysis", "brm")
OUT = HERE


def load_script(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L3 = load_script("l3lib", "80_l3_lib.py")
R3M = load_script("l2r3", "78c_l2_r3_verdicts.py")
S78B = load_script("l2sim", "78b_l2_simulated_gaps.py")

DPQ = [f"DPQ0{i}0" for i in range(1, 9)]
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
ITEM_NAMES = ["1 interest", "2 mood", "3 sleep", "4 fatigue", "5 appetite", "6 worthless",
              "7 concentr", "8 psychomotor"]
RACES, SEXES, INCS, MARS = L3.RACES, L3.SEXES, L3.INCS, L3.MARS
CELLS = [(r, s, i, m) for r in RACES for s in SEXES for i in INCS for m in MARS]
CIDX = {c: k for k, c in enumerate(CELLS)}
MODELS = ["deepseek-chat-v3", "gemini-3-flash-preview", "glm-4.7", "gpt-4o-mini"]
FULLNAME = {"deepseek-chat-v3": "deepseek/deepseek-chat-v3",
            "gemini-3-flash-preview": "google/gemini-3-flash-preview",
            "glm-4.7": "z-ai/glm-4.7", "gpt-4o-mini": "openai/gpt-4o-mini"}
SHORT = {"deepseek-chat-v3": "DeepSeek-V3", "gemini-3-flash-preview": "Gemini-3-Flash",
         "glm-4.7": "GLM-4.7", "gpt-4o-mini": "GPT-4o-mini"}
FRAMINGS = ["clinical", "narrative"]
GROUPS = [("Overall", None, None)] + [(r, 0, r) for r in RACES] + [(s, 1, s) for s in SEXES] + \
         [(i, 2, i) for i in INCS]
_tb = pd.read_csv(os.path.join(BRM, "80a_tolerance_basis.csv"))
SD0 = float(_tb[(_tb.window == "2005-2018") & (_tb.population == "all adults 18+")].sd.iloc[0])
TOLS = {"0.2SD": 0.2 * SD0, "1pt": 1.0, "2pt": 2.0}


def attr(k):
    return np.array([c[k] for c in CELLS])


# ----------------------------------------------------------------------------------- NHANES
def load_nhanes():
    """NHANES 2005-2018 adults: 80a frame (cells, design, L3 weight w) joined to the item scores."""
    fr = pd.read_csv(os.path.join(BRM, "80a_nhanes_frame.csv"), low_memory=False)
    d = L3.window_frame(fr, "2005-2018", asian_adjust=True)
    sc = pd.read_csv(os.path.join(BRM, "l1_scores_nhanes.csv"))
    sc = sc[sc.reference == "2005_2018"][["SEQN", "w", "total"] + DPQ].rename(columns={"w": "w7"})
    n0 = len(d)
    d = d.merge(sc, on="SEQN", how="inner")
    assert len(d) == n0 == len(sc), (n0, len(d), len(sc))
    assert np.allclose(d.phq8, d.total)
    key = list(zip(d.race, d.sex, d.pir_band, d.mar))
    d["c48"] = np.array([CIDX.get(k, -1) for k in key])
    return d.reset_index(drop=True)


class CellRef:
    """48 NHANES cell totals with JKn replicates (80_l3_lib.Design). `sub` selects respondents."""

    def __init__(self, d, wcol="w", ycol="phq8"):
        self.des = L3.Design(d)
        ci = d.c48.to_numpy()
        ok = ci >= 0
        oh = np.zeros((len(d), 48))
        oh[np.flatnonzero(ok), ci[ok]] = 1.0
        w = d[wcol].to_numpy(float)
        y = d[ycol].to_numpy(float)
        self.Nf, self.Nr = self.des.replicate(self.des.psu_totals(oh * w[:, None]))
        self.Yf, self.Yr = self.des.replicate(self.des.psu_totals(oh * (w * y)[:, None]))
        self.mf, self.mr = self.Yf / self.Nf, self.Yr / self.Nr
        self.df = self.des.dfree
        self.n_resp = oh[ok].sum(0)


# ------------------------------------------------------------------------------------ corpus
def load_corpus():
    m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v3.csv"), low_memory=False)
    assert len(m) == 28800
    m = m[m.phq8_valid.astype(bool)].copy()
    assert len(m) == 28799
    m[ITEMS] = m[ITEMS].astype(int)
    m["total"] = m[ITEMS].sum(axis=1)
    m["short"] = m.model.map({v: k for k, v in FULLNAME.items()})
    m["framing"] = m.prompt_condition
    return m.reset_index(drop=True)


def cis_matched(m):
    """Draws of the 48 matched persona cells (cisgender, four races) with a c48 index."""
    x = m[m.gender.isin(["Cisgender Man", "Cisgender Woman"]) & m.race.isin(RACES)].copy()
    x["sex"] = x.gender.map({"Cisgender Man": "Men", "Cisgender Woman": "Women"})
    x["c48"] = [CIDX[(r, s, i, mm)] for r, s, i, mm in zip(x.race, x.sex, x.ses_normalized,
                                                          x.relationship)]
    return x


def sim_cells(x):
    """dict (model, framing) -> dict with s, v, n, s2, items (48 x 8 means). framing in clinical,
    narrative, combined; model in MODELS + pooled. v is the variance of the cell mean."""
    out = {}
    for mdl in MODELS:
        per = {}
        for f in FRAMINGS:
            g = x[(x.short == mdl) & (x.framing == f)]
            grp = g.groupby("c48")
            n = grp.size().reindex(range(48)).to_numpy(float)
            s = grp.total.mean().reindex(range(48)).to_numpy()
            s2 = grp.total.var(ddof=1).reindex(range(48)).to_numpy()
            it = np.vstack([grp[c].mean().reindex(range(48)).to_numpy() for c in ITEMS]).T
            itv = np.vstack([grp[c].var(ddof=1).reindex(range(48)).to_numpy() for c in ITEMS]).T
            per[f] = dict(s=s, n=n, s2=s2, v=s2 / n, items=it, item_var=itv)
        out[(mdl, f"clinical")] = per["clinical"]
        out[(mdl, f"narrative")] = per["narrative"]
        c, nn = per["clinical"], per["narrative"]
        out[(mdl, "combined")] = dict(s=(c["s"] + nn["s"]) / 2, v=(c["v"] + nn["v"]) / 4,
                                      n=c["n"] + nn["n"], s2=(c["s2"] + nn["s2"]) / 2,
                                      items=(c["items"] + nn["items"]) / 2,
                                      item_var=(c["item_var"] + nn["item_var"]) / 2)
    for f in ["clinical", "narrative", "combined"]:
        S = np.vstack([out[(m_, f)]["s"] for m_ in MODELS])
        V = np.vstack([out[(m_, f)]["v"] for m_ in MODELS])
        IT = np.stack([out[(m_, f)]["items"] for m_ in MODELS])
        out[("pooled", f)] = dict(s=S.mean(0), v=V.sum(0) / 16.0, n=None, s2=None,
                                  items=IT.mean(0), item_var=None)
    return out


# --------------------------------------------------------------------------- level-3 engine
def tost(est, var_sim, var_ref, df_ref):
    var = var_sim + var_ref
    se = float(np.sqrt(var))
    df = var ** 2 / (var_ref ** 2 / df_ref) if var_ref > 0 else 1e9
    tc = float(stats.t.ppf(0.95, df))
    return se, df, tc, est - tc * se, est + tc * se


def l3_rows(cref, s, v, groups=GROUPS):
    """PS residual per group (80c 'PS', mean outcome), s and v are 48 cell values and variances."""
    rows = []
    for gname, dim, val in groups:
        g = np.ones(48, bool) if dim is None else attr(dim) == val
        ga = g & ~np.isnan(s)
        Ng, Ngr = cref.Nf[ga].sum(), cref.Nr[:, ga].sum(1)
        p, pr = cref.Nf[ga] / Ng, cref.Nr[:, ga] / Ngr[:, None]
        sim, simr = float(p @ s[ga]), pr @ s[ga]
        ref, refr = cref.Yf[ga].sum() / Ng, cref.Yr[:, ga].sum(1) / Ngr
        res, resr = sim - ref, simr - refr
        vref = float(cref.des.jk_var(res, resr))
        vsim = float(np.sum(p ** 2 * v[ga]))
        se, df, tc, lo, hi = tost(res, vsim, vref, cref.df)
        rows.append(dict(group=gname, sim=sim, ref=ref, resid=res, se=se, df=df, ci90_lo=lo,
                         ci90_hi=hi, se_sim=np.sqrt(vsim), se_ref=np.sqrt(vref)))
    return rows


# --------------------------------------------------------------------------- level-2 helpers
def l2_reference():
    r = pd.read_csv(os.path.join(BRM, "l2_reference.csv"))
    return r[r.variant == "primary"].set_index(["estimand", "contrast"])


STOPS = {"reference too imprecise", "no population gap"}


def l2_model_verdict(verdicts):
    live = [v for v in verdicts if v not in STOPS]
    if any(v != "undetermined" and "kept" not in v.split(" or ") for v in live):
        return "fail"
    if live and all(v == "kept" for v in live):
        return "pass"
    return "unresolved"


def l2_sim_gaps():
    """Per-model combined-framing persona-pair gaps by 78b's own functions: dict
    (model_fullname, contrast) -> (gap, se, df) on the four-model, four-race v3 corpus."""
    v3 = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v3.csv"), low_memory=False)
    v3 = v3[v3.phq8_valid.astype(bool)].copy()
    cell = S78B.cells(v3)
    rows, _ = S78B.run(cell, "all", "four models", "four", S78B.MODELS)
    out = {}
    for r in rows:
        if r["framing"] == "combined":
            out[(r["scope"], r["contrast"])] = (r["simulated"], r["se"], r["df"])
    return out


def r3_verdict(g, se_g, df_g, ref_row, alpha=0.05 / 7):
    return R3M.r3(g, se_g, df_g, ref_row.estimate, ref_row.se_taylor, ref_row.df, alpha,
                  R3M.BANDS["asym"])


# -------------------------------------------------------------------------------- misc stats
def wvar(x, w):
    """Unbiased weighted (reliability) variance."""
    w = np.asarray(w, float)
    m = np.average(x, weights=w)
    d = np.sum(w * (x - m) ** 2)
    den = w.sum() - np.sum(w ** 2) / w.sum()
    return d / den if den > 0 else np.nan


def spearman(a, b):
    return stats.spearmanr(a, b)[0]


def pearson(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])
