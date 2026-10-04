"""Malmberg (2026) silicon-sample conjoint: GPT-4o-mini, Llama 3.2 and Magistral-small-2506 against their
matched CHIP50 live respondents, plus a human split-half ideal-simulator control.

DESIGN (fixed on 3 Oct 2026 before any ratio was computed; do not change after seeing results)

Data.  One row of each model file is one task (a pair of location profiles) shown to one CHIP50 respondent,
with the live choice (`which-profile-selected`, 1 or 2) and the silicon choice for the same task and the same
covariates (`outzeroshotresults1` for GPT, `choiceoutput` for Llama and Magistral).  Columns F-1-<profile>-<attr>
hold the six attribute levels of profile 1 and 2.  The same respondent-task can appear in several rows (the
authors drew 2,000 tasks per trial over 15 trials); the human choice is identical across those rows, the silicon
choice is a fresh draw.  All rows are used, as in the authors' own AMCEs.

Outcome.  Choice of a profile (0/1), two profile rows per task.  The AMCE of a level is the coefficient of its
dummy in the linear regression of the choice on all 14 non-baseline level dummies (cjoint::amce with an unconstrained
design), computed separately within each respondent subgroup.

Estimand (the "group gap").  The subgroup difference in an AMCE: AMCE of a level among subgroup A minus the
AMCE of the same level among subgroup B.  Primary contrast: party, Republican minus Democrat (the conjoint
analogue of the AMCE of candidate party among Republicans minus among Democrats; here the political content is
the three policy attributes, not a party label).  Further contrasts, fixed in advance: gender (Female - Male),
race (African American - White), age (a young adult - a senior citizen).  Education and income are not in the
released files, so no education contrast exists.
Levels read: the eight non-baseline levels of the three political attributes (abortion 3, LGBTQ policy 3,
guns 2).  The three valence attributes (cost of living, crime, schools) enter the regression as controls but no
gap is read on them: they carry no a priori group disagreement, and 6 more levels x 4 contrasts would mostly add
reference-stopped rows.  Per model: 8 levels x 4 contrasts = 32 contrasts, one family (family_size = rows).

Uncertainty.  Respondents (CHIP50 `respondent`, not the row id) are resampled with all their rows,
B = 1000, the same resample for the human and the silicon gap, so se_g and se_gamma are the bootstrap SDs and cov
is their bootstrap covariance (df = inf).  Outcome per resample is the weighted least-squares AMCE, which equals
the regression on the resampled data.

Control `human_split_half`.  Per model file, the human respondents are split in two halves (seed 20261003);
half A's human choices play the simulator, half B's the reference; each half is bootstrapped over its own
respondents (B = 1000) and the two are independent (cov = 0).
"""
import os

import numpy as np
import pandas as pd

NAME = "malmberg2026"
KIND = "language model silicon samples (conjoint) and a human split-half control"
SOURCE = ("Malmberg (2026), 'Investigating the Utility of LLM-Based Silicon Samples in Political Science Conjoint "
          "Experiments', Public Opinion Quarterly; Harvard Dataverse doi:10.7910/DVN/2LVYOO (CC0), "
          "https://doi.org/10.7910/DVN/2LVYOO")

RAW = r"C:\Research\PsychBench\UpdatedRun\paper_brm\external\raw\malmberg2026"
SEED = 20261003
B = 1000

MODELS = [  # label, file, silicon choice column
    ("GPT-4o-mini", "fullGPTzeroshotAMCEs.csv", "outzeroshotresults1"),
    ("Llama 3.2", "fulllamazeroshotAMCEs.csv", "choiceoutput"),
    ("Magistral-small-2506", "fullMistralsmall.csv", "choiceoutput"),
]

# attribute -> (label, F-1-p-<attr index>, baseline first, then the non-baseline levels in the authors' order)
ATTRS = [
    ("Abortion", 1, ["Abortion is banned after second trimester", "Abortion is banned after first trimester",
                     "Abortion is always banned", "Abortion is protected by state constitution"]),
    ("LGBTQ", 2, ["No LGBTQ legal protections, no restrictions on hormone replacement therapy",
                  "LGBTQ legal protections, no restrictions on hormone replacement therapy",
                  "No LGBTQ legal protections, only adults may seek hormone replacement therapy",
                  "No LGBTQ legal protections, full ban on hormone replacement therapy"]),
    ("Guns", 3, ["Assault rifle ban in place; firearm open carry allowed",
                 "Assault rifle ban in place; no firearm open carry allowed",
                 "No ban on assault rifles; firearm open carry allowed"]),
    ("COL", 4, ["National average", "20% below national average", "20% above national average"]),
    ("Crime", 5, ["National average", "20% below national average", "20% above national average"]),
    ("Schools", 6, ["National average", "20% below national average", "20% above national average"]),
]
# the 14 non-baseline dummies in the order of the authors' bootstrap files (Abortion, COL, Crime, Guns, LGBTQ,
# Schools by cjoint's alphabetical term order); our own order below is the attribute order of ATTRS
DUMMIES = [(a, lv) for a, _, lvs in ATTRS for lv in lvs[1:]]
SHORT = {
    "Abortion is banned after first trimester": "abortion: ban after 1st trimester",
    "Abortion is always banned": "abortion: always banned",
    "Abortion is protected by state constitution": "abortion: constitutionally protected",
    "LGBTQ legal protections, no restrictions on hormone replacement therapy": "LGBTQ: protections, no HRT limits",
    "No LGBTQ legal protections, only adults may seek hormone replacement therapy": "LGBTQ: no protections, HRT 18+",
    "No LGBTQ legal protections, full ban on hormone replacement therapy": "LGBTQ: no protections, HRT banned",
    "Assault rifle ban in place; no firearm open carry allowed": "guns: AR ban, no open carry",
    "No ban on assault rifles; firearm open carry allowed": "guns: no AR ban, open carry",
}
READ_LEVELS = [(a, lv) for a, lv in DUMMIES if a in ("Abortion", "LGBTQ", "Guns")]  # 8 levels
CONTRASTS = [("pid", "Republican", "Democrat"), ("gender", "Female", "Male"),
             ("race", "African American", "White"), ("age-cat2", "a young adult", "a senior citizen")]
P = 1 + len(DUMMIES)  # intercept + 14


def read_file(fn):
    return pd.read_csv(os.path.join(RAW, fn), skiprows=[1], low_memory=False)


def build(fn, choice_col):
    """Long design (two profile rows per task) and per-respondent sufficient statistics."""
    d = read_file(fn)
    assert ((d["selected-Profile-1"] == 1) == (d["which-profile-selected"] == 1)).all()
    n = len(d)
    X = np.zeros((2 * n, P))
    X[:, 0] = 1.0
    for p in (1, 2):
        rows = slice((p - 1) * n, p * n)
        for j, (a, lv) in enumerate(DUMMIES, start=1):
            ai = next(i for lab, i, _ in ATTRS if lab == a)
            X[rows, j] = (d[f"F-1-{p}-{ai}"].to_numpy() == lv).astype(float)
    yh = np.r_[(d["which-profile-selected"] == 1), (d["which-profile-selected"] == 2)].astype(float)
    ys = np.r_[(d[choice_col] == 1), (d[choice_col] == 2)].astype(float)
    assert set(d[choice_col].unique()) <= {1, 2} and set(d["which-profile-selected"].unique()) <= {1, 2}
    codes, uniq = pd.factorize(d["respondent"])
    code2 = np.r_[codes, codes]
    order = np.argsort(code2, kind="stable")
    Xo, yho, yso, co = X[order], yh[order], ys[order], code2[order]
    starts = np.r_[0, np.flatnonzero(np.diff(co)) + 1]
    Z = (Xo[:, :, None] * Xo[:, None, :]).reshape(len(Xo), P * P)
    S = np.add.reduceat(Z, starts, axis=0).reshape(-1, P, P)
    th = np.add.reduceat(Xo * yho[:, None], starts, axis=0)
    ts = np.add.reduceat(Xo * yso[:, None], starts, axis=0)
    nrow = np.add.reduceat(np.ones(len(Xo)), starts)  # profile rows per respondent
    R = len(uniq)
    cov = {}
    for c in ["pid", "gender", "race", "age-cat2"]:
        g = d.groupby("respondent", sort=False)[c].nunique()
        if (g > 1).any():
            raise ValueError(f"{c} varies within respondent in {fn}")
        cov[c] = d.groupby("respondent", sort=False)[c].first().reindex(uniq).to_numpy()
    return dict(S=S, th=th, ts=ts, R=R, nrow=nrow, cov=cov, n_tasks=n, ids=np.asarray(uniq))


def make_counts(rng, members, R, b=B):
    """b x R matrix of resample counts, drawing len(members) respondents with replacement from `members`."""
    draw = members[rng.integers(0, len(members), (b, len(members)))]
    flat = (np.arange(b)[:, None] * R + draw).ravel()
    return np.bincount(flat, minlength=b * R).reshape(b, R).astype(float)


def wls(S, t, counts):
    """AMCE vectors for each row of `counts` (resample weights over respondents)."""
    A = (counts @ S.reshape(len(S), P * P)).reshape(-1, P, P)
    rhs = (counts @ t)[:, :, None]
    try:
        return np.linalg.solve(A, rhs)[:, :, 0]
    except np.linalg.LinAlgError:
        return np.einsum("bij,bjk->bik", np.linalg.pinv(A), rhs)[:, :, 0]


def subgroup_betas(m, y, counts_all, mask):
    """(full-sample beta, bootstrap betas) for the respondents in `mask`; y is 'th' or 'ts'."""
    S, t = m["S"][mask], m[y][mask]
    beta0 = wls(S, t, np.ones((1, mask.sum())))[0]
    return beta0, wls(S, t, counts_all[:, mask])


def level_index(a, lv):
    return 1 + DUMMIES.index((a, lv))


def gap_rows(model, fam, bet_sim, bet_ref, n_info, cov_zero=False):
    """One row per (read level x contrast).  bet_sim / bet_ref: {group value: (beta0, betas)}."""
    rows = []
    for attr, a, b in CONTRASTS:
        for (la, lv) in READ_LEVELS:
            j = level_index(la, lv)
            g0 = bet_sim[(attr, a)][0][j] - bet_sim[(attr, b)][0][j]
            p0 = bet_ref[(attr, a)][0][j] - bet_ref[(attr, b)][0][j]
            bg = bet_sim[(attr, a)][1][:, j] - bet_sim[(attr, b)][1][:, j]
            bp = bet_ref[(attr, a)][1][:, j] - bet_ref[(attr, b)][1][:, j]
            c = 0.0 if cov_zero else float(np.cov(bg, bp, ddof=1)[0, 1])
            rows.append(dict(family=fam, contrast=f"{SHORT[lv]} | {attr}: {a} - {b}", model=model,
                             level=SHORT[lv], contrast_attr=attr, group_a=a, group_b=b,
                             primary=bool(attr == "pid"),
                             g=float(g0), se_g=float(bg.std(ddof=1)), gamma=float(p0),
                             se_gamma=float(bp.std(ddof=1)), cov=c,
                             n_resp_a=n_info[(attr, a)], n_resp_b=n_info[(attr, b)]))
    return rows


def load():
    all_rows, inputs = [], []
    for k, (label, fn, ch) in enumerate(MODELS):
        inputs.append(os.path.join(RAW, fn))
        m = build(fn, ch)
        R = m["R"]
        masks = {(attr, v): (m["cov"][attr] == v) for attr, a, b in CONTRASTS for v in (a, b)}
        n_info = {k2: int(v.sum()) for k2, v in masks.items()}

        # --- model vs matched live respondents: same resamples for both sides
        rng = np.random.default_rng([SEED, k])
        counts = make_counts(rng, np.arange(R), R)
        bs = {gk: subgroup_betas(m, "ts", counts, mk) for gk, mk in masks.items()}
        bh = {gk: subgroup_betas(m, "th", counts, mk) for gk, mk in masks.items()}
        rows = gap_rows(label, label, bs, bh, n_info)
        all_rows += rows

        # --- ideal-simulator control: human half A as the simulator against human half B
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(R)
        half = R // 2
        A_idx, B_idx = np.sort(perm[:half]), np.sort(perm[half:])
        rngA, rngB = np.random.default_rng([SEED, 100 + k]), np.random.default_rng([SEED, 200 + k])
        cA, cB = make_counts(rngA, A_idx, R), make_counts(rngB, B_idx, R)
        maskA = {gk: mk & np.isin(np.arange(R), A_idx) for gk, mk in masks.items()}
        maskB = {gk: mk & np.isin(np.arange(R), B_idx) for gk, mk in masks.items()}
        bA = {gk: subgroup_betas(m, "th", cA, mk) for gk, mk in maskA.items()}
        bB = {gk: subgroup_betas(m, "th", cB, mk) for gk, mk in maskB.items()}
        n_half = {gk: f"{int(maskA[gk].sum())}/{int(maskB[gk].sum())}" for gk in masks}
        rows = gap_rows(label, f"human_split_half: {label} file", bA, bB, n_half, cov_zero=True)
        all_rows += rows
    gaps = pd.DataFrame(all_rows)
    gaps["family_size"] = gaps.groupby("family")["contrast"].transform("size")
    return dict(mode="gaps", gaps=gaps, inputs=inputs,
                notes=f"B={B} respondent bootstrap, seed {SEED}; split-half seed {SEED}; "
                      "n_resp_* in the split-half rows read 'half A/half B'.")
