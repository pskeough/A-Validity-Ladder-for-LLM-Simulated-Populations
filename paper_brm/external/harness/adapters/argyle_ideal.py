"""Ideal-simulator control on Argyle et al. (2023) Study 3 (ANES 2016 respondents, 9 outcomes x 7
contrasts, family of 61 as in the frozen reading).

The simulator is group-faithful: each respondent's simulated answer is the real answer of a
respondent drawn at random (with replacement) from the same contrast group, so every group gap is
reproduced in expectation and nothing else is. The respondents, outcomes, codings and contrasts are
those of external/scripts/argyle_ladder.py (its definitions are executed from the script itself, up
to the point where it starts running). SEs and the human-silicon covariance are the analytic
versions of that script's paired respondent bootstrap: var(mean) = var / n within each group, cov =
within-group sample covariance / n. df = inf.

Families: rep_000 ... rep_199 (seeds 0..199). Receipt: the human gaps equal the paper's stored
human_gap for the main run (checked in load()).
"""
import os

import numpy as np
import pandas as pd

NAME = "argyle_ideal"
KIND = "ideal simulator (group-faithful resample of the real ANES answers)"
SOURCE = "Argyle et al. (2023) Study 3, Harvard Dataverse doi:10.7910/DVN/JPV20K, anesgpt3_task3.csv"

EXT = r"C:\Research\PsychBench\UpdatedRun\paper_brm\external"
SCRIPT = os.path.join(EXT, "scripts", "argyle_ladder.py")
DATA = os.path.join(EXT, "raw", "argyle2023", "anesgpt3_task3.csv")
STORED = os.path.join(EXT, "results", "argyle", "level2_contrasts.csv")
REPS = 200
FAMILY_SIZE = 61


def _definitions():
    src = open(SCRIPT, encoding="utf-8").read()
    cut = src.index("rng = np.random.default_rng(20260924)")
    ns = {"__file__": SCRIPT}
    exec(compile(src[:cut], SCRIPT, "exec"), ns)
    return ns


def load():
    ns = _definitions()
    o = ns["clean"](DATA)
    stored = pd.read_csv(STORED)
    stored = stored[stored.run == "t0.7_main"]
    cells = []
    for out, (label, _, _) in ns["OUTCOMES"].items():
        h, s = f"{out}_h", f"{out}_s"
        for gv, a, b in ns["CONTRASTS"]:
            if out == "pid7" and gv == "party":
                continue
            sub = o[(o[gv].isin([a, b])) & o[h].notna() & o[s].notna()]
            A, B = sub[sub[gv] == a][h].to_numpy(float), sub[sub[gv] == b][h].to_numpy(float)
            if len(A) < 20 or len(B) < 20:
                continue
            ref = stored[(stored.outcome == label) & (stored.contrast == f"{a} - {b}")]
            assert len(ref) == 1 and np.isclose(ref.human_gap.iloc[0], A.mean() - B.mean(), atol=1e-12), (label, a, b)
            cells.append((label, f"{a} - {b}", A, B))
    assert len(cells) == FAMILY_SIZE, len(cells)
    rows = []
    for rep in range(REPS):
        rng = np.random.default_rng(rep)
        for label, con, A, B in cells:
            sA, sB = A[rng.integers(0, len(A), len(A))], B[rng.integers(0, len(B), len(B))]
            nA, nB = len(A), len(B)
            cov = np.cov(sA, A, ddof=1)[0, 1] / nA + np.cov(sB, B, ddof=1)[0, 1] / nB
            rows.append(dict(family=f"rep_{rep:03d}", outcome=label, contrast=con, n_a=nA, n_b=nB,
                             g=sA.mean() - sB.mean(),
                             se_g=np.sqrt(sA.var(ddof=1) / nA + sB.var(ddof=1) / nB),
                             gamma=A.mean() - B.mean(),
                             se_gamma=np.sqrt(A.var(ddof=1) / nA + B.var(ddof=1) / nB),
                             cov=cov, family_size=FAMILY_SIZE))
    return dict(mode="gaps", gaps=pd.DataFrame(rows), inputs=[SCRIPT, DATA, STORED],
                notes="group-faithful ideal simulator; analytic paired SEs; family 61")
