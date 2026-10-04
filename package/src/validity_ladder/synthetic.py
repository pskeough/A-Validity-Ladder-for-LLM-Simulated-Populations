"""Synthetic data in the shape run_ladder expects, for examples and tests.

Answers come from a graded response model. The reference is a sample of respondents with two
attributes (sex, income); the simulated data are personas crossed over the same attributes, with
repeated draws. Model "faithful" draws its personas from the reference's latent distribution;
model "flat" ignores the attributes and compresses the spread, so it should fail several rungs.
"""
import numpy as np
import pandas as pd


def _grm_answers(theta, a, b, rng):
    """Answers 0..M-1 for latent values theta under a GRM with slopes a (J,) and thresholds b (J, M-1)."""
    p_ge = 1 / (1 + np.exp(-a[None, :, None] * (theta[:, None, None] - b[None, :, :])))   # P(X >= k)
    u = rng.random((len(theta), len(a)))
    return (u[:, :, None] < p_ge).sum(axis=2)


def simulate_example(n_ref=6000, personas_per_cell=12, draws=20, n_items=6, seed=0):
    """Return (sim, ref) DataFrames.

    sim: columns model, framing, persona, draw, sex, income, item_1..item_J.
    ref: columns sex, income, w (weights), item_1..item_J.
    """
    rng = np.random.default_rng(seed)
    J, M = n_items, 4
    a = np.linspace(1.2, 2.2, J)
    b = np.column_stack([np.linspace(-0.2, 0.4, J), np.linspace(0.8, 1.4, J), np.linspace(1.8, 2.4, J)])
    items = [f"item_{j + 1}" for j in range(J)]

    def mu(sex, income):
        return 0.5 * (np.asarray(sex) == "F") + 0.8 * (np.asarray(income) == "Low") - 0.6

    sex = rng.choice(["F", "M"], n_ref)
    income = rng.choice(["Low", "High"], n_ref, p=[0.35, 0.65])
    theta = mu(sex, income) + rng.normal(0, 1, n_ref)
    ref = pd.DataFrame(dict(sex=sex, income=income, w=rng.uniform(0.5, 1.5, n_ref)))
    ref[items] = _grm_answers(theta, a, b, rng)

    rows = []
    for model in ("faithful", "flat"):
        for framing in ("plain", "story"):
            pid = 0
            for s in ("F", "M"):
                for inc in ("Low", "High"):
                    for _ in range(personas_per_cell):
                        pid += 1
                        if model == "faithful":
                            tp = mu(s, inc) + rng.normal(0, 0.8)
                            th = tp + rng.normal(0, 0.6, draws)
                        else:
                            tp = -0.6 + rng.normal(0, 0.1)
                            th = tp + rng.normal(0, 0.2, draws)
                        X = _grm_answers(th, a, b, rng)
                        for d in range(draws):
                            rows.append(dict(model=model, framing=framing, persona=f"p{pid:03d}", draw=d + 1,
                                             sex=s, income=inc, **dict(zip(items, X[d]))))
    sim = pd.DataFrame(rows)
    return sim, ref
