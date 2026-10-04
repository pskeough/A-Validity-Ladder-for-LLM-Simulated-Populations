"""Huang, Wu & Wang (2025) OpinionQA synthetic-profile simulation, read at level 2.

385 OpinionQA questions (4 ordered options + Refused), 8 LLMs, 100-200 synthetic profiles per question
and model (profiles bootstrapped from the real respondent profiles; every profile carries region,
citizenship, sex, age, marital status, race, education, income, religion, attendance, party and
ideology, all written into the prompt). Family = LLM. Nine contrasts fixed in adapters/_huang.py
before any ratio was computed. Scalar and gap construction: see _huang.py.
"""
from . import _huang as hu

NAME = "huang2025"
KIND = "language model (profile-prompted survey simulation)"
SOURCE = ("Huang, Wu & Wang (2025), How many human survey respondents is a LLM worth? ICML 2025, "
          "arXiv:2502.17773; github.com/yw3453/uq-llm-survey-simulation (data.zip, MIT licence)")


def load():
    S, P, A = hu.tables()
    H = hu.human_cells(S)
    parts = []
    for m in hu.LLMS:
        M = hu.llm_cells(P, A[m])
        g = hu.build_gaps(H, M, family=m, extra=dict(model=m, n_profiles=len(next(iter(A[m].values())))))
        parts.append(g)
    import pandas as pd
    gaps = hu.with_family_size(pd.concat(parts, ignore_index=True))
    return dict(mode="gaps", gaps=gaps, inputs=hu.inputs(),
                notes=("9 fixed contrasts; Refused dropped on both sides; unweighted (no weights in the released "
                       "records); family_size = contrasts with >= 20 questions per LLM; first len(answers) "
                       "profiles per question matched to answers by position"))
