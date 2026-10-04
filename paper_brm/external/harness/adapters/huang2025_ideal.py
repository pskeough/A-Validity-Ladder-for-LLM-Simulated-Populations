"""Ideal-simulator control on the Huang et al. (2025) OpinionQA data (same questions, contrasts and
reading as huang2025).

The simulator is a fresh sample of the same people: for each question and group, the human respondents
of that group are resampled with replacement at the group's own human size (a multinomial draw over
the four options, which is the same thing), read exactly as the LLMs are (_huang.build_gaps). The
human gap gamma is the original one. Families rep_000 ... rep_099 (rng seeds 0..99). family_size is
the number of contrasts in the LLM families (9, checked against huang2025/l2_models.csv).

huang2025_ideal_k (module huang2025_ideal_k) draws the group sizes of the 100-profile LLM design instead.
"""
import numpy as np
import pandas as pd

from . import _huang as hu

NAME = "huang2025_ideal"
KIND = "ideal simulator (fresh resample of the real OpinionQA respondents, human group sizes)"
SOURCE = "Huang, Wu & Wang (2025), github.com/yw3453/uq-llm-survey-simulation data.zip, opinionqa_data.json"
REPS = 100
FAMILY_SIZE = len(hu.CONTRASTS)
SIZE_FROM_PROFILES = None  # overridden by huang2025_ideal_k


def make(size_from_profiles):
    S, P, A = hu.tables()
    H = hu.human_cells(S)
    sizes = hu.profile_cells(P, size_from_profiles) if size_from_profiles else None
    parts = []
    for rep in range(REPS):
        rng = np.random.default_rng(rep)
        M = hu.resample_cells(H, rng, size_from=sizes)
        parts.append(hu.build_gaps(H, M, family=f"rep_{rep:03d}"))
    gaps = pd.concat(parts, ignore_index=True)
    return hu.with_family_size(gaps, FAMILY_SIZE)


def load():
    gaps = make(SIZE_FROM_PROFILES)
    return dict(mode="gaps", gaps=gaps, inputs=[hu.OQA],
                notes=f"ideal simulator; {REPS} replicate families; family_size {FAMILY_SIZE}")
