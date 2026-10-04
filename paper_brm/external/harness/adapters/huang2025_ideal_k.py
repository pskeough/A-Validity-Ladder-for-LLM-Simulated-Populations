"""Matched-design ideal simulator (sensitivity to huang2025_ideal): the same fresh resample of the real
respondents of each group, but drawn at the size the 100-profile LLM design gives that group in that
question (the number of the first 100 synthetic profiles in the group; questions with no profile in
either group of a contrast drop out, as they do for the LLMs). This carries the same finite-profile
noise as the LLM families and so is the stricter control. Families rep_000 ... rep_099.
"""
from . import huang2025_ideal as base

NAME = "huang2025_ideal_k"
KIND = "ideal simulator (fresh resample of the real OpinionQA respondents, group sizes of the 100-profile design)"
SOURCE = base.SOURCE


def load():
    gaps = base.make(100)
    return dict(mode="gaps", gaps=gaps, inputs=base.hu.inputs()[:1],
                notes=f"ideal simulator at the 100-profile design; {base.REPS} replicate families; family_size {base.FAMILY_SIZE}")
