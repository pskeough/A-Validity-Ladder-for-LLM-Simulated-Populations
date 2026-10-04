"""Null control on the Huang et al. (2025) OpinionQA data: the authors' own random baseline
(clean/random.json, answers drawn uniformly from {-1, -1/3, 0, 1/3, 1}, 200 per question, independent of
the profile). Read exactly as the LLMs in huang2025 (the zeros are Refused-coded and dropped). It should
read as a model with no demographic signal. Not an LLM; kept out of huang2025 so those counts are
LLMs only.
"""
from . import _huang as hu

NAME = "huang2025_random"
KIND = "null control (random answers, authors' random.json)"
SOURCE = "Huang, Wu & Wang (2025), github.com/yw3453/uq-llm-survey-simulation data.zip, clean/random.json"


def load():
    S, P, A = hu.tables()
    H = hu.human_cells(S)
    M = hu.llm_cells(P, A["random"])
    gaps = hu.with_family_size(hu.build_gaps(H, M, family="random", extra=dict(model="random", n_profiles=200)))
    return dict(mode="gaps", gaps=gaps, inputs=hu.inputs()[:1] + [hu.os.path.join(hu.CLEAN, "random.json")])
