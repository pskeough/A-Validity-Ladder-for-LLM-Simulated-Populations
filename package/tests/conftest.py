"""Shared paths for the tests.

tests/data holds verbatim row subsets of the repository's stored outputs (see
tests/data/make_fixtures.py), so the reproduction tests that need only those run anywhere. Tests
that need larger stored files (the corpus, the NHANES-derived frames written by the analysis
scripts) look for them in the repository that holds this package and skip when absent. No test
downloads anything.
"""
import os

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BRM = os.path.join(REPO, "analysis", "brm")


def fixture(name):
    return pd.read_csv(os.path.join(DATA, name), low_memory=False)


def repo_file(*parts):
    p = os.path.join(REPO, *parts)
    if not os.path.exists(p):
        pytest.skip(f"stored output not present: {os.path.join(*parts)}")
    return p


PHQ_ITEMS = [f"phq8_{i}" for i in range(1, 9)]
DPQ = [f"DPQ0{i}0" for i in range(1, 9)]
SHORT = {"deepseek/deepseek-chat-v3": "DeepSeek-V3", "google/gemini-3-flash-preview": "Gemini-3-Flash",
         "openai/gpt-4o-mini": "GPT-4o-mini", "z-ai/glm-4.7": "GLM-4.7"}
