"""Validation: OpinionQA headline rows as stored by scripts/78e."""
from adapters._stored import external

NAME = "val_opinionqa"
KIND = "language model (validation of the harness against scripts/78e)"
SOURCE = "Santurkar et al. (2023), OpinionQA; rows from analysis/brm/l2_external_r3.csv, config headline"


def load():
    g, inputs = external("OpinionQA")
    return dict(mode="gaps", gaps=g, inputs=inputs)
