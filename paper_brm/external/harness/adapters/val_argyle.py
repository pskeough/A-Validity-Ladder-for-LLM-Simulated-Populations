"""Validation: Argyle et al. Study 3 headline rows as stored by scripts/78e."""
from adapters._stored import external

NAME = "val_argyle"
KIND = "language model (validation of the harness against scripts/78e)"
SOURCE = "Argyle et al. (2023), Study 3; rows from analysis/brm/l2_external_r3.csv, config headline"


def load():
    g, inputs = external("Argyle")
    return dict(mode="gaps", gaps=g, inputs=inputs)
