"""Validation: Twin-2K-500 level-2 rows (all twin specifications and the wave 1-3 retest) as
stored by scripts/93."""
import pandas as pd

from adapters._stored import twin

NAME = "val_twin2k"
KIND = "digital twins and human retest (validation of the harness against scripts/93)"
SOURCE = "Toubia et al. (2025), Twin-2K-500; rows from paper_brm/external/results/twin2k/level2_contrasts.csv"


def load():
    a, inputs = twin("headline")
    b, _ = twin("ref_wave1_3")
    return dict(mode="gaps", gaps=pd.concat([a, b], ignore_index=True), inputs=inputs)
