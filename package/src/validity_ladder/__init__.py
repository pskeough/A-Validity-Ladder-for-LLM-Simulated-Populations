"""validity_ladder: the Validity Ladder for LLM-simulated populations.

Rungs, each read per model and per framing:
  gate     precision of a persona mean (gate)
  level 1  individual coherence: lz* tail ratios against the reference (level1)
  level 2  subgroup fidelity: ratio of simulated to population gaps (level2, level2_contrast)
  level 3  population calibration: post-stratified residual with TOST (level3)
  level 4  structural fidelity: general factor, loading congruence and size (level4)
run_ladder runs them all on a long simulated dataset and a reference.

The frozen rules are in paper_brm/LADDER_SPEC.md (version 1.0); the defaults in `thresholds` are
those rules.
"""
from . import thresholds
from .gate import gate, k_for_phi, k_for_se, one_facet, phi_k
from .ladder import RUNGS, run_ladder
from .level1 import L1Reference, level1, read_verdict, resolved_departure, tau_from_intervals, tau_from_subgroups
from .level2 import (certifiable, faithful_simulation, level2, level2_contrast, level2_model_reading,
                     level2_model_verdict, population_gap, simulated_gap, three_way)
from .level3 import level3, level3_from_cells, level3_model_verdict, level3_tost, prevalence_ratio
from .level4 import (L4Reference, congruence, general_factor_vs_ref, level4, level4_verdict, one_factor_uls,
                     polychoric_matrix, r2_verdict)

__version__ = "0.1.1"

__all__ = [
    "gate", "level1", "level2", "level2_contrast", "level3", "level4", "run_ladder",
    "L1Reference", "L4Reference", "thresholds", "RUNGS",
    "one_facet", "phi_k", "k_for_se", "k_for_phi",
    "read_verdict", "resolved_departure", "tau_from_subgroups", "tau_from_intervals",
    "level2_model_verdict", "level2_model_reading", "certifiable", "faithful_simulation", "simulated_gap", "population_gap", "three_way",
    "level3_tost", "level3_from_cells", "level3_model_verdict", "prevalence_ratio",
    "polychoric_matrix", "one_factor_uls", "congruence", "general_factor_vs_ref", "r2_verdict",
    "level4_verdict",
]
