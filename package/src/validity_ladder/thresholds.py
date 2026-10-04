"""Frozen thresholds of the Validity Ladder (paper_brm/LADDER_SPEC.md, version 1.0, frozen 2026-10-02).

Every function in the package takes these as keyword defaults, so a user can change any of them for
a sensitivity analysis. A verdict produced with a changed threshold is not the frozen ladder's
verdict and should be reported as a sensitivity.
"""

# Reading rule common to every rung: intervals are 90% unless a rung says otherwise.
INTERVAL_LEVEL = 0.90

# ---------------------------------------------------------------------------------------------- gate
GATE_SE_MIN_SD = 0.25        # minimum rule: SE(k) <= 0.25 reference SD in every framing
GATE_SE_REC_SD = 0.125       # recommended rule: SE(k) <= 0.125 reference SD
GATE_PHI_MIN = 0.80          # separation phi(k), required only for persona-comparison uses
GATE_PHI_REC = 0.90
GATE_N_BOOT = 1000           # percentile bootstrap over personas

# -------------------------------------------------------------------------------------------- level 1
L1_TAUS = (1.50, 2.00, 3.00)  # candidate tolerances for the tail ratios
L1_TAU_FLOOR = 1.50           # real respondents drawn through the persona design pass at 1.5, not 1.25
L1_TAIL_Q = 0.05              # tails: below the within-total 5th and above the 95th percentile of lz*
L1_MIN_STRATUM_N = 100        # totals pooled into strata of at least 100 reference respondents
L1_MIN_GROUP_N = 100          # reference subgroups used for tau need n >= 100
L1_N_BOOT = 2000

# -------------------------------------------------------------------------------------------- level 2
L2_BOUNDS = (-0.25, 0.25, 0.75, 1.25)
L2_LABELS = ("reversed", "missing", "attenuated", "kept", "steepened")
L2_ALPHA = 0.05               # one-sided, divided by the family size (Bonferroni)
L2_STOPS = ("no population gap", "reference too imprecise")

# -------------------------------------------------------------------------------------------- level 3
L3_TOLERANCE_SD = 0.5         # default tolerance, in reference SD units (2 points on the PHQ-8)
L3_REPORT_SD = (0.2,)         # tolerances reported beside the default
L3_ALPHA = 0.05               # TOST: 90% interval inside (-delta, +delta)
L3_PREV_BANDS = (2.5, 5.0)    # prevalence difference bands, percentage points
L3_RATIO_BAND = (0.80, 1.25)  # prevalence ratio band (Fieller)

# -------------------------------------------------------------------------------------------- level 4
R1_LOAD = 0.30                # every loading >= .30 (or the reference's own value where lower)
R1_RATIO = 3.0                # lambda1 / lambda2 >= 3 (or the reference's own value where lower)
R2_PHI = 0.95                 # Tucker congruence lower 90% limit >= .95
R2_RMSD = 0.10                # RMS loading difference upper 90% limit <= .10
L4_N_BOOT = 1000
