# validity-ladder

Python implementation of the Validity Ladder for LLM-simulated populations: a precision gate and four
levels of validity checks for survey or questionnaire answers produced by a language model that is
asked to answer as a set of personas. Each rung returns pass, fail or unresolved, or a named stop
when the reference cannot support a verdict.

The rules, thresholds and stops are frozen in `paper_brm/LADDER_SPEC.md` (version 1.0, 2 Oct 2026)
in the paper repository. The package defaults are those rules; every threshold is a keyword
argument, and a changed threshold gives a sensitivity analysis, not the frozen verdict.

## Install

```
pip install git+https://github.com/pskeough/A-Validity-Ladder-for-LLM-Simulated-Populations#subdirectory=package
```

Python 3.10 or later. Dependencies: numpy, scipy, pandas.

## Quick start

```python
import validity_ladder as vl
from validity_ladder.synthetic import simulate_example

sim, ref = simulate_example(seed=1)          # sim: one row per draw; ref: one row per respondent
items = [c for c in sim.columns if c.startswith("item_")]

table = vl.run_ladder(
    sim, ref,
    persona="persona", items=items, by=["model", "framing"],
    ref_weight="w",                            # also ref_psu=, ref_stratum= for a survey design
    contrasts=[("sex", "F", "M"), ("income", "Low", "High")],
)
print(table[["model", "framing", "rung", "verdict", "summary"]])
```

`examples/quickstart.py` runs this and two single-rung calls. The simulated data need a persona
column, item columns scored 0 to M-1 after reverse keying (or a score column), and persona
attribute columns with the same labels as the reference. `run_ladder` reads every rung per model
and framing (`by`), skips a rung whose inputs are missing and records the reason.

## Rungs

| Rung | Question | Statistic | Pass | Fail | Function |
|---|---|---|---|---|---|
| Gate | Is a k-draw persona mean precise enough to read? | One-facet G-study; SE(k) = sqrt(s2_r / k); phi(k) reported | SE(k) <= 0.25 reference SD (recommended 0.125) | SE(k) above 0.25 SD | `gate` |
| Level 1 | Are the answer vectors as coherent as people's at the same total? | lz* under a GRM fitted to the reference; misfit and overfit ratios against the within-total 5th and 95th percentiles | Both ratios' 90% intervals inside [1/tau, tau]; tau from reference subgroups, floor 1.5 | Either tail wholly outside; both below = compressed | `level1`, `L1Reference`, `tau_from_subgroups` |
| Level 2 | Are the population's group gaps reproduced at their size? | rho = g / gamma, standardised gaps, Fieller interval, Bonferroni over the model's contrasts | Every contrast not stopped is kept (0.75 to 1.25) | Any contrast whose verdict excludes kept | `level2`, `level2_contrast` |
| Level 3 | Are the population's means reproduced, overall and per group? | Post-stratified residual, delete-one-PSU jackknife, TOST at alpha .05 | Every group's 90% interval inside +/- delta (default 0.5 reference SD) | Any interval wholly outside | `level3`, `level3_from_cells` |
| Level 4 | Is the factor structure the population's? | Polychorics, one-factor ULS; R1 general factor, R2 congruence and RMS loading difference, R3 diagnostic | R1, R2 and R4 pass | R1, R2 or R4 fails | `level4`, `L4Reference` |

Step 0, certifiability (amendment of 3 Oct 2026): `certifiable(gamma, se_gamma, family_size=...)` says
whether the reference can certify a contrast at all, that is, whether a noise-free faithful simulator
would be read kept. A model family read unresolved with no certifiable contrast is reported as
"reference cannot certify" (`level2_model_reading`; `run_ladder` and `level2` report it). Before
generating data, `faithful_simulation(gamma, se_gamma, se_g, ...)` estimates how often a faithful
simulator would pass at a planned design, so a study can be sized to certify the gaps it cares about.

Level-2 stops: "no population gap" (gamma's interval covers 0) and "reference too imprecise"
(k = t SE_gamma / |gamma| > 1/4, unless the reading already excludes kept). Each contrast also gets
a label: kept, not kept, unresolved or not read.

What a verdict supports and rules out is stated in each function's docstring.

## Not in this package (version 0.1.1)

- R4 of level 4 (multi-group CFA invariance across attributes). Pass its verdict through
  `level4(..., r4=...)` or `run_ladder(..., r4=...)`; without it level 4 can fail or be unresolved
  but cannot pass. Use `r4="not applicable"` when the personas carry no attribute the population
  records.
- The level-2 draw-only SE sensitivity and the Rao-Wu bootstrap SE of gamma (the Taylor SE, the
  primary, is implemented).
- The level-3 design-standardised and marginal estimands (the post-stratified estimand, the frozen
  one, is implemented). The prevalence table is reported beside the level-3 verdict and does not
  enter it.
- The "in every framing" aggregation of the gate and the model-level summaries: `run_ladder` returns
  one row per model and framing.

One extension is not in the paper's scripts: when a level-2 stratum holds several personas on a
side, the simulated gap's SE is computed from the persona means within strata (Welch-Satterthwaite
df). With one persona per stratum side, as in the PHQ-8 design, the persona-pair rule of the paper
is used.

## Changes

- 0.1.1 (3 Oct 2026): step 0 added (`certifiable`, `level2_model_reading`, `faithful_simulation`;
  level 2 now also reports "reference cannot certify"). `level2_contrast` reads a contrast whose simulated gap has no sampling
  error (se_g = 0, for example a model that answers identically in both groups) with the
  closed-form interval, as the Twin-2K-500 script does. 0.1.0 returned no interval and the label
  "unresolved" for these contrasts; the verdict was already correct. A population gap of exactly
  0 passed as a Python float now gives the stop "no population gap"; 0.1.0 raised
  ZeroDivisionError.

## Tests

```
pip install -e "package[test]"
pytest package/tests
```

The tests reproduce the paper's stored outputs: the gate's variance components, phi(30) and k for
each model and framing; the GRM, lz* and level-1 tail ratios and verdicts; every level-2 interval
and verdict of the headline table and of the OpinionQA, Argyle and Bisbee releases; the level-3
post-stratified residuals, intervals and verdicts; and the level-4 loadings and R1 verdicts. Tests
that need the corpus or the NHANES-derived frames written by the analysis scripts skip when those
files are absent. Nothing is downloaded.

## Citation

Keough, P. S. A Validity Ladder for LLM-Simulated Populations. Code:
https://github.com/pskeough/A-Validity-Ladder-for-LLM-Simulated-Populations

## Licence

MIT
