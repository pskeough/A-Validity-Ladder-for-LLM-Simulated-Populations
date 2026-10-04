"""Quick start: run every rung of the Validity Ladder on synthetic data.

    python examples/quickstart.py

The synthetic simulated data hold two models. "faithful" draws personas from the reference's own
latent distribution; "flat" ignores the persona attributes and compresses the spread. With 12
personas per cell the faithful model's level-2 intervals are wide, so its level-2 readings are
mostly unresolved and one seed can give a chance "not kept".
"""
import pandas as pd

import validity_ladder as vl
from validity_ladder.synthetic import simulate_example

sim, ref = simulate_example(seed=1)
items = [c for c in sim.columns if c.startswith("item_")]

table = vl.run_ladder(
    sim, ref,
    persona="persona", items=items, by=["model", "framing"],
    ref_weight="w",
    contrasts=[("sex", "F", "M", "Women minus Men"), ("income", "Low", "High", "Low minus High income")],
    options={"level1": {"n_boot": 200}, "level4": {"n_boot": 200}},   # 2,000 and 1,000 by default
)

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)
print(table[["model", "framing", "rung", "verdict", "summary"]].to_string(index=False))

# One rung on its own: the gate for one model and framing.
one = sim[(sim.model == "faithful") & (sim.framing == "plain")]
g = vl.gate(one[items].sum(axis=1), one.persona, sd_ref=float(ref[items].sum(axis=1).std()), k=20)
print(f"\ngate, faithful / plain: SE(20) = {g['se_k']:.3f}, tolerance {g['tol_min']:.3f}, "
      f"phi(20) = {g['phi_k']:.2f}, verdict {g['verdict']} ({g['rule_met']} rule)")

# One level-2 contrast from summary numbers: g, its SE and df, gamma, its SE and design df.
c = vl.level2_contrast(0.94, 0.15, 0.93, 0.06, df_g=47, df_gamma=109, family_size=7)
print(f"level-2 contrast: ratio {c['ratio']:.2f}, {100 * c['interval_level']:.1f}% interval "
      f"[{c['ci_lo']:.2f}, {c['ci_hi']:.2f}], verdict {c['verdict']}, label {c['label']}")
