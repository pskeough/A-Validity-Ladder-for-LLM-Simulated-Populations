"""Validation of the long mode: the package's synthetic example run end to end."""
from validity_ladder.synthetic import simulate_example

NAME = "val_synthetic"
KIND = "synthetic (long-mode check)"
SOURCE = "validity_ladder.synthetic.simulate_example(seed=1)"


def load():
    sim, ref = simulate_example(seed=1)
    items = [c for c in sim.columns if c.startswith("item_")]
    kw = dict(persona="persona", items=items, by=["model", "framing"], ref_weight="w",
              contrasts=[("sex", "F", "M"), ("income", "Low", "High")], options={"level1": {"n_boot": 200}})
    return dict(mode="long", sim=sim, ref=ref, kwargs=kw)
