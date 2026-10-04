"""Numbers the article prints for the IPIP-50 release (levels 1 and 4), computed from the raw release and
the ladder outputs in out/, so each has a receipt. Writes out/p23_paper_numbers.csv."""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
RAW = r"C:\Research\PsychBench\UpdatedRun\paper_brm\external\raw\ipip_audit"

l1 = pd.read_csv(os.path.join(OUT, "a10_l1.csv"))
l4 = pd.read_csv(os.path.join(OUT, "a10_l4.csv"))
c1 = pd.read_csv(os.path.join(OUT, "p12_control_l1.csv"))
c4 = pd.read_csv(os.path.join(OUT, "p12_control_l4.csv"))
desc = pd.read_csv(os.path.join(OUT, "a10_desc.csv"))
n_human = sum(1 for _ in open(os.path.join(RAW, "human_data_cleaned.csv"), encoding="utf-8")) - 1
c1f = c1[c1.tau_rule == "frozen"]

rows = [
    ("models", l1.model.nunique()),
    ("scales", l1.scale.nunique()),
    ("cells", len(l1)),
    ("draws_per_model", int(desc.n.max())),
    ("human_respondents", n_human),
    ("l1_fail_cells", int(l1.verdict.str.startswith("fail").sum())),
    ("l1_misfit_deficit_cells", int(l1.misfit_verdict.eq("misfit deficit").sum())),
    ("l1_compressed_cells", int(l1.verdict.str.contains("compressed").sum())),
    ("r1_pass_cells", int(l4.R1.sum())),
    ("r2_fail_cells", int(l4.R2.eq("fail").sum())),
    ("closest_r2_rmsd_lo90", round(float(l4.loading_rmsd_lo90.min()), 3)),
    ("control_n", int(c4.n.max())),
    ("control_l1_pass_scales", int(c1f.verdict.eq("pass").sum())),
    ("control_misfit_ratio_min", round(float(c1f.misfit_ratio.min()), 3)),
    ("control_misfit_ratio_max", round(float(c1f.misfit_ratio.max()), 3)),
    ("control_r1_pass_scales", int(c4.R1.sum())),
    ("control_r2_pass_scales", int(c4.R2.eq("pass").sum())),
]
pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(os.path.join(OUT, "p23_paper_numbers.csv"), index=False)
print(pd.DataFrame(rows, columns=["quantity", "value"]).to_string())
