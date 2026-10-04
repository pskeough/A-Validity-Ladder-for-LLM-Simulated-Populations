"""Cross-release level-2 summary from every harness result folder. The validation folders (val_*)
carry the paper's own model readings of OpinionQA, Argyle and Twin-2K-500 and enter as those datasets.

One row per dataset and arm (models / ideal or split-half control / random): families, family verdicts
(four-way reading, step 0) and contrast labels. Arm comes from the folder name (_ideal, _splithalf, _random) and, inside a folder,
from family names starting with "human_split_half".
Output: results/CROSS_RELEASE.csv
"""
import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["kept", "not kept", "unresolved", "not read"]


def arm_of(folder, family):
    if family.startswith("human_split_half"):
        return "split-half control"
    if family.startswith("human_retest"):
        return "human retest control"
    if family.startswith("subpop :"):
        return "survey-trained simulator"
    for tag, arm in (("_ideal", "ideal control"), ("_splithalf", "split-half control"), ("_random", "random")):
        if tag in folder:
            return arm + (" (matched size)" if folder.endswith("_k") else "")
    return "models"


rows = []
for f in sorted(glob.glob(os.path.join(HERE, "results", "*", "l2_models.csv"))):
    folder = os.path.basename(os.path.dirname(f))
    if folder.startswith("val_synthetic"):
        continue
    m = pd.read_csv(f)
    m["arm"] = [arm_of(folder, fam) for fam in m.family]
    base = folder.replace("val_", "")
    for suffix in ("_ideal_k", "_ideal", "_splithalf", "_random"):
        base = base.replace(suffix, "")
    for arm, d in m.groupby("arm"):
        r = dict(dataset=base, folder=folder, arm=arm, families=len(d))
        for v in ("pass", "fail", "unresolved", "reference cannot certify"):
            r[f"families_{v.replace(' ', '_')}"] = int((d.reading == v).sum())
        tot = d[LABELS].sum()
        r["contrasts"] = int(tot.sum())
        for lab in LABELS:
            r[lab] = int(tot[lab])
            r[f"share_{lab.replace(' ', '_')}"] = round(tot[lab] / tot.sum(), 3) if tot.sum() else float("nan")
        rows.append(r)
out = pd.DataFrame(rows).sort_values(["dataset", "arm"])
out.to_csv(os.path.join(HERE, "results", "CROSS_RELEASE.csv"), index=False)
print(out[["dataset", "arm", "families", "families_pass", "families_fail", "families_unresolved",
           "families_reference_cannot_certify",
           "contrasts", "kept", "not kept", "unresolved", "not read"]].to_string(index=False))
