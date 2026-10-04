"""Test D, supplement. Which answer codes each specification uses (un-oriented 1-5 codes; 1 = strongly oppose,
5 = strongly support), pooled over the 8 items, by party, on the 1,558-respondent fine-tuned sample, plus the
Republican - Democrat gap on the deportation item (QID287_11, the one item where Republicans sit above
Democrats). Output D_answer_code_shares.csv.
"""
import os

import numpy as np
import pandas as pd

from common import *

sims, h4, h13, demo = load_all()
ft_ids = sims["finetune500_gpt41mini"].index
party = demo.party
rows = []
sets = dict(twin=sims, human_w4={"human_w4": h4})
for k, S in {**sims, "human_w4": h4}.items():
    for pty in ("Republican", "Democrat"):
        ids = [i for i in ft_ids if party.get(i) == pty]
        X = S.reindex(ids)[ITEMS]
        vals = X.values.ravel()
        vals = vals[~np.isnan(vals)]
        r = dict(run=k, party=pty, answers=len(vals))
        for c in (1, 2, 3, 4, 5):
            r[f"share_{c}"] = (vals == c).mean()
        r["share_modal_code"] = max((vals == c).mean() for c in (1, 2, 3, 4, 5))
        r["mean_deport_11"] = X["QID287_11"].mean()
        rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "D_answer_code_shares.csv"), index=False)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
