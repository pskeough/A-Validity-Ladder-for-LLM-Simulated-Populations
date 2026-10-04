"""Test B, supplement. Where the extra twin gap sits (which party moves) and how much of a twin's answer is
party versus person.

Sample per specification and item: Republican or Democrat respondents with twin, wave-4 and waves-1-3
answers. Items are oriented (y -> 6 - y where the human waves-1-3 R - D gap is negative) so that the Republican mean is above
the Democrat mean for humans. Item means over the 8 items.

Output B2_shifts_within_party.csv, per spec:
  r2_party_twin / r2_party_w4      share of answer variance explained by party alone
  wp_corr_twin_w4 / wp_corr_twin_w13   within-party correlation (party means removed) of twin with the person's
                                   own wave-4 / waves-1-3 answer
  wp_corr_retest                   within-party correlation of wave 4 with waves 1-3 (human ceiling)
  shift_R, shift_D                 twin mean minus human wave-4 mean, by party (oriented; positive moves up the scale)
  gap_excess                       shift_R - shift_D, the twin gap minus the human gap
"""
import os

import numpy as np
import pandas as pd

from common import *

sims, h4, h13, demo = load_all()
party = demo.party
rd = party.isin(["Republican", "Democrat"])
SIGN = {}
for it in ITEMS:
    x = h13[it][rd]
    SIGN[it] = np.sign(x[party[rd] == "Republican"].mean() - x[party[rd] == "Democrat"].mean())


def wp_corr(a, b, rep):
    a = a - np.where(rep == 1, a[rep == 1].mean(), a[rep == 0].mean())
    b = b - np.where(rep == 1, b[rep == 1].mean(), b[rep == 0].mean())
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def r2(y, rep):
    if y.var() == 0:
        return np.nan
    fit = np.where(rep == 1, y[rep == 1].mean(), y[rep == 0].mean())
    return 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()


rows = []
for k, S in sims.items():
    per = []
    for it in ITEMS:
        d = pd.DataFrame(dict(t=S[it], y4=h4.loc[S.index, it], y13=h13.loc[S.index, it]), index=S.index)
        d["party"] = party.reindex(d.index)
        d = d.dropna().query("party in ['Republican','Democrat']")
        rep = (d.party == "Republican").astype(int).values
        s = SIGN[it]
        orient = (lambda v: v) if s > 0 else (lambda v: 6.0 - v)
        t, y4, y13 = orient(d.t.values), orient(d.y4.values), orient(d.y13.values)
        per.append(dict(r2_party_twin=r2(t, rep), r2_party_w4=r2(y4, rep),
                        wp_corr_twin_w4=wp_corr(t, y4, rep), wp_corr_twin_w13=wp_corr(t, y13, rep),
                        wp_corr_retest=wp_corr(y4, y13, rep),
                        shift_R=t[rep == 1].mean() - y4[rep == 1].mean(),
                        shift_D=t[rep == 0].mean() - y4[rep == 0].mean(),
                        mean_twin_R=t[rep == 1].mean(), mean_twin_D=t[rep == 0].mean(),
                        mean_w4_R=y4[rep == 1].mean(), mean_w4_D=y4[rep == 0].mean()))
    m = pd.DataFrame(per).mean(numeric_only=True)
    m["gap_excess"] = m.shift_R - m.shift_D
    rows.append(dict(run=k, **m.to_dict()))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "B2_shifts_within_party.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(R.round(3).to_string(index=False))
