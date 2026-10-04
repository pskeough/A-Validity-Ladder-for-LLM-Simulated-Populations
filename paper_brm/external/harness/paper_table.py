"""The article's cross-release level-2 table (Table 5) from results/CROSS_RELEASE.csv.

One row per release: the simulators' family readings (pass / fail / unresolved / reference cannot certify)
and, where the release allows one, a faithful control read the same way. Writes results/PAPER_TABLE.csv and
paper_brm/manuscript/tables/tab_external_all.tex. Run after summarise.py.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "..", "..", "manuscript", "tables", "tab_external_all.tex")
RECEIPT = "../../paper_brm/external/harness/results/PAPER_TABLE.csv"

# (dataset, arms read as simulators, release label, simulator label, control arm, control label)
ROWS = [
    ("bisbee", ["models"], r"\citet{bisbee2024synthetic}", "ChatGPT", None, ""),
    ("argyle", ["models"], r"\citet{argyle2023outofone}", "GPT-3", "ideal control", "Ideal"),
    ("opinionqa", ["models"], r"\citet{meister2025distributional}", "5 LLMs", "ideal control", "Ideal"),
    ("huang2025", ["models"], r"\citet{huang2025uncertainty}", "8 LLMs", "ideal control", "Ideal"),
    ("huang2025", [], "", "", "ideal control (matched size)", "Ideal, small"),
    ("subpop2025", ["models"], r"\citet{suh2025subpop}", "Base", None, ""),
    ("subpop2025", ["survey-trained simulator"], "", "SubPOP", None, ""),
    ("twin2k", ["models"], r"\citet{toubia2025twin2k}", "13 twins", "human retest control", "Retest"),
    ("twin2k_mega_bystudy", ["models"], r"\citet{peng2025funhouse}", "23 twins", "split-half control",
     "Split half"),
    ("malmberg2026", ["models"], r"\citet{malmberg2026conjoint}", "3 LLMs", "split-half control", "Split half"),
    ("cummins2025", ["models"], r"\citet{cummins2026flexibility}", "9 LLMs", "split-half control", "Split half"),
]
COLS = ["families", "families_pass", "families_fail", "families_unresolved", "families_reference_cannot_certify"]


def main():
    cr = pd.read_csv(os.path.join(HERE, "results", "CROSS_RELEASE.csv"))
    out = []
    for ds, arms, rel, sim, carm, clab in ROWS:
        d = cr[cr.dataset == ds]
        s = d[d.arm.isin(arms)][COLS].sum() if arms else pd.Series(0, index=COLS)
        c = d[d.arm == carm][COLS].sum() if carm else pd.Series(0, index=COLS)
        assert not arms or len(d[d.arm.isin(arms)]), (ds, arms)
        assert not carm or len(d[d.arm == carm]), (ds, carm)
        out.append(dict(dataset=ds, release=rel, simulator=sim, n=int(s.families), pass_=int(s.families_pass),
                        fail=int(s.families_fail), unresolved=int(s.families_unresolved),
                        cannot=int(s.families_reference_cannot_certify), control=clab,
                        c_n=int(c.families), c_pass=int(c.families_pass), c_fail=int(c.families_fail),
                        c_unresolved=int(c.families_unresolved), c_cannot=int(c.families_reference_cannot_certify)))
    t = pd.DataFrame(out)
    tot = t[t.n > 0][["n", "pass_", "fail", "unresolved", "cannot"]].sum()
    t = pd.concat([t, pd.DataFrame([dict(dataset="all simulators", release="All simulators", **tot.to_dict())])])
    t.to_csv(os.path.join(HERE, "results", "PAPER_TABLE.csv"), index=False)
    t = t[t.dataset != "all simulators"].copy()
    for c in ("c_n", "c_pass", "c_fail", "c_unresolved", "c_cannot"):
        t[c] = t[c].astype(int)
    design()

    def cells(n, p, f, u, k):
        return " & ".join(str(x) for x in (n, p, f, u, k)) if n else " & ".join(["--"] * 5)

    lines = [f"% Written by paper_brm/external/harness/paper_table.py from results/CROSS_RELEASE.csv. Do not edit by hand.",
             r"\begin{table}[!ht]", r"\centering",
             r"\caption{Level-2 Readings of Released Simulators and of Faithful Controls}",
             r"\label{tab:external-all}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{@{}llrrrrrlrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{5}{c}{Simulators} & & \multicolumn{5}{c}{Faithful control} \\",
             r"\cmidrule(lr){3-7}\cmidrule(l){9-13}",
             r"Release & Simulator & $n$ & Pass & Fail & Unres. & Cannot & Control & $n$ & Pass & Fail & Unres. & Cannot \\",
             r"\midrule"]
    for r in t.itertuples():
        lines.append(f"{r.release} & {r.simulator} & {cells(r.n, r.pass_, r.fail, r.unresolved, r.cannot)} & "
                     f"{r.control} & {cells(r.c_n, r.c_pass, r.c_fail, r.c_unresolved, r.c_cannot)} \\\\ % R: {RECEIPT}")
    lines += [r"\midrule",
              f"All simulators & & {int(tot.n)} & {int(tot.pass_)} & {int(tot.fail)} & {int(tot.unresolved)} & "
              f"{int(tot.cannot)} & & & & & & \\\\ % R: {RECEIPT}",
              r"\bottomrule", r"\end{tabular}}", r"\vspace{4pt}", r"\begin{tabnote}",
              r"\textit{Note.} Each reading is one family of contrasts read together, such as one model under one "
              r"prompt (one model and study for \citet{peng2025funhouse}). Pass: every contrast read is kept. Fail: at least one is "
              r"not kept. Unres.: neither, with at least one contrast the reference could certify. Cannot: none is not kept, and the reference "
              r"could not certify a kept verdict on any contrast in the family (step 0). Simulators: ChatGPT under 3 "
              r"prompts; GPT-3 in 3 runs; 5 LLMs under 3 elicitation and 2 steering methods; Mistral-7B before (Base) and "
              r"after (SubPOP) survey training; twin specifications, over 18 studies for \citet{peng2025funhouse}; 252 "
              r"configurations of 9 LLMs. Ideal: a fresh sample of the released human answers at the human group sizes; "
              r"Ideal, small: the same at the size of the LLM samples. Retest: the respondents' own answers in earlier "
              r"waves. Split half: one half of the respondents read against the other. \citet{suh2025subpop} is read "
              r"against SubPOP's processed copy of the human answers to the questions of "
              r"\citet{meister2025distributional}, whose ideal simulator is the nearest control (Supplement S8).",
              r"\end{tabnote}", r"\end{table}"]
    with open(TEX, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(t.to_string())
    print("totals", tot.to_dict())


def design():
    """Design counts the article states for the added releases, read from the harness outputs
    (results/DESIGN.csv), so each has a receipt."""
    R = os.path.join(HERE, "results")

    def rd(name):
        return pd.read_csv(os.path.join(R, name, "l2_contrasts.csv"), low_memory=False)

    hu, mg, cu, ml, sp = rd("huang2025"), rd("twin2k_mega_bystudy"), rd("cummins2025"), rd("malmberg2026"), \
        rd("subpop2025")
    mg_m = mg[~mg.family.str.startswith("human_split_half")]
    rows = [
        ("huang2025", "llms", hu.model.nunique()), ("huang2025", "max_questions_per_contrast", hu.n_questions.max()),
        ("huang2025", "profiles_per_question", hu.n_profiles.max()),
        ("twin2k_mega", "specifications", mg_m.specification.nunique()), ("twin2k_mega", "studies", mg_m.study.nunique()),
        ("cummins2025", "configurations", cu[cu.family.str.contains("split") == False].family.nunique()),  # noqa: E712
        ("cummins2025", "max_people_per_contrast", (cu.n_hi + cu.n_lo).max()),
        ("malmberg2026", "llms", ml.model.dropna().nunique()),
        ("subpop2025", "questions_all", sp[sp.family == "subpop : all"].n_questions.max()),
        ("subpop2025", "questions_conservative", sp[sp.family == "subpop : conservative"].n_questions.max()),
        ("subpop2025", "ratio_min_subpop_all", sp[sp.family == "subpop : all"].ratio.min()),
        ("subpop2025", "ratio_max_subpop_all_nonparty",
         sp[(sp.family == "subpop : all") & (sp.contrast != "Republican - Democrat")].ratio.max()),
        ("subpop2025", "ratio_party_subpop_all",
         sp[(sp.family == "subpop : all") & (sp.contrast == "Republican - Democrat")].ratio.iloc[0]),
        ("subpop2025", "base_all_negative_or_below_0.3",
         int((sp[sp.family == "base : all"].ratio < 0.3).sum())),
    ]
    pd.DataFrame(rows, columns=["release", "quantity", "value"]).to_csv(os.path.join(R, "DESIGN.csv"), index=False)
    print(pd.DataFrame(rows, columns=["release", "quantity", "value"]).to_string())


if __name__ == "__main__":
    main()
