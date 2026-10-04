"""Supplement S8 tables from the harness outputs: SubPOP per contrast (S8_subpop.tex) and the leniency
totals (S8_leniency.tex), written to paper_brm/manuscript/supplement/tables/. Run after leniency.py."""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
T = os.path.join(HERE, "..", "..", "manuscript", "supplement", "tables")


def tex_escape(s):
    return s.replace("$", r"\$").replace("&", r"\&")


def subpop():
    d = pd.read_csv(os.path.join(R, "subpop2025", "l2_contrasts.csv"))
    d["contrast"] = d.contrast.str.replace("College graduate/some postgrad", "College graduate", regex=False)
    lines = ["% Written by paper_brm/external/harness/supp_tables.py from harness/results/subpop2025/l2_contrasts.csv.",
             r"\begin{table}[!ht]", r"\centering", r"\caption{SubPOP and Its Base Model on the OpinionQA Contrasts}",
             r"\label{tab:s8subpop}", r"\footnotesize", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}lrrlrrl@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Base model} & \multicolumn{3}{c}{SubPOP} \\",
             r"\cmidrule(lr){3-4}\cmidrule(l){5-7}",
             r"Contrast & Human gap & Ratio & Label & Ratio & Interval & Label \\", r"\midrule"]
    for subset, title in (("all", "All questions"), ("conservative", "Questions fielded before SubPOP's training waves")):
        b = d[d.family == f"base : {subset}"].set_index("contrast")
        s = d[d.family == f"subpop : {subset}"].set_index("contrast")
        n = int(s.n_questions.max())
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{title} ($n$ = {n})}}}} \\")
        def num(x):
            return f"{x:.2f}".replace("-", "$-$")

        for c in s.index:
            ci = f"[{num(s.loc[c, 'ci_lo'])}, {num(s.loc[c, 'ci_hi'])}]" if pd.notna(s.loc[c, "ci_lo"]) else "--"
            lines.append(f"{tex_escape(c).replace(' - ', '--')} & {s.loc[c, 'gamma']:.3f} & {num(b.loc[c, 'ratio'])} & "
                         f"{b.loc[c, 'verdict']} & {num(s.loc[c, 'ratio'])} & {ci} & {s.loc[c, 'verdict']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\vspace{4pt}", r"\begin{tabnote}",
              r"\textit{Note.} Human gap: oriented mean gap in the ordinal answer position (0 to 1) over questions, "
              r"from SubPOP's processed OpinionQA file. Ratio: simulated over human gap. Label: the level-2 label of "
              r"Section 2. Intervals are Fieller intervals at the Bonferroni level for a family of 8.",
              r"\end{tabnote}", r"\end{table}"]
    with open(os.path.join(T, "S8_subpop.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def leniency():
    d = pd.read_csv(os.path.join(R, "LENIENCY_totals.csv"))
    order = ["frozen", "wide band 0.5-1.5", "no multiplicity", "wide band, no multiplicity"]
    names = {"frozen": "Frozen rule", "wide band 0.5-1.5": "Kept region 0.5 to 1.5",
             "no multiplicity": "No multiplicity correction", "wide band, no multiplicity": "Both"}
    lines = ["% Written by paper_brm/external/harness/supp_tables.py from harness/results/LENIENCY_totals.csv.",
             r"\begin{table}[!ht]", r"\centering", r"\caption{Family Verdicts Under Easier Level-2 Rules}",
             r"\label{tab:s8leniency}", r"\footnotesize", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Simulators} & \multicolumn{3}{c}{Faithful controls} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Rule & Pass & Fail & Other & Pass & Fail & Other \\", r"\midrule"]
    for rule in order:
        m = d[(d.arm == "models") & (d.rule == rule)].iloc[0]
        c = d[(d.arm == "faithful control") & (d.rule == rule)].iloc[0]
        lines.append(f"{names[rule]} & {m['pass']} & {m['fail']} & {m['unresolved']} & {c['pass']} & {c['fail']} & "
                     f"{c['unresolved']} \\\\")
    m0 = d[(d.arm == "models")].families.iloc[0]
    c0 = d[(d.arm == "faithful control")].families.iloc[0]
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabnote}",
              rf"\textit{{Note.}} {m0} simulator families and {c0} faithful-control families (ideal simulators, "
              r"split halves and the Twin-2K-500 retest) from the nine releases, every stored contrast re-read under "
              r"each rule. Other: unresolved, including families the reference cannot certify.",
              r"\end{tabnote}", r"\end{table}"]
    with open(os.path.join(T, "S8_leniency.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    subpop()
    leniency()
    print("written", os.listdir(T)[-3:])
