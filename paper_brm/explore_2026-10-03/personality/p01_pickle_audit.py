"""List every global (module.name) that the released .pickle files import, without executing them.
pickletools.genops only decodes opcodes. The files come from github.com/nikbpetrov/LLMs-Simulate-Humans
(a research repository), and p20_petrov.py reads two of them with pandas.read_pickle. This check
confirms they reference pandas / numpy / builtin containers only. Output: out/p01_pickle_audit.txt
"""
import collections
import os
import pickletools

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "C:/Research/PsychBench/UpdatedRun/paper_brm/external/raw/petrov2024"
OUT = os.path.join(HERE, "out")


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    for f in ["bbc_silicon_samples_df.pickle", "bbc_summary_scores_df.pickle", "bbc_meta.pickle"]:
        data = open(os.path.join(RAW, f), "rb").read()
        globs = collections.Counter()
        strings = []
        for op, arg, pos in pickletools.genops(data):
            if op.name in ("GLOBAL", "INST"):
                globs[str(arg).replace("\n", ".").replace(" ", ".")] += 1
            elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
                strings.append(arg)
        # STACK_GLOBAL takes module and name from the two preceding string pushes; list candidate pairs
        stack_globals = []
        prev = []
        for op, arg, pos in pickletools.genops(data):
            if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8", "MEMOIZE", "BINGET", "LONG_BINGET"):
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
                    prev.append(arg)
            if op.name == "STACK_GLOBAL":
                stack_globals.append(".".join(prev[-2:]))
        lines.append(f"== {f}: {len(data)} bytes")
        lines.append("  GLOBAL/INST: " + (", ".join(globs) or "none"))
        lines.append("  STACK_GLOBAL (module.name): " + ", ".join(sorted(set(stack_globals))))
    text = "\n".join(lines)
    print(text)
    open(os.path.join(OUT, "p01_pickle_audit.txt"), "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
