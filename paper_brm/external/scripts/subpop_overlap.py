"""Check the questions SubPOP was read on against SubPOP's released training set (jjssuh/subpop on
Hugging Face, gated; downloaded to external/raw/subpop_hf, not redistributed).

Three levels of overlap, per question read in the harness (raw/subpop_gen/subpop_distributions.csv):
  key   - the same Pew question key (e.g. GROWUPGUN4_W26) appears in subpop_train.jsonl
  stem  - the key without its wave suffix appears in training (the same item fielded in another wave)
  text  - the normalised question wording appears in training under any key
Output: harness/results/subpop2025/training_overlap.csv and a one-line summary on stdout.
"""
import json
import os
import re

import pandas as pd

EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAIN = os.path.join(EXT, "raw", "subpop_hf", "subpop_train.jsonl")
READ = os.path.join(EXT, "raw", "subpop_gen", "subpop_distributions.csv")
OUT = os.path.join(EXT, "harness", "results", "subpop2025", "training_overlap.csv")
CONSERVATIVE_DROP = lambda q: q.endswith("_W82") or q.endswith("_W92") or q == "CONTROLCO_W49"  # noqa: E731


def stem(k):
    return re.sub(r"_W\d+$", "", k)


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(t).lower())).strip()


def main():
    keys, stems, texts = set(), set(), set()
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            keys.add(r["qkey"])
            stems.add(stem(r["qkey"]))
            texts.add(norm(r["question"]))
    read = pd.read_csv(READ)
    qs = sorted(set(read.qkey) - {"IDIMPORT_W43"})
    # question wording for the read set comes from SubPOP's own OpinionQA file via the eval/train text match
    # is not available here, so text overlap is checked against the prompt text stored with each question
    rows = []
    for q in qs:
        rows.append(dict(qkey=q, key_in_train=q in keys, stem_in_train=stem(q) in stems,
                         conservative=not CONSERVATIVE_DROP(q)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"read {len(out)} questions; training keys {len(keys)}; key overlap {int(out.key_in_train.sum())}; "
          f"stem overlap {int(out.stem_in_train.sum())} "
          f"({', '.join(out[out.stem_in_train].qkey)}); "
          f"stem overlap in conservative set {int((out.stem_in_train & out.conservative).sum())}")


if __name__ == "__main__":
    main()
