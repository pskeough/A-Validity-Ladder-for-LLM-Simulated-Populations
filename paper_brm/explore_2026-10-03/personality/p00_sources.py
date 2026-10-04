"""Write SOURCE.txt (URL, retrieval date, licence, SHA-256 of every file) for the two raw folders.
Run after the downloads. Hashes are computed here from the files on disk. Retrieval date 2026-10-03.
"""
import hashlib
import os

RAW = "C:/Research/PsychBench/UpdatedRun/paper_brm/external/raw"
DATE = "2026-10-03"

HEAD = {
    "ipip_audit": """SOURCE: LLM Psychometric Fidelity Audit (IPIP Big Five markers, 50 items; 8 LLMs x 10,000 draws; human sample)
Retrieved: {date}
Hugging Face dataset (files downloaded from): https://huggingface.co/datasets/WilliamWang0624/llm-psychometric-fidelity-audit
  revision sha ff53b747529ff705f3efa174f567239311b3ad34 (last modified 2026-05-05). An identical-size copy sits at
  https://huggingface.co/datasets/Anonymous0624/llm-psychometric-fidelity-audit (revision 6bf1be4699fbd7134fb4714c8c8f090317965766;
  same file sizes, 11,991,277 and 406,814,346 bytes; byte equality not tested).
Licence: dataset card states cc-by-4.0. The human responses are derived from the Open Psychometrics IPIP-FFM
  file (https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip); the card does not restate that site's own terms (not verified).
Code and notebook: https://github.com/anonymous0624/llm-psychometric-fidelity-audit, commit 373d317be65d1d93b38ee3207659cb6cd523eb36
  (2026-05-05). No licence file (GitHub API license: null). The repository is an anonymised review copy; authors not named.
  The paper itself was not located. Files analysis_full.ipynb, llm_collect_main.py, models.csv, GITHUB_README.md come from this repository
  (raw.githubusercontent.com/anonymous0624/llm-psychometric-fidelity-audit/main/...); HF_README.md is the dataset card.
Generation protocol (llm_collect_main.py): system prompt "You are a human participant completing an online personality survey ...";
  temperature 1.0; JSON answer; reasoning off; one persona (no persona variation); OpenRouter.
""",
    "petrov2024": """SOURCE: Petrov, Serapio-Garcia & Rentfrow (2024), "Limited ability of LLMs to simulate human psychological behaviours: a
  psychometric analysis", arXiv 2405.07248. GPT-3.5 and GPT-4 answering BFI-44 (and PANAS, BPAQ, SSCS) as generic and "silicon" personas.
Retrieved: {date}
Repository: https://github.com/nikbpetrov/LLMs-Simulate-Humans, commit 85c3c2f78d5371cf16b71b8b94006d379b4c24fd (2024-05-08).
  Files fetched from raw.githubusercontent.com/nikbpetrov/LLMs-Simulate-Humans/main/ (data/data_for_R, data/bbc_data_for_sharing, shared, README.md).
Licence: none declared in the repository (GitHub API license: null; no LICENSE file seen). Used for research reproduction only; not redistributed.
Human reference: the BBC Big Personality Test raw item answers are NOT in the release (README: "the raw data is not shared"). The release
  holds, per BBC respondent, demographics (bbc_silicon_samples_df.pickle) and five BFI domain means (bbc_summary_scores_df.pickle).
Pickles: bbc_silicon_samples_df.pickle and bbc_summary_scores_df.pickle were opcode-audited before loading (out/p01_pickle_audit.txt in
  explore_2026-10-03/personality): they import pandas, numpy and builtins only. bbc_meta.pickle imports pyreadstat; it is never loaded.
Not downloaded: data/raw_data/*.jsonl (about 1 GB of raw API request and result logs).
""",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for name, head in HEAD.items():
        folder = os.path.join(RAW, name)
        lines = [head.format(date=DATE).rstrip("\n"), "", "FILES (SHA-256, bytes, name)"]
        for fn in sorted(os.listdir(folder)):
            if fn in ("SOURCE.txt",) or fn.startswith("."):
                continue
            p = os.path.join(folder, fn)
            lines.append(f"{sha256(p)}  {os.path.getsize(p):>11d}  {fn}")
        open(os.path.join(folder, "SOURCE.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print("\n".join(lines[-(len(os.listdir(folder)) + 1):]))


if __name__ == "__main__":
    main()
