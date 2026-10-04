"""Fetch the Twin-2K-500 files that 93_twin2k_ladder.py needs (Toubia et al., 2025; Hugging Face
dataset LLM-Digital-Twin/Twin-2K-500, CC BY 4.0) into paper_brm/external/raw/twin2k/ and write
paper_brm/external/results/twin2k/SOURCE.txt with the resolve URL, size and SHA-256 of every raw file
the ladder reads (files fetched earlier for script 92 are hashed too).

Files: the human wave-4 CSVs, the wave-4 mapping and its README, the catalog generator, and for every
LLM simulation specification its csv_formatted CSVs (simulated, wave 1-3 human, wave 4 human) and
its accuracy_evaluation spreadsheets (used as receipts). Labelled CSVs, PNGs and PDFs are not
fetched. raw/ is git-ignored; nothing here is committed.

The dataset revision is pinned to the commit sha that the Hub reports at fetch time and recorded.
"""
import hashlib
import os
import shutil

from huggingface_hub import HfApi, hf_hub_download

REPO = "LLM-Digital-Twin/Twin-2K-500"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(BASE, "paper_brm", "external", "raw", "twin2k")
OUT = os.path.join(BASE, "paper_brm", "external", "results", "twin2k")
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def wanted(f):
    if f.startswith("question_catalog_and_human_response_csv/"):
        return os.path.basename(f) in ("wave4_response.csv", "wave4_response_label.csv",
                                       "generate_catalog_and_csvs.py", "question_catalog.json",
                                       "question_catalog_README.md", "wave1_3_response.csv")
    if f.startswith("LLM_simulation_results/"):
        if f.endswith(".DS_Store") or f.endswith(".png") or f.endswith(".pdf"):
            return False
        if "/csv_formatted_label/" in f:
            return False
        return True
    return f == "README.md"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    api = HfApi()
    info = api.dataset_info(REPO, files_metadata=True)
    rev = info.sha
    files = sorted(s.rfilename for s in info.siblings if wanted(s.rfilename))
    lines = [f"Twin-2K-500 (Toubia, Gui, Peng, Merlau, Li and Chen, 2025), Hugging Face dataset {REPO}, CC BY 4.0.",
             f"Dataset revision (commit sha) at fetch: {rev}. Fetched 2026-10-03 with huggingface_hub hf_hub_download.",
             "Local path is relative to paper_brm/external/raw/twin2k/. Columns: local path | bytes | sha256 | source URL", ""]
    for f in files:
        # the four files fetched for script 92 sat at the top level of raw/twin2k; keep that layout for them
        top = os.path.basename(f) in ("README.md", "question_catalog.json", "question_catalog_README.md",
                                      "wave1_3_response.csv") and not f.startswith("LLM_")
        local_rel = os.path.basename(f) if top else f
        dst = os.path.join(RAW, local_rel)
        if not os.path.exists(dst):
            p = hf_hub_download(REPO, f, repo_type="dataset", revision=rev)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(p, dst)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/{f.replace(' ', '%20')}"
        lines.append(f"{local_rel} | {os.path.getsize(dst)} | {sha(dst)} | {url}")
        print(local_rel)
    # the default-spec simulated file fetched for script 92 sits at the top level too
    extra = os.path.join(RAW, "responses_llm_imputed_formatted.csv")
    if os.path.exists(extra):
        lines.append(f"responses_llm_imputed_formatted.csv | {os.path.getsize(extra)} | {sha(extra)} | "
                     f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/LLM_simulation_results/"
                     "GPT4.1-mini-simulation-llm-vs-human/responses_llm_imputed_formatted.csv (fetched for script 92)")
    with open(os.path.join(OUT, "SOURCE.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
