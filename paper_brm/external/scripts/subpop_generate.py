"""Answer distributions of a survey-trained simulator (SubPOP, Suh et al. 2025) and its base model on the
paper's 100 contested OpinionQA questions, for the 14 groups of the paper's 8 OpinionQA contrasts.

Prompt and scoring follow the SubPOP repository exactly (scripts/data_generation/prepare_finetuning_data.py,
QA steering; scripts/experiment/run_inference.py): a steering multiple-choice question about the group
answered with the group's option, "Answer the following question keeping in mind your previous answers.",
then the survey question built by subpop.utils.survey_utils.generate_mcq(add_answer_forcing=True), ending
in "Answer:". The next-token probabilities of "A", " A", "B", " B", ... are summed per option and
renormalised over the non-refusal options. Base and SubPOP use the same loaded weights (the LoRA adapter
is switched off for the base).

Weights: mistralai/Mistral-7B-v0.1 (Apache-2.0) and jjssuh/mistral-7b-v0.1-subpop (LoRA r=8, Apache-2.0).
Questions, options, ordinals and group distributions: SubPOP's data/opinionqa/processed/opinionqa.csv.

Run (overlay env; the pyshim folder holds the WMI sitecustomize, without it imports can hang on this machine):
  $env:PYTHONPATH='C:\\Research\\PsychBench\\UpdatedRun\\paper_ml4h\\plan\\rev2026-09\\pyshim;C:\\LocalAI\\LLM\\hf_validity_ladder\\overlay'
  & 'C:\\AI Coding Projects\\Local AI\\Podcast_Audio_Gen\\.venv\\Scripts\\python.exe' subpop_generate.py [--limit N]
Output: ../raw/subpop_gen/subpop_distributions.csv (+ run log, prompt sample)
"""
import argparse
import ast
import faulthandler
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

faulthandler.enable()
HF =r"C:\LocalAI\LLM\hf_validity_ladder"
BASE = os.path.join(HF, "Mistral-7B-v0.1")
ADAPTER = os.path.join(HF, "mistral-7b-v0.1-subpop")
REPO = os.path.join(HF, "subpop_repo")
EXT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INFO = os.path.join(EXT, "raw", "meister2024", "opinions_qa", "data", "human_resp",
                    "Pew_American_Trends_Panel_disagreement_500", "info.csv")
T0 = os.path.join(EXT, "raw", "meister2024", "results", "opinionqa", "express_distribution", "gpt-4", "task0",
                  "Pew_American_Trends_Panel_disagreement_100", "NONE", "Democrat.json")
OUT = os.path.join(EXT, "raw", "subpop_gen")

sys.path.insert(0, REPO)
from subpop.utils.survey_utils import generate_mcq  # noqa: E402

GROUPS = {  # SubPOP attribute -> groups (demographics_22.csv names), covering the paper's 8 contrasts
    "POLPARTY": ["Republican", "Democrat"], "SEX": ["Female", "Male"],
    "RACE": ["Black", "White", "Hispanic", "Asian"],
    "EDUCATION": ["Less than high school", "College graduate/some postgrad"],
    "INCOME": ["Less than $30,000", "$100,000 or more"], "CREGION": ["South", "Northeast"],
}


def steering(attribute, group, prompts):
    row = prompts[prompts.attribute == attribute].iloc[0]
    options = ast.literal_eval(row["options"]) if isinstance(row["options"], str) else list(row["options"])
    assert group in options, (attribute, group)
    s = generate_mcq(question_body=row["qa_prompt"], options=options)
    s += f" {chr(ord('A') + options.index(group))}. {group}\n\n"
    s += "Answer the following question keeping in mind your previous answers.\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke test)")
    ap.add_argument("--gpu", default="1", help="CUDA index to place the whole model on")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    qa = pd.read_csv(os.path.join(REPO, "data", "opinionqa", "processed", "opinionqa.csv"))
    prompts = pd.read_json(os.path.join(REPO, "data", "subpopulation_metadata", "steering_prompts.json"))
    t0 = json.load(open(T0, encoding="utf-8"))
    keys = sorted(set(t0))  # the paper's 100 contested questions (99 unique keys)
    rows = []
    for attr, groups in GROUPS.items():
        for grp in groups:
            sub = qa[(qa.attribute == attr) & (qa.group == grp) & qa.qkey.isin(keys)]
            for r in sub.itertuples():
                options = ast.literal_eval(r.options)
                prompt = steering(attr, grp, prompts) + generate_mcq(question_body=r.question, options=options,
                                                                    add_answer_forcing=True)
                rows.append(dict(qkey=r.qkey, attribute=attr, group=grp, n_options=len(ast.literal_eval(r.ordinal)),
                                 options=r.options, ordinal=r.ordinal, human=r.responses, prompt=prompt))
    jobs = pd.DataFrame(rows)
    print("questions", jobs.qkey.nunique(), "prompts", len(jobs), flush=True)
    if args.limit:
        jobs = jobs.head(args.limit)
    with open(os.path.join(OUT, "prompt_example.txt"), "w", encoding="utf-8") as f:
        f.write(jobs.prompt.iloc[0])

    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    # Reading the shards through a memory map (transformers' loader or safetensors.get_tensor) hits a Windows
    # access violation on this machine once the model occupies the card. So the model is built on the card and
    # each weight is read with ordinary file reads from the safetensors layout (8-byte header length, JSON
    # header, raw bytes) and copied in; any missing or unexpected key stops the run.
    dev = torch.device(f"cuda:{args.gpu}")
    with dev:
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(BASE), dtype=torch.bfloat16)
    params = dict(model.named_parameters())
    seen = set()
    for shard in sorted(f for f in os.listdir(BASE) if f.endswith(".safetensors")):
        with open(os.path.join(BASE, shard), "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n))
            for k, meta in header.items():
                if k == "__metadata__":
                    continue
                assert k in params and meta["dtype"] == "BF16", f"unexpected weight {k} {meta['dtype']}"
                a, b = meta["data_offsets"]
                fh.seek(8 + n + a)
                t = torch.frombuffer(bytearray(fh.read(b - a)), dtype=torch.bfloat16).reshape(meta["shape"])
                with torch.no_grad():
                    params[k].copy_(t)
                seen.add(k)
    assert seen == set(params), f"missing weights {sorted(set(params) - seen)[:5]}"
    check = tok("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named "
                "after the engineer Gustave Eiffel, whose company designed and built the tower.",
                return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        loss = float(model(check, labels=check).loss)
    print("weights", len(seen), "| sanity loss on a Wikipedia sentence:", round(loss, 3), flush=True)
    assert loss < 2.5, loss  # a trained 7B model scores well under 2.5 nats per token here; random weights near 10
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    letter_ids = [(tok.encode("Answer:" + chr(ord("A") + i), add_special_tokens=False)[-1],
                   tok.encode("Answer: " + chr(ord("A") + i), add_special_tokens=False)[-1]) for i in range(26)]
    dev = next(model.parameters()).device

    out = []
    t_start = time.time()
    for arm in ("base", "subpop"):
        for j, r in enumerate(jobs.itertuples()):
            ids = tok(r.prompt, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                if arm == "base":
                    with model.disable_adapter():
                        logits = model(ids).logits[0, -1].float()
                else:
                    logits = model(ids).logits[0, -1].float()
            p = torch.softmax(logits, -1).cpu().numpy()
            per = np.array([p[a] + p[b] for a, b in letter_ids[:r.n_options]])
            out.append(dict(arm=arm, qkey=r.qkey, attribute=r.attribute, group=r.group, prob_sum=float(per.sum()),
                            dist=json.dumps((per / per.sum()).tolist()), human=r.human, ordinal=r.ordinal,
                            options=r.options))
            if j % 100 == 0:
                print(f"{arm} {j}/{len(jobs)} {time.time() - t_start:.0f}s", flush=True)
        pd.DataFrame(out).to_csv(os.path.join(OUT, "subpop_distributions.csv"), index=False)
    print("done", len(out), f"{time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
