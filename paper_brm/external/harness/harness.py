"""Standard external-dataset harness for the Validity Ladder (3 Oct 2026).

Every dataset goes through one adapter and the released package (validity_ladder, frozen defaults)
and writes the same outputs:

  results/<name>/l2_contrasts.csv   one row per level-2 contrast reading (package output columns)
  results/<name>/l2_models.csv      one level-2 verdict per model family, with label counts
  results/<name>/rungs.csv          run_ladder rows, one per family and rung (long adapters only)
  results/<name>/manifest.json      adapter, source, input files with sha256, package version, time
  results/<name>/REPORT.md          counts and verdicts written from the CSVs, nothing else

An adapter is a module in adapters/ with NAME, KIND ("language model", "fine-tuned simulator",
"human retest" ...), SOURCE (citation and URLs) and load(), which returns one of

  {"mode": "gaps", "gaps": DataFrame, "inputs": [paths]}
      gaps columns: family, contrast, g, se_g, gamma, se_gamma; optional df_g, df_gamma (default
      inf), cov (default 0), family_size (default the number of contrasts in the family), and any
      extra descriptive columns (model, framing, outcome), which are carried through.
  {"mode": "long", "sim": DataFrame, "ref": DataFrame, "kwargs": dict, "inputs": [paths]}
      sim and ref as validity_ladder.run_ladder takes them; kwargs are passed to run_ladder, whose
      `by` columns define the families.

Usage:  python harness.py <adapter> [<adapter> ...]     (adapter = module name in adapters/)
"""
import hashlib
import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import validity_ladder as vl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

LABELS = ["kept", "not kept", "unresolved", "not read"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_gaps(gaps):
    """Read every contrast of a gaps table with level2_contrast; family size per family."""
    need = ["family", "contrast", "g", "se_g", "gamma", "se_gamma"]
    miss = [c for c in need if c not in gaps.columns]
    if miss:
        raise ValueError(f"gaps table lacks {miss}")
    gaps = gaps.copy()
    for c, d in [("df_g", np.inf), ("df_gamma", np.inf), ("cov", 0.0)]:
        if c not in gaps.columns:
            gaps[c] = d
    if "family_size" not in gaps.columns:
        gaps["family_size"] = gaps.groupby("family")["contrast"].transform("size")
    keep = [c for c in gaps.columns if c not in need + ["df_g", "df_gamma", "cov", "family_size"]]
    rows = []
    for r in gaps.to_dict("records"):
        o = vl.level2_contrast(r["g"], r["se_g"], r["gamma"], r["se_gamma"], df_g=r["df_g"],
                               df_gamma=r["df_gamma"], cov=r["cov"], family_size=int(r["family_size"]))
        rows.append(dict({k: r[k] for k in ["family", "contrast"] + keep}, **o))
    return pd.DataFrame(rows)


def read_long(sim, ref, kwargs):
    """run_ladder per family; level-2 contrast rows taken from the details."""
    kw = dict(kwargs)
    by = kw.get("by") or []
    table, det = vl.run_ladder(sim, ref, return_details=True, **kw)
    table = table.copy()
    table.insert(0, "family", table[by].astype(str).agg(" | ".join, axis=1) if by else "all")
    parts = []
    for key, res in det.items():
        if key[-1] != "level2" or not isinstance(res, dict) or "contrasts" not in res:
            continue
        c = res["contrasts"].copy()
        c.insert(0, "family", " | ".join(str(x) for x in key[:-1]) if by else "all")
        for b, v in zip(by, key[:-1]):
            c[b] = v
        parts.append(c)
    l2 = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return table, l2


def add_certifiable(l2):
    """Step 0 per contrast (package `certifiable`), computed from the stored reference gap when the
    rows predate it."""
    if "certifiable" in l2 and l2["certifiable"].notna().all():
        return l2
    l2 = l2.copy()
    dfg = l2["df_gamma"].fillna(np.inf) if "df_gamma" in l2 else pd.Series(np.inf, index=l2.index)
    l2["certifiable"] = [vl.certifiable(g, s, df_gamma=d, family_size=int(m))
                         for g, s, d, m in zip(l2.gamma, l2.se_gamma, dfg, l2.family_size)]
    return l2


def model_table(l2):
    l2 = add_certifiable(l2)
    rows = []
    for fam, d in l2.groupby("family", sort=False):
        verdicts = [v for v in d["verdict"] if v != "not read"]
        row = dict(family=fam, contrasts=len(d), verdict=vl.level2_model_verdict(verdicts),
                   reading=vl.level2_model_reading(verdicts, d["certifiable"]),
                   certifiable=int(d["certifiable"].sum()))
        for lab in LABELS:
            row[lab] = int((d["label"] == lab).sum())
        row["stopped_no_population_gap"] = int((d["verdict"] == "no population gap").sum())
        row["stopped_reference_imprecise"] = int((d["verdict"] == "reference too imprecise").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out)


def report(name, ad, l2, models, rungs):
    tot = {lab: int((l2["label"] == lab).sum()) for lab in LABELS} if len(l2) else {}
    lines = [f"# {name}", "", f"Kind: {ad.KIND}", "", f"Source: {ad.SOURCE}", "",
             "Generated by harness.py from the CSVs in this folder; every number below is in them.", ""]
    if len(l2):
        lines += ["## Level 2", "",
                  f"{len(l2)} contrast readings in {len(models)} families: "
                  + ", ".join(f"{tot[lab]} {lab}" for lab in LABELS) + ".", "",
                  "Family verdicts: " + ", ".join(f"{int(n)} {v}" for v, n in
                                                  models["verdict"].value_counts().items()) + ".", "",
                  md_table(models), ""]
    if rungs is not None and len(rungs):
        lines += ["## All rungs (run_ladder)", "",
                  md_table(rungs.drop(columns=[c for c in ["reason"] if c in rungs.columns])), ""]
    return "\n".join(lines)


def run(adapter):
    ad = importlib.import_module(f"adapters.{adapter}")
    t0 = time.time()
    data = ad.load()
    out = os.path.join(OUT_ROOT, ad.NAME)
    os.makedirs(out, exist_ok=True)
    rungs = None
    if data["mode"] == "gaps":
        l2 = read_gaps(data["gaps"])
    elif data["mode"] == "long":
        rungs, l2 = read_long(data["sim"], data["ref"], data["kwargs"])
        rungs.to_csv(os.path.join(out, "rungs.csv"), index=False)
    else:
        raise ValueError(data["mode"])
    l2.to_csv(os.path.join(out, "l2_contrasts.csv"), index=False)
    models = model_table(l2) if len(l2) else pd.DataFrame()
    models.to_csv(os.path.join(out, "l2_models.csv"), index=False)
    manifest = dict(adapter=adapter, name=ad.NAME, kind=ad.KIND, source=ad.SOURCE, mode=data["mode"],
                    package_version=vl.__version__,
                    inputs=[dict(path=p, sha256=sha256(p), bytes=os.path.getsize(p)) for p in data.get("inputs", [])],
                    run_at=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1),
                    notes=data.get("notes", ""))
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report(ad.NAME, ad, l2, models, rungs))
    print(f"{ad.NAME}: {len(l2)} contrasts, families {models['verdict'].value_counts().to_dict() if len(models) else {}}"
          f" ({manifest['seconds']}s)")
    return l2, models


if __name__ == "__main__":
    for a in sys.argv[1:]:
        run(a)
