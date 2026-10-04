"""run_ladder: every rung on a simulated long dataset and a population reference, one tidy table."""
import numpy as np
import pandas as pd

from .gate import gate
from .level1 import L1Reference, level1
from .level2 import level2
from .level3 import level3
from .level4 import L4Reference, level4
from .thresholds import L1_N_BOOT, L4_N_BOOT

RUNGS = ("gate", "level1", "level2", "level3", "level4")


def _fmt(x, d=2):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def _persona_constant(sim, persona, col):
    return bool((sim.groupby(persona, observed=True)[col].nunique(dropna=False) <= 1).all())


def run_ladder(sim, ref=None, *, persona="persona", items=None, score=None, by=None, attributes=None,
               ref_items=None, ref_score=None, ref_weight=None, ref_psu=None, ref_stratum=None,
               sd_ref=None, contrasts=None, cell_attributes=None, n_categories=None, r4=None,
               rungs=RUNGS, options=None, seed=0, return_details=False):
    """Run the ladder's rungs per model and framing and return one verdict row per rung.

    Parameters
    ----------
    sim : long DataFrame, one row per draw: a persona column, item columns and/or a score column,
        persona attribute columns, and the columns named in `by`.
    ref : reference DataFrame, one row per respondent, with the same item / score and attribute
        columns (same level labels) and optional weight / PSU / stratum columns. None skips every
        rung that needs it.
    persona : persona column. items : item columns (scored 0..M-1 after reverse keying).
    score : score column; default the sum of `items`.
    by : columns that split the simulated data into separate readings (e.g. ["model", "framing"]);
        every verdict is read per model and per framing.
    attributes : persona attributes the reference records (levels 2 and 3). Default: columns of
        sim that are also in ref, constant within persona, and not items, score or design columns.
    ref_items, ref_score : reference column names when they differ from sim's.
    ref_weight, ref_psu, ref_stratum : reference design columns.
    sd_ref : reference SD of the score; default the weighted SD of the reference score.
    contrasts : level-2 contrasts, (attribute, hi, lo[, name]) tuples; default every level pair.
    cell_attributes : level-3 post-stratification cells; default `attributes`.
    r4 : level-4 R4 verdict computed elsewhere (see level4); None leaves level 4 short of a pass.
    rungs : which rungs to run.
    options : dict rung -> keyword arguments for that rung's function, e.g.
        {"gate": {"k": 30}, "level1": {"n_boot": 2000}, "level3": {"tolerance": 2.0}}.
        Every threshold of every rung can be set this way.

    Returns
    -------
    DataFrame with the `by` columns, rung, verdict (pass / fail / unresolved / not applicable /
    not run; level 2 also 'reference cannot certify' when no contrast is certifiable, step 0),
    summary and reason. With return_details=True, also a dict
    {(by values..., rung): full result}.
    """
    options = {k: dict(v) for k, v in (options or {}).items()}
    by = [] if by is None else ([by] if isinstance(by, str) else list(by))
    sim = sim.copy()
    items = None if items is None else list(items)
    ref_items = items if ref_items is None else list(ref_items)
    if score is None:
        if items is None:
            raise ValueError("give `score` or `items`")
        score = "_score"
        sim[score] = sim[items].sum(axis=1)
    if ref is not None:
        ref = ref.copy()
        ref_score = ref_score or score
        if ref_score not in ref.columns:
            if ref_items is None or not set(ref_items) <= set(ref.columns):
                raise ValueError(f"reference has no score column {ref_score!r} and no item columns")
            ref[ref_score] = ref[ref_items].sum(axis=1)
        w = np.ones(len(ref)) if ref_weight is None else ref[ref_weight].to_numpy(float)
        y = ref[ref_score].to_numpy(float)
        ok = np.isfinite(y)
        if sd_ref is None:
            mu = np.average(y[ok], weights=w[ok])
            sd_ref = float(np.sqrt(np.average((y[ok] - mu) ** 2, weights=w[ok]) * ok.sum() / (ok.sum() - 1)))
    design = {persona, score, ref_score, ref_weight, ref_psu, ref_stratum, "_score"} | set(by) | set(items or []) \
        | set(ref_items or [])
    if attributes is None:
        attributes = []
        if ref is not None:
            attributes = [c for c in sim.columns if c in ref.columns and c not in design
                          and _persona_constant(sim, persona, c)]
    attributes = list(attributes)
    cell_attributes = attributes if cell_attributes is None else list(cell_attributes)
    have_items = items is not None and ref is not None and ref_items is not None \
        and set(ref_items) <= set(ref.columns)

    l1ref = l4ref = None
    l1_ref_err = l4_ref_err = None
    if have_items and "level1" in rungs:
        o = options.get("level1", {})
        rr = ref.dropna(subset=ref_items)
        try:
            l1ref = L1Reference(rr[ref_items].to_numpy(int),
                                weights=None if ref_weight is None else rr[ref_weight].to_numpy(float),
                                psu=None if ref_psu is None else rr[ref_psu].to_numpy(),
                                stratum=None if ref_stratum is None else rr[ref_stratum].to_numpy(),
                                n_categories=n_categories, n_boot=o.get("n_boot", L1_N_BOOT), seed=seed)
            if "ref_groups" not in o and "tau" not in o:
                gcols = [c for c in attributes if c in rr.columns]
                if gcols:
                    o["ref_groups"] = rr[gcols].reset_index(drop=True)
            options["level1"] = o
        except ValueError as e:
            l1_ref_err = str(e)
    if have_items and "level4" in rungs:
        o = options.get("level4", {})
        rr = ref.dropna(subset=ref_items)
        cluster = None
        if ref_psu is not None:
            cluster = rr[ref_psu].astype(str)
            if ref_stratum is not None:      # PSU labels need only be unique within a stratum
                cluster = rr[ref_stratum].astype(str) + "|" + cluster
            cluster = cluster.to_numpy()
        try:
            l4ref = L4Reference(rr[ref_items].to_numpy(int),
                                weights=None if ref_weight is None else rr[ref_weight].to_numpy(float),
                                cluster=cluster,
                                stratum=None if ref_stratum is None else rr[ref_stratum].to_numpy(),
                                n_categories=n_categories, n_boot=o.get("n_boot", L4_N_BOOT), seed=seed)
        except ValueError as e:
            l4_ref_err = str(e)

    groups = [((), sim)] if not by else [((k,) if not isinstance(k, tuple) else k, g)
                                          for k, g in sim.groupby(by, observed=True, sort=True)]
    rows, details = [], {}
    for key, g in groups:
        keyd = dict(zip(by, key))
        draws_per_persona = g.groupby(persona, observed=True).size()
        repeated = bool(draws_per_persona.max() >= 2)

        def add(rung, verdict, summary="", reason="", detail=None):
            rows.append(dict(keyd, rung=rung, verdict=verdict, summary=summary, reason=reason))
            if detail is not None:
                details[tuple(key) + (rung,)] = detail

        for rung in RUNGS:
            if rung not in rungs:
                continue
            o = dict(options.get(rung, {}))
            try:
                if rung == "gate":
                    if not repeated:
                        add(rung, "not applicable", reason="no repeated draws of a persona")
                        continue
                    if sd_ref is None:
                        add(rung, "not run", reason="no reference SD (give ref or sd_ref)")
                        continue
                    r = gate(g[score].to_numpy(float), g[persona].to_numpy(), sd_ref, seed=seed, **o)
                    add(rung, r["verdict"],
                        f"SE({_fmt(r['k'], 0)}) = {_fmt(r['se_k'], 3)} vs {_fmt(r['tol_min'], 3)} (minimum) / "
                        f"{_fmt(r['tol_rec'], 3)} (recommended); rule met: {r['rule_met']}; "
                        f"phi({_fmt(r['k'], 0)}) = {_fmt(r['phi_k'])}; k for minimum rule {_fmt(r['k_min'], 0)}",
                        detail=r)
                elif rung == "level1":
                    if not have_items:
                        add(rung, "not run", reason="needs item-level answers in sim and reference")
                        continue
                    if l1ref is None:
                        add(rung, "not run", reason=f"reference model not built: {l1_ref_err}")
                        continue
                    if not repeated:
                        add(rung, "not applicable", reason="no repeated draws of a persona")
                        continue
                    gg = g.dropna(subset=items)
                    r = level1(gg[items].to_numpy(int), gg[persona].to_numpy(), l1ref, seed=seed, **o)
                    add(rung, r["verdict"],
                        f"misfit {_fmt(r['misfit_ratio'])} [{_fmt(r['misfit_ratio_ci90_lo'])}, "
                        f"{_fmt(r['misfit_ratio_ci90_hi'])}], overfit {_fmt(r['overfit_ratio'])} "
                        f"[{_fmt(r['overfit_ratio_ci90_lo'])}, {_fmt(r['overfit_ratio_ci90_hi'])}]; "
                        f"tau {_fmt(r['tau'])} ({r['tau_source']}); {r['verdict_detail']}", detail=r)
                elif rung == "level2":
                    if ref is None:
                        add(rung, "not run", reason="no reference")
                        continue
                    if not attributes:
                        add(rung, "not applicable", reason="personas carry no attribute the reference records")
                        continue
                    r = level2(g, ref, contrasts, attributes=attributes, score=score, persona=persona,
                               ref_score=ref_score, weight=ref_weight, psu=ref_psu, stratum=ref_stratum, **o)
                    lab = r["contrasts"]["label"].value_counts()
                    add(rung, r["reading"],
                        f"{len(r['contrasts'])} contrasts at {_fmt(100 * r['interval_level'], 1)}%, "
                        f"{r['n_certifiable']} certifiable: "
                        + ", ".join(f"{k} {int(lab.get(k, 0))}" for k in ("kept", "not kept", "unresolved", "not read")),
                        detail=r)
                elif rung == "level3":
                    if ref is None:
                        add(rung, "not run", reason="no reference")
                        continue
                    if not cell_attributes:
                        add(rung, "not applicable", reason="personas carry no attribute the reference records")
                        continue
                    o.setdefault("sd_ref", sd_ref)
                    r = level3(g, ref, cell_attributes, score=score, persona=persona, ref_score=ref_score,
                               weight=ref_weight, psu=ref_psu, stratum=ref_stratum, **o)
                    t = r["groups"]
                    vc = t.verdict.value_counts()
                    add(rung, r["verdict"],
                        f"tolerance {_fmt(r['tolerance'], 3)}; {len(t)} groups: "
                        + ", ".join(f"{k} {int(vc.get(k, 0))}" for k in ("pass", "fail", "unresolved"))
                        + f"; largest |residual| {_fmt(np.nanmax(np.abs(t.resid)), 3)}", detail=r)
                elif rung == "level4":
                    if not have_items:
                        add(rung, "not run", reason="needs item-level answers in sim and reference")
                        continue
                    if l4ref is None:
                        add(rung, "not run", reason=f"reference model not built: {l4_ref_err}")
                        continue
                    o.setdefault("r4", r4)
                    gg = g.dropna(subset=items)
                    r = level4(gg[items].to_numpy(int), gg[persona].to_numpy(), l4ref, seed=seed, **o)
                    add(rung, r["verdict"],
                        f"R1 {'pass' if r['R1'] else 'fail'} (ev ratio {_fmt(r['ev_ratio'])}, min loading "
                        f"{_fmt(r['load_min'])}); R2 {r['R2']} (congruence lower limit {_fmt(r['phi_lo90'], 3)}, "
                        f"RMSD upper limit {_fmt(r['loading_rmsd_hi90'], 3)}); R4 {r['r4'] or 'not run'}",
                        reason="" if r["r4"] else "R4 (multi-group invariance) is not computed by this package",
                        detail=r)
            except ValueError as e:
                add(rung, "not run", reason=str(e))
    out = pd.DataFrame(rows, columns=by + ["rung", "verdict", "summary", "reason"])
    return (out, details) if return_details else out
