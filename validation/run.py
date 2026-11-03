"""Tier 1 validation of a damage method: null calibration + synthetic frost injection.

    python -m validation.run --method evi2_gpr                 # dev split, 150 fields
    python -m validation.run --method evi2_gpr --split holdout # lockbox, final comparison only
    python -m validation.run --rescore validation/results/<id> # recompute the scorecard only

Uses only the method plugin interface: Method(cfg, store).damage(field_id, night) -> score
(higher = more damage), valid; the method's binary threshold is `Method.score_threshold`.
Optional layers predicted / pred_std / observed enable the probabilistic calibration block.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import methods
from data.common import ROOT, load_config
from data.store import FieldStore

from .cases import (InjectedStore, endmembers, field_cases, inject, inject_haze, inject_spectral,
                    injection_weight, is_holdout)
from .metrics import scorecard

VCFG = Path(__file__).with_name("config.yaml")
RESULTS = Path(__file__).with_name("results")

_W = {}


def _init(cfg, vcfg, method_name):
    store = FieldStore(cfg)
    spectral = vcfg["injection"].get("mode", "evi2") == "spectral"
    _W.update(cfg=cfg, vcfg=vcfg, store=store, mod=methods.load(method_name),
              em=endmembers(cfg, store, vcfg) if spectral else None)


def _field_job(args):
    """All cases of one field: the null run and the injected run of each case, per mask pixel."""
    field_id, cases = args
    cfg, vcfg, store, mod = _W["cfg"], _W["vcfg"], _W["store"], _W["mod"]
    f = store.field(field_id)
    mask = f["mask"]
    ys, xs = np.nonzero(mask)
    m_null = mod.Method(cfg, store)         # the store is unchanged, so one instance (and its caches) is fine
    weights = {}
    out = []
    for c in cases:
        r0 = m_null.damage(field_id, c["night"])
        if c["pattern"] not in weights:
            weights[c["pattern"]] = injection_weight(cfg, store, field_id, c["pattern"], vcfg)
        w = weights[c["pattern"]]
        if _W["em"] is None:
            fi, drop_abs, drop_rel = inject(f, c["night"], c["level"], w, vcfg)
            m_inj = mod.Method(cfg, InjectedStore(store, field_id, fi))   # fresh: no cache shared with real data
        else:
            refl = store.bands(field_id)
            fi, ri, drop_abs, drop_rel = inject_spectral(f, refl, c["night"], c["level"], w, c, _W["em"], vcfg)
            m_inj = mod.Method(cfg, InjectedStore(store, field_id, fi, ri))
        r1 = m_inj.damage(field_id, c["night"])
        lay = r0.get("layers") or {}
        d = dict(case_id=np.full(len(ys), c["case_id"], "int32"), y=ys.astype("int16"), x=xs.astype("int16"),
                 score_null=r0["score"][ys, xs].astype("float32"), valid_null=r0["valid"][ys, xs],
                 score_inj=r1["score"][ys, xs].astype("float32"), valid_inj=r1["valid"][ys, xs],
                 w=w[ys, xs], drop_abs=drop_abs[ys, xs], drop_rel=drop_rel[ys, xs])
        if _W["em"] is not None and vcfg.get("haze", {}).get("enabled"):
            fh, rh = inject_haze(f, store.bands(field_id), c["night"], c, vcfg)
            r2 = mod.Method(cfg, InjectedStore(store, field_id, fh, rh)).damage(field_id, c["night"])
            d.update(score_haze=r2["score"][ys, xs].astype("float32"), valid_haze=r2["valid"][ys, xs])
        if all(k in lay for k in ("predicted", "pred_std", "observed")):
            d.update(pred=lay["predicted"][ys, xs].astype("float32"),
                     pstd=lay["pred_std"][ys, xs].astype("float32"),
                     obs=lay["observed"][ys, xs].astype("float32"))
        out.append(pd.DataFrame(d))
    return field_id, pd.concat(out) if out else None


def build_cases(store, vcfg, split, n_fields, seed):
    sh = vcfg["split"]["holdout_share"]
    ids = [i for i in store.field_ids() if is_holdout(i, sh) == (split == "holdout")]
    rng = np.random.default_rng(seed)
    ids = sorted(rng.choice(ids, min(n_fields, len(ids)), replace=False))
    rows = []
    for fid in ids:
        for c in field_cases(store, fid, vcfg):
            nights = c.pop("nights")
            c["night"] = nights[rng.integers(len(nights))]
            rows.append(c)
    cases = pd.DataFrame(rows)
    ic = vcfg["injection"]
    combos = [(p, l) for p in ic["patterns"] for l in ic["levels"]]
    order = np.concatenate([rng.permutation(len(combos)) for _ in range(len(cases) // len(combos) + 1)])
    cases["pattern"] = [combos[k][0] for k in order[:len(cases)]]
    cases["level"] = [combos[k][1] for k in order[:len(cases)]]
    cases.insert(0, "case_id", np.arange(len(cases)))
    # conservative frost-free rule, applied after all random draws so the remaining cases are
    # identical (same case_id) to those of runs made before the rule (null_min_tmin_c: null = no filter)
    t = vcfg["domain"].get("null_min_tmin_c")
    if t is not None:
        cases = cases[cases.tmin_min >= t].reset_index(drop=True)
    # spectral-mode draws from a separate stream, so the cases above stay identical to earlier runs
    r2 = np.random.default_rng(seed + 1)
    ic = vcfg["injection"]
    cases["half_days"] = r2.choice(ic.get("half_days", [np.nan]), len(cases))
    cases["dead_mix"] = r2.uniform(*ic.get("dead_mix", [np.nan, np.nan]), len(cases))
    cases["haze_blue"] = r2.choice(vcfg.get("haze", {}).get("blue_refl", [np.nan]), len(cases))
    return cases


def rescore(out):
    vcfg = yaml.safe_load((out / "validation_config.yaml").read_text())
    meta = json.loads((out / "run_meta.json").read_text())
    cases = pd.read_parquet(out / "cases.parquet")
    pix = pd.read_parquet(out / "pixels.parquet")
    keep = [k for k in ("case_id", "field_id", "stage", "pattern", "level", "half_days", "haze_blue")
            if k in cases]
    pix = pix.merge(cases[keep], on="case_id")
    for k in ("field_id", "stage", "pattern"):
        pix[k] = pix[k].astype("category")
    sc, ct = scorecard(pix, cases, meta["threshold"], vcfg)
    sc = dict(run_id=meta["run_id"], method=meta["method"], split=meta["split"],
              git_commit=meta["git_commit"], **sc)
    (out / "scorecard.json").write_text(json.dumps(sc, indent=1, default=str))
    ct.to_parquet(out / "case_results.parquet")
    _leaderboard(sc)
    return sc


def _leaderboard(sc):
    fp = RESULTS / "leaderboard.csv"
    row = dict(run_id=sc["run_id"], method=sc["method"], split=sc["split"], git_commit=sc["git_commit"],
               n_fields=sc["n_fields"], n_cases=sc["n_cases"], passed=sc["passed"],
               **sc["headline"], **{f"crit_{k}": v for k, v in sc["criteria"].items()})
    lb = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
    lb = lb[lb.run_id != sc["run_id"]] if len(lb) else lb
    pd.concat([lb, pd.DataFrame([row])]).to_csv(fp, index=False)


def main():
    from run import git_commit
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="evi2_gpr")
    ap.add_argument("--split", choices=["dev", "holdout"], default="dev")
    ap.add_argument("--fields", type=int, help="override sample.n_fields")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run-id")
    ap.add_argument("--rescore", type=Path)
    a = ap.parse_args()
    if a.rescore:
        sc = rescore(a.rescore)
        print(json.dumps(dict(headline=sc["headline"], criteria=sc["criteria"]), indent=1))
        return
    cfg = load_config()
    vcfg = yaml.safe_load(VCFG.read_text())
    store = FieldStore(cfg)
    mod = methods.load(a.method)
    thr = float(mod.Method(cfg, store).score_threshold)
    n = a.fields or vcfg["sample"]["n_fields"]
    run_id = a.run_id or f"{dt.datetime.now():%Y%m%d-%H%M}_{a.method}_{a.split}"
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now().isoformat(timespec="seconds")

    cases = build_cases(store, vcfg, a.split, n, vcfg["sample"]["seed"])
    cases.to_parquet(out / "cases.parquet")
    print(f"{len(cases)} cases on {cases.field_id.nunique()} fields", flush=True)
    jobs = [(fid, g.to_dict("records")) for fid, g in cases.groupby("field_id")]
    jobs.sort(key=lambda j: -len(store.field(j[0])["mask"].nonzero()[0]) * len(j[1]))  # big first
    parts = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(cfg, vcfg, a.method)) as ex:
        for k, (fid, df) in enumerate(ex.map(_field_job, jobs), 1):
            if df is not None:
                parts.append(df)
            print(f"  field {k}/{len(jobs)} {fid} {dt.datetime.now():%H:%M:%S}", flush=True)
    pd.concat(parts).to_parquet(out / "pixels.parquet")
    (out / "validation_config.yaml").write_text(yaml.safe_dump(vcfg, sort_keys=False))
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "run_meta.json").write_text(json.dumps(dict(
        run_id=run_id, method=a.method, split=a.split, threshold=thr, git_commit=git_commit(),
        started=started, finished=dt.datetime.now().isoformat(timespec="seconds")), indent=1))
    sc = rescore(out)
    print(json.dumps(dict(headline=sc["headline"], criteria=sc["criteria"]), indent=1))


if __name__ == "__main__":
    main()
