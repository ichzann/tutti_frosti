"""Competition harness (FRAME.md section 10.1): freeze the field set, draw the lockbox, write the per-field package.

    BR_SET=full python -m validation.brazil_psr.handcheck_store   # first: S2 10 bands + QA for selection.parquet
    python -m competition.harness package [SET]                   # SET = full (default) or sel (189-field test set)

Fields get anonymous ids (C0001 ...) so agents cannot look labels up in the PSR caches; the id map, lockbox labels
and harness-only columns (area match, ERA5 coldest night) stay in competition/_harness/ (agents: never read).
Lockbox = 30 % of fields per year x group cell (round half up), numpy seed SEED, drawn once; package refuses to
redraw if competition/_harness/split.parquet exists for a different field set.

Package per field, competition/data/fields/<id>.npz:
  dates (T,) str, refl (T, 10, H, W) uint16 reflectance x 10000 (bands BANDS), clear (T, H, W) bool,
  usable (T,) bool (scene QA: cloud / haze / aot dropped), outline (H, W) bool, year int.
competition/data/fields.parquet: field_id, year, split (dev / lockbox), H, W, T, n_outline.
competition/data/labels_dev.parquet: field_id, year, group, claim (bool), loss (paid only; NaN = censored < 0.35).
group paid_unknown = paid frost claim whose policy has no coverage level, so relative loss is unknown (loss NaN,
claim True); the PSR group rule had put these in heavy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.common import ROOT
from data.s2bands import BANDS

C = Path(os.environ.get("COMP_DIR", ROOT / "competition"))     # COMP_DIR = elsewhere for a test package
DATA = C / "data"
HARN = C / "_harness"
SEED = 20261108
LOCK_SHARE = 0.3
GROUPS = ["none", "mild", "medium", "heavy"]


def _sources(SET):
    """One row per selected field: wid, year, group, loss, area_diff_rel, case cache dir."""
    rows = []
    for d in sorted((ROOT / "cache" / "brazil_psr").glob(f"wheat20[0-9][0-9]{SET}")):
        case = d.name
        p = pd.read_parquet(d / "policies.parquet")
        q = pd.read_parquet(d / f"s2_qa_{case}.parquet")
        ok = {f.stem for f in (d / "raw").glob("*.npz")}
        p = p[p.wid.isin(ok)]
        for r in p.itertuples():
            rows.append(dict(wid=r.wid, year=int(r.year), group=r.group, loss=float(r.loss),
                             area_diff_rel=float(r.area_diff_rel), case=case, dir=str(d)))
        _QA[case] = q
    return pd.DataFrame(rows)


_QA: dict[str, pd.DataFrame] = {}


def split(src):
    rng = np.random.default_rng(SEED)
    lock = set()
    for _, g in src.sort_values("wid").groupby(["year", "group"]):
        n = int(np.floor(len(g) * LOCK_SHARE + 0.5))
        lock |= set(rng.choice(g.wid.to_numpy(), n, replace=False)) if n else set()
    return src.wid.isin(lock).map({True: "lockbox", False: "dev"})


def coldest_night(src):
    """ERA5-Land coldest season night per field (reference baseline only; never given to methods)."""
    out = {}
    for d, g in src.groupby("dir"):
        n = pd.read_parquet(f"{d}/nights.parquet")
        fc = pd.read_parquet(f"{d}/field_cell.parquet").set_index("field_id").cell_id
        tmin = n.groupby("cell_id").tmin_c.min()
        for w in g.wid:
            out[w] = float(tmin.get(fc.get(w), np.nan))
    return pd.Series(out)


def package(SET="full"):
    src = _sources(SET)
    src = src.sort_values(["year", "wid"]).reset_index(drop=True)
    HARN.mkdir(parents=True, exist_ok=True)
    sp_fp = HARN / "split.parquet"
    if sp_fp.exists():
        old = pd.read_parquet(sp_fp)
        if set(old.wid) != set(src.wid):
            sys.exit(f"split exists for a different field set ({len(old)} vs {len(src)}); delete {sp_fp} on purpose")
        src = src.merge(old[["wid", "field_id", "split"]], on="wid")
    else:
        src["field_id"] = [f"C{i + 1:04d}" for i in np.random.default_rng(SEED + 1).permutation(len(src))]
        src["split"] = split(src).to_numpy()
    unk = (src.group != "none") & ~np.isfinite(src.loss)                 # paid, no coverage level in PSR: loss unknown
    src.loc[unk, "group"] = "paid_unknown"
    src["era5_tmin"] = src.wid.map(coldest_night(src))
    (DATA / "fields").mkdir(parents=True, exist_ok=True)
    meta = []
    for r in src.itertuples():
        z = np.load(f"{r.dir}/raw/{r.wid}.npz")
        o = np.load(f"{r.dir}/s2/{r.wid}.npz")["poly"]
        q = _QA[r.case]
        u = q[q.field_id == r.wid].set_index("date").usable
        dates = z["dates"]
        usable = np.array([bool(u.get(str(d), False)) for d in dates])
        fp = DATA / "fields" / f"{r.field_id}.npz"
        if not fp.exists():
            np.savez_compressed(fp, dates=dates, refl=z["refl"], clear=z["clear"], usable=usable, outline=o,
                                year=np.int32(r.year))
        meta.append(dict(field_id=r.field_id, year=r.year, split=r.split, H=o.shape[0], W=o.shape[1],
                         T=len(dates), n_outline=int(o.sum())))
    src.drop(columns=["dir"]).to_parquet(sp_fp)
    pd.DataFrame(meta).sort_values("field_id").to_parquet(DATA / "fields.parquet")
    dev = src[src.split == "dev"]
    claim = dev.group != "none"
    pd.DataFrame(dict(field_id=dev.field_id, year=dev.year, group=dev.group, claim=claim,
                      loss=np.where(claim, dev.loss, np.nan))).sort_values("field_id").to_parquet(
        DATA / "labels_dev.parquet")
    print(f"{len(src)} fields:", pd.crosstab([src.split], src.group).reindex(columns=GROUPS).to_dict("index"))
    print(pd.crosstab(src.year, src.split).to_string())


if __name__ == "__main__":
    {"package": package}[sys.argv[1]](*sys.argv[2:])
