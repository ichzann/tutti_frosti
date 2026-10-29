"""End-to-end Phase 1 run:  python run.py [--run-id ID] [--method evi2_gpr] [--workers N]

prepare inputs (cached) -> evaluate the method for every field on every night of its
growth phase (plus every trigger / near-trigger night) -> write runs/<id>/ -> report.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from matplotlib import colormaps
from PIL import Image

import methods
from data import dem
from data.common import ROOT, cache_dir, load_config
from data.prepare import prepare
from data.store import FieldStore

EVI2_RANGE = (0.0, 0.9)
Z_RANGE = (-4.0, 4.0)
REASON_COLORS = {0: (40, 160, 70), 1: (0, 0, 0), 2: (150, 150, 150), 3: (230, 160, 40),
                 4: (200, 110, 30), 5: (120, 80, 200), 6: (220, 40, 40)}


# ------------------------------------------------------------------ images --
def _png(rgba, fp):
    Image.fromarray(rgba, "RGBA").save(fp, optimize=True)


def _cmap_png(a, mask, cmap, vmin, vmax, fp):
    v = np.clip((a - vmin) / (vmax - vmin), 0, 1)
    rgba = (colormaps[cmap](np.nan_to_num(v)) * 255).astype("uint8")
    rgba[..., 3] = np.where(mask & np.isfinite(a), 255, 0)
    _png(rgba, fp)


def _rgb_png(rgb, mask, fp):
    rgba = np.concatenate([np.moveaxis(rgb, 0, -1), np.full(rgb.shape[1:] + (1,), 255, "uint8")], -1)
    p = np.pad(mask, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    rgba[mask & ~inner] = (255, 255, 0, 255)  # outline of the valid field pixels
    _png(rgba, fp)


def _reason_png(reason, fp):
    rgba = np.zeros(reason.shape + (4,), "uint8")
    for k, c in REASON_COLORS.items():
        sel = reason == k
        rgba[sel, :3] = c
        rgba[sel, 3] = 0 if k == 1 else 255
    _png(rgba, fp)


# ------------------------------------------------------------ field worker --
_W = {}


def _init(cfg, method_name):
    store = FieldStore(cfg)
    _W.update(cfg=cfg, store=store, method=methods.load(method_name).Method(cfg, store))


def _field_job(args):
    field_id, nights, flagged, out_dir = args
    cfg, store, m = _W["cfg"], _W["store"], _W["method"]
    f = store.field(field_id)
    mask = f["mask"]
    dates = f["dates"]
    emerg, peak, (sd, smed, ssm) = m.growth(field_id)
    img = out_dir / "img" / field_id
    ras = out_dir / "rasters" / field_id
    img.mkdir(parents=True, exist_ok=True)
    ras.mkdir(parents=True, exist_ok=True)

    row = store.fields[store.fields.field_id == field_id].iloc[0]
    elev = dem.field_elevation(cfg, row)
    rel = elev - np.nanmedian(elev[mask])
    rel_lim = float(np.nanpercentile(np.abs(rel[mask]), 98)) if mask.any() else 1.0
    rel_lim = max(rel_lim, 0.5)
    _cmap_png(rel, mask, "RdBu_r", -rel_lim, rel_lim, img / "relelev.png")

    in_phase = [n for n in nights if emerg and peak and emerg.isoformat() <= n <= peak.isoformat()]
    todo = sorted(set(in_phase) | set(flagged))
    rows, curves, dates_needed = [], {}, set()
    for n in todo:
        r = m.damage(field_id, n)
        meta, valid, lay = r["meta"], r["valid"], r["layers"]
        z = lay["z"]
        nv = int(valid.sum())
        zv = z[valid]
        rows.append(dict(meta, n_px=int(mask.sum()), n_valid=nv,
                         valid_share=nv / max(int(mask.sum()), 1),
                         mean_z=float(zv.mean()) if nv else np.nan,
                         median_z=float(np.median(zv)) if nv else np.nan,
                         mean_score=float(r["score"][valid].mean()) if nv else np.nan,
                         share_damaged=float((zv <= cfg["damage"]["z_damaged"]).mean()) if nv else np.nan,
                         rendered=False))
        if n not in flagged or meta["growth_stage"] != "growth":
            continue
        rows[-1]["rendered"] = True
        tag = n.replace("-", "")
        idx = {k: lay[f"{k}_idx"] for k in ("pre", "post", "next")}
        np.savez_compressed(
            ras / f"{tag}.npz", score=r["score"], valid=valid, reason=lay["reason"], z=z,
            observed=lay["observed"], predicted=lay["predicted"], pred_std=lay["pred_std"],
            **{f"rgb_{k}": f["rgb"][i] for k, i in idx.items() if i is not None},
            **{f"evi2_{k}": f["evi2"][i] for k, i in idx.items() if i is not None})
        _cmap_png(lay["predicted"], valid, "viridis", *EVI2_RANGE, img / f"{tag}_pred.png")
        _cmap_png(z, valid, "RdBu", *Z_RANGE, img / f"{tag}_z.png")
        _reason_png(lay["reason"], img / f"{tag}_valid.png")
        dates_needed |= {i for i in idx.values() if i is not None}
        c = r["curve"]
        if c is not None:
            curves[n] = dict(day=c["days"].astype(int).tolist(),
                             mean=np.round(c["mean"], 3).tolist(), std=np.round(c["std"], 4).tolist())
    for i in sorted(dates_needed):
        d = dates[i].isoformat().replace("-", "")
        _rgb_png(f["rgb"][i], mask, img / f"rgb_{d}.png")
        _cmap_png(f["evi2"][i].astype("float32"), mask, "viridis", *EVI2_RANGE, img / f"evi2_{d}.png")
    clear_share = [float(np.isfinite(e[mask]).mean()) for e in f["evi2"]]
    series = dict(dates=[d.isoformat() for d in sd], median=np.round(smed, 3).tolist(),
                  smooth=np.round(ssm, 3).tolist(),
                  all_dates=[d.isoformat() for d in dates], clear_share=np.round(clear_share, 3).tolist())
    m.clear_cache()
    return field_id, rows, dict(series=series, curves=curves, rel_lim=round(rel_lim, 1),
                                shape=[int(mask.shape[0]), int(mask.shape[1])],
                                emergence=emerg and emerg.isoformat(), peak=peak and peak.isoformat())


# -------------------------------------------------------------------- main --
def git_commit():
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip())
        return c + ("-dirty" if dirty else "")
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--method", default="evi2_gpr")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fields", type=int, help="only the first N fields (debug)")
    a = ap.parse_args()
    cfg = load_config()
    fips = cfg["study"]["county_fips"]
    run_id = a.run_id or f"{dt.datetime.now():%Y%m%d-%H%M}_{a.method}_{fips}"
    out = ROOT / cfg["paths"]["runs"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now().isoformat(timespec="seconds")

    prepare(cfg)
    store = FieldStore(cfg)
    c = cache_dir(cfg)

    # nights table + RTMA cells for the county
    cells = gpd.read_file(c / "rtma_cells.gpkg")
    cells = cells[cells[f"in_{fips}"]].copy()
    pot_cells = set(store.field_cell.values())
    cells["has_potato"] = cells.cell_id.isin(pot_cells)
    nights = store.nights[store.nights.cell_id.isin(cells.cell_id)].reset_index(drop=True)
    nights.to_parquet(out / "nights.parquet")
    cells[["cell_id", "has_potato", "lon", "lat", "geometry"]].to_crs(4326).to_file(
        out / "rtma_cells.geojson", driver="GeoJSON")

    pn = nights[nights.cell_id.isin(pot_cells)]
    g = pn.groupby("night")
    flagged = sorted(g.trigger.any().pipe(lambda s: s[s]).index.union(
        g.near_trigger.any().pipe(lambda s: s[s]).index))
    all_nights = sorted(nights.night.unique())

    ids = store.field_ids()[: a.fields] if a.fields else store.field_ids()
    rows, fdata = [], {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(cfg, a.method)) as ex:
        for k, (fid, r, d) in enumerate(ex.map(_field_job, [(i, all_nights, flagged, out) for i in ids]), 1):
            rows += r
            fdata[fid] = d
            print(f"  field {k}/{len(ids)} {fid}: {len(r)} field-nights", flush=True)
    fn = pd.DataFrame(rows)
    fn.to_parquet(out / "field_nights.parquet")
    (out / "fields_detail.json").write_text(json.dumps(fdata))

    fl = store.fields[store.fields.field_id.isin(ids)].copy()
    fl["cell_id"] = fl.field_id.map(store.field_cell)
    fl[["field_id", "area_ha", "cell_id", "geometry"]].to_crs(4326).to_file(
        out / "fields.geojson", driver="GeoJSON")
    pd.read_parquet(c / f"s2_acquisitions_{fips}.parquet").to_parquet(out / "s2_acquisitions.parquet")

    hs, bounds = dem.hillshade(cfg, fips)
    Image.fromarray(np.dstack([hs, np.where(hs > 0, 255, 0).astype("uint8")]), "LA").save(
        out / "hillshade.png", optimize=True)

    meta = dict(run_id=run_id, method=a.method, county_fips=fips, git_commit=git_commit(),
                started=started, finished=dt.datetime.now().isoformat(timespec="seconds"),
                hillshade_bounds=bounds, flagged_nights=flagged, n_fields=len(ids),
                n_field_nights=len(fn), n_rendered=int(fn.rendered.sum()) if len(fn) else 0,
                image_ranges=dict(evi2=EVI2_RANGE, z=Z_RANGE),
                reason_codes=methods.load(a.method).REASONS, reason_colors=REASON_COLORS)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    from report.build import build
    print("report:", build(out))


if __name__ == "__main__":
    main()
