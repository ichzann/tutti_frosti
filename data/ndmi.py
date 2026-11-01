"""Sentinel-2 NDMI (moisture) quicklooks for the report, only for dates a run already shows.

    python -m data.ndmi runs/<id>      (also called by run.py; skips existing PNGs)

NDMI = (B8A - B11) / (B8A + B11), both 20 m bands read onto the field's 10 m grid (nearest).
For every img/<field>/rgb_<date>.png of the run, writes img/<field>/ndmi_<date>.png: clear
(SCL 4/5) pixels of the field's bounding box coloured BrBG over NDMI_RANGE, the field mask
outlined in black.  Display only: the method does not use NDMI.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import yaml
from matplotlib import colormaps
from PIL import Image
from shapely.geometry import box, shape

from .common import cache_dir
from .fields import field_mask
from .s2 import _offset, _read, clusters, search_items

NDMI_RANGE = (-0.2, 0.6)   # brown (dry / bare soil) -> green-blue (moist canopy)


def ndmi_rgba(ndmi, mask):
    v = np.clip((ndmi - NDMI_RANGE[0]) / (NDMI_RANGE[1] - NDMI_RANGE[0]), 0, 1)
    rgba = (colormaps["BrBG"](np.nan_to_num(v)) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isfinite(ndmi), 255, 0)
    p = np.pad(mask, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    rgba[mask & ~inner] = (0, 0, 0, 255)   # outline of the field pixels
    return rgba


def _date_job(date, items, cl_list, fields, clear_scl):
    """{field index: NDMI (ny, nx)} for one acquisition date (first clear item wins per pixel)."""
    out = {}
    for item in items:
        item = pc.sign(item)
        foot = shape(item.geometry)
        for bounds, idx in cl_list:
            bb = gpd.GeoSeries([box(*bounds)], crs="EPSG:32612").to_crs(4326).iloc[0]
            if not foot.intersects(bb):
                continue
            H = int(round((bounds[3] - bounds[1]) / 10)); W = int(round((bounds[2] - bounds[0]) / 10))
            clear = np.isin(_read(item.assets["SCL"].href, bounds, (H, W)), clear_scl)
            if not clear.any():
                continue
            off = _offset(item)
            b8a, b11 = ((_read(item.assets[b].href, bounds, (H, W)).astype("float32") + off) / 1e4
                        for b in ("B8A", "B11"))
            with np.errstate(invalid="ignore", divide="ignore"):
                nd = np.where(clear & (b8a + b11 > 0), (b8a - b11) / (b8a + b11), np.nan)
            for i in idx:
                r = fields.iloc[i]
                ox = int(round((r.x0 - bounds[0]) / 10)); oy = int(round((bounds[3] - r.y0) / 10))
                v = nd[oy:oy + int(r.ny), ox:ox + int(r.nx)]
                o = out.setdefault(i, np.full(v.shape, np.nan, "float32"))
                fill = np.isnan(o)
                o[fill] = v[fill]
    return date, out


def render(run_dir, workers=6, log=print):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    fips = cfg["study"]["county_fips"]
    need = {}   # date "YYYYMMDD" -> set of field ids
    for p in (run_dir / "img").glob("*/rgb_*.png"):
        d = p.stem[4:]
        if not (p.parent / f"ndmi_{d}.png").exists():
            need.setdefault(d, set()).add(p.parent.name)
    if not need:
        return 0
    fl = gpd.read_file(cache_dir(cfg) / f"fields_used_{fips}.gpkg")
    by_date = {}
    for it in search_items(cfg, fips):
        d = it.datetime.date().strftime("%Y%m%d")
        crs = it.properties.get("proj:code") or f"EPSG:{it.properties['proj:epsg']}"
        if d in need and crs == "EPSG:32612":
            by_date.setdefault(d, []).append(it)
    log(f"NDMI: {sum(map(len, need.values()))} field-dates on {len(need)} dates")
    jobs = []
    for d, fids in need.items():
        sub = fl[fl.field_id.isin(fids)].reset_index(drop=True)
        jobs.append((d, by_date.get(d, []), clusters(sub), sub))
    n = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_date_job, d, its, cl, sub, cfg["data"]["s2_clear_scl"]) for d, its, cl, sub in jobs]
        for f, (_, _, _, sub) in zip(futs, jobs):
            d, res = f.result()
            for i, nd in res.items():
                r = sub.iloc[i]
                Image.fromarray(ndmi_rgba(nd, field_mask(cfg, r)), "RGBA").save(
                    run_dir / "img" / r.field_id / f"ndmi_{d}.png", optimize=True)
                n += 1
            log(f"  NDMI {d}: {len(res)} fields")
    return n


if __name__ == "__main__":
    print(render(sys.argv[1]), "NDMI images")
