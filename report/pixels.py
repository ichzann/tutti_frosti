"""Per-pixel map layers: z rasters of rendered field-nights warped to Web Mercator, for Leaflet.

    python -m report.pixels runs/<id>      (also called by report.build; skips existing PNGs)

Writes img/<field>/<night>_zmap.png and returns {field_id: [south, west, north, east]}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import yaml
from PIL import Image
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

from data.common import cache_dir

Z_MIN = -4.0      # darkest red
WEB_RES_M = 5.0   # output pixel size (Mercator m), finer than 10 m so nearest-neighbour edges stay crisp
INVALID = (120, 120, 120, 110)   # field pixel without a valid z (semi-transparent grey)
NOT_LOW = (215, 225, 215, 200)   # z >= 0: no deficit (pale)
# z < 0: pale yellow -> orange -> red -> dark red as z goes 0 -> Z_MIN
STOPS = np.array([[255, 237, 160], [253, 141, 60], [215, 25, 28], [103, 0, 13]], float)


def z_rgba(z, valid, field):
    rgba = np.zeros(z.shape + (4,), "uint8")
    rgba[field] = INVALID
    neg = valid & (z < 0)
    rgba[valid & ~neg] = NOT_LOW
    t = np.clip(z[neg] / Z_MIN, 0, 1) * (len(STOPS) - 1)
    k = np.minimum(t.astype(int), len(STOPS) - 2)
    f = (t - k)[:, None]
    rgba[neg, :3] = (STOPS[k] + (STOPS[k + 1] - STOPS[k]) * f).round().astype("uint8")
    rgba[neg, 3] = 235
    return rgba


def _web_grid(h, w, src_tr, crs):
    dtr, dw, dh = calculate_default_transform(crs, "EPSG:3857", w, h, *array_bounds(h, w, src_tr),
                                              resolution=WEB_RES_M)
    w_, s_, e_, n_ = transform_bounds("EPSG:3857", "EPSG:4326", *array_bounds(dh, dw, dtr))
    return dtr, dw, dh, [round(v, 7) for v in (s_, w_, n_, e_)]


def _warp(rgba, src_tr, crs, dtr, dw, dh):
    out = np.zeros((4, dh, dw), "uint8")
    for b in range(4):
        reproject(np.ascontiguousarray(rgba[..., b]), out[b], src_transform=src_tr, src_crs=crs,
                  dst_transform=dtr, dst_crs="EPSG:3857", resampling=Resampling.nearest)
    return np.moveaxis(out, 0, -1)


def build_pixel_layers(run_dir) -> dict:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    res = cfg["data"]["s2_resolution_m"]
    crs = cfg["data"]["work_crs"]
    fl = gpd.read_file(cache_dir(cfg) / f"fields_used_{cfg['study']['county_fips']}.gpkg").set_index("field_id")
    bounds = {}
    for fdir in sorted((run_dir / "rasters").iterdir()):
        npzs = sorted(fdir.glob("*.npz"))
        if not npzs or fdir.name not in fl.index:
            continue
        row = fl.loc[fdir.name]
        tr = from_origin(row.x0, row.y0, res, res)
        dtr, dw, dh, bounds[fdir.name] = _web_grid(int(row.ny), int(row.nx), tr, crs)
        for p in npzs:
            out = run_dir / "img" / fdir.name / f"{p.stem}_zmap.png"
            if out.exists():
                continue
            r = np.load(p)
            rgba = z_rgba(r["z"], r["valid"], r["reason"] != 1)
            Image.fromarray(_warp(rgba, tr, crs, dtr, dw, dh), "RGBA").save(out, optimize=True)
    return bounds


if __name__ == "__main__":
    print(len(build_pixel_layers(sys.argv[1])), "fields")
