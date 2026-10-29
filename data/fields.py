"""Field definition: CSB potato polygon, 10 m inward buffer, >= 4 ha, CDL potato & confidence mask.

Each field gets its own 10 m pixel grid (snapped to the Sentinel-2 UTM 12N grid).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from . import cdl, csb
from .common import cache_dir

PAD = 2  # pixels of padding around the buffered polygon


def county_fields(cfg, fips: str | None = None) -> gpd.GeoDataFrame:
    """Buffered potato fields >= min area, with grid shape/origin. Cached."""
    fips = fips or cfg["study"]["county_fips"]
    fp = cache_dir(cfg) / f"fields_{fips}.gpkg"
    if fp.exists():
        return gpd.read_file(fp)
    d = cfg["data"]
    raw = csb.potato_fields(cfg, fips)
    f = raw.copy()
    f["area_raw_ha"] = raw.area / 1e4
    f["geometry"] = raw.buffer(-d["csb_inward_buffer_m"])
    f = f[~f.geometry.is_empty]
    f["area_ha"] = f.area / 1e4
    f = f[f.area_ha >= d["min_field_area_ha"]].copy()
    f["field_id"] = f["CSBID"].astype(str)
    res = d["s2_resolution_m"]
    b = f.bounds
    f["x0"] = np.floor(b.minx / res) * res - PAD * res
    f["y0"] = np.ceil(b.maxy / res) * res + PAD * res
    f["nx"] = ((np.ceil(b.maxx / res) * res + PAD * res - f.x0) / res).astype(int)
    f["ny"] = ((f.y0 - (np.floor(b.miny / res) * res - PAD * res)) / res).astype(int)
    keep = ["field_id", "CSBID", "GEOID", "area_raw_ha", "area_ha", "x0", "y0", "nx", "ny", "geometry"]
    f = f[keep].reset_index(drop=True)
    f.to_file(fp, driver="GPKG")
    return f


def field_transform(row, res=10):
    return from_origin(row.x0, row.y0, res, res)


def polygon_mask(cfg, row) -> np.ndarray:
    """Boolean (ny, nx): pixel centre inside the buffered CSB polygon."""
    tr = field_transform(row, cfg["data"]["s2_resolution_m"])
    return rasterize([(row.geometry, 1)], out_shape=(int(row.ny), int(row.nx)), transform=tr,
                     fill=0, dtype="uint8").astype(bool)


def field_mask(cfg, row) -> np.ndarray:
    """Boolean (ny, nx): inside buffered polygon AND CDL potato AND confidence >= threshold."""
    fp = cache_dir(cfg, "masks") / f"{row.field_id}.npy"
    if fp.exists():
        return np.load(fp)
    d = cfg["data"]
    tr = field_transform(row, d["s2_resolution_m"])
    shape = (int(row.ny), int(row.nx))
    poly = polygon_mask(cfg, row)
    cls_fp, conf_fp = cdl.county_cdl(cfg, row.GEOID)
    out = {}
    for k, p in (("cls", cls_fp), ("conf", conf_fp)):
        dst = np.zeros(shape, dtype="uint8")
        with rasterio.open(p) as src:
            reproject(rasterio.band(src, 1), dst, dst_transform=tr, dst_crs=d["work_crs"],
                      resampling=Resampling.nearest)
        out[k] = dst
    m = poly & (out["cls"] == d["cdl_potato_class"]) & (out["conf"] >= d["cdl_min_confidence"])
    np.save(fp, m)
    return m
