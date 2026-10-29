"""Sentinel-2 L2A (Planetary Computer) -> per-field clear-observation stacks.

For every acquisition, SCL is read first over small windows around field
clusters; the reflectance bands are read only where at least one field pixel
is clear (SCL 4/5).  Per field we store, per acquisition date: EVI2 (NaN where
not clear), an 8-bit RGB quicklook and the clear mask.
"""
from __future__ import annotations

import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box, shape
from shapely.ops import unary_union
import geopandas as gpd

from .common import cache_dir, county
from .fields import county_fields, field_mask

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


def evi2(red, nir):
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)


def search_items(cfg, fips):
    c = county(cfg, fips).to_crs(4326)
    start = cfg["study"]["season_start"]
    end = (dt.date.fromisoformat(cfg["study"]["season_end"])
           + dt.timedelta(days=cfg["gpr"]["max_gap_after_days"] + 15)).isoformat()
    cat = pystac_client.Client.open(STAC)
    items = list(cat.search(collections=[cfg["data"]["s2_collection"]],
                            intersects=c.geometry.iloc[0].__geo_interface__,
                            datetime=f"{start}/{end}").items())
    # drop duplicate (tile, sensing time) entries, keep the latest processing
    best = {}
    for it in items:
        k = (it.properties["s2:mgrs_tile"], it.datetime)
        if k not in best or it.properties.get("s2:processing_baseline", "") > \
                best[k].properties.get("s2:processing_baseline", ""):
            best[k] = it
    return sorted(best.values(), key=lambda i: i.datetime)


def clusters(fields: gpd.GeoDataFrame, gap_m=300):
    """Group fields whose padded bboxes lie within gap_m; returns list of (bbox, field indices)."""
    boxes = [box(r.x0, r.y0 - r.ny * 10, r.x0 + r.nx * 10, r.y0) for r in fields.itertuples()]
    merged = unary_union([b.buffer(gap_m / 2) for b in boxes])
    geoms = list(getattr(merged, "geoms", [merged]))
    out = []
    for g in geoms:
        idx = [i for i, b in enumerate(boxes) if g.intersects(b)]
        out.append((unary_union([boxes[i] for i in idx]).bounds, idx))
    return out


def _offset(item):
    """BOA_ADD_OFFSET introduced with processing baseline 04.00."""
    pb = float(item.properties.get("s2:processing_baseline", "0") or 0)
    return -1000.0 if pb >= 4.0 else 0.0


def _read(href, bounds, out_shape):
    with rasterio.open(href) as src:
        w = from_bounds(*bounds, transform=src.transform)
        return src.read(1, window=w, out_shape=out_shape, boundless=True, fill_value=0,
                        resampling=rasterio.enums.Resampling.nearest)


def _process_item(item, cl_list, fields, masks, clear_scl):
    item = pc.sign(item)
    foot = shape(item.geometry)
    p = item.properties
    crs = p.get("proj:code") or f"EPSG:{p['proj:epsg']}"
    res = []
    for bounds, idx in cl_list:
        if crs != "EPSG:32612":
            continue
        # cluster footprint test in lon/lat
        bb = gpd.GeoSeries([box(*bounds)], crs="EPSG:32612").to_crs(4326).iloc[0]
        if not foot.intersects(bb):
            continue
        H = int(round((bounds[3] - bounds[1]) / 10)); W = int(round((bounds[2] - bounds[0]) / 10))
        scl = _read(item.assets["SCL"].href, bounds, (H, W))
        clear = np.isin(scl, clear_scl)
        sub = {}
        any_clear = False
        for i in idx:
            r = fields.iloc[i]
            ox = int(round((r.x0 - bounds[0]) / 10)); oy = int(round((bounds[3] - r.y0) / 10))
            sl = (slice(oy, oy + int(r.ny)), slice(ox, ox + int(r.nx)))
            sub[i] = sl
            if (clear[sl] & masks[i]).any():
                any_clear = True
        n_field = sum(int(masks[i].sum()) for i in idx)
        n_clear = sum(int((clear[sub[i]] & masks[i]).sum()) for i in idx)
        n_cov = sum(int(((scl[sub[i]] > 0) & masks[i]).sum()) for i in idx)
        bands = {}
        if any_clear:
            off = _offset(item)
            for b, name in BANDS.items():
                v = _read(item.assets[b].href, bounds, (H, W)).astype("float32")
                bands[name] = (v + off) / 10000.0
        for i in idx:
            sl = sub[i]
            c = clear[sl] & masks[i]
            rec = {"clear": c, "covered": scl[sl] > 0}
            if bands and c.any():
                rec["evi2"] = evi2(bands["red"][sl], bands["nir"][sl]).astype("float16")
                rec["rgb"] = np.stack([np.clip(bands[b][sl] / 0.3 * 255, 0, 255)
                                       for b in ("red", "green", "blue")]).astype("uint8")
            res.append((i, rec))
        res.append(("_stats", (n_field, n_cov, n_clear)))
    return item, res


def build(cfg, fips: str | None = None, workers=6, log=print):
    """Write cache/s2/<field_id>.npz and cache/s2_acquisitions_<fips>.parquet."""
    fips = fips or cfg["study"]["county_fips"]
    out_dir = cache_dir(cfg, "s2")
    acq_fp = cache_dir(cfg) / f"s2_acquisitions_{fips}.parquet"
    if acq_fp.exists():
        return acq_fp
    fields = county_fields(cfg, fips)
    masks = [field_mask(cfg, r) for r in fields.itertuples()]
    keep = [i for i, m in enumerate(masks) if m.any()]
    fields = fields.iloc[keep].reset_index(drop=True)
    masks = [masks[i] for i in keep]
    cl_list = clusters(fields)
    items = search_items(cfg, fips)
    log(f"S2: {len(fields)} fields in {len(cl_list)} clusters, {len(items)} items")
    per_field = {i: {} for i in range(len(fields))}  # i -> {date: [recs]}
    acq = {}
    clear_scl = cfg["data"]["s2_clear_scl"]
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_process_item, it, cl_list, fields, masks, clear_scl) for it in items]
        for n, f in enumerate(futs, 1):
            item, res = f.result()
            d = item.datetime.date().isoformat()
            a = acq.setdefault(d, [0, 0, 0])
            for i, rec in res:
                if i == "_stats":
                    for k in range(3):
                        a[k] += rec[k]
                else:
                    per_field[i].setdefault(d, []).append(rec)
            if n % 20 == 0:
                log(f"  S2 items {n}/{len(items)}")
    for i, r in enumerate(fields.itertuples()):
        dates = sorted(per_field[i])
        H, W = int(r.ny), int(r.nx)
        T = len(dates)
        ev = np.full((T, H, W), np.nan, "float32")
        rgb8 = np.zeros((T, 3, H, W), "uint8")
        cov = np.zeros((T, H, W), bool)
        for t, d in enumerate(dates):
            for rec in per_field[i][d]:
                cov[t] |= rec["covered"]
                if "evi2" in rec:
                    c = rec["clear"] & np.isnan(ev[t])
                    ev[t][c] = rec["evi2"][c]
                    fill = (rgb8[t].sum(0) == 0) & (rec["rgb"].sum(0) > 0)
                    rgb8[t][:, fill] = rec["rgb"][:, fill]
        np.savez_compressed(out_dir / f"{r.field_id}.npz", dates=np.array(dates), evi2=ev,
                            rgb=rgb8, covered=cov, mask=masks[i])
    df = pd.DataFrame([(d, *v) for d, v in sorted(acq.items())],
                      columns=["date", "field_px", "covered_px", "clear_px"])
    df["clear_frac"] = df.clear_px / df.covered_px.where(df.covered_px > 0)
    df.to_parquet(acq_fp)
    fields.to_file(cache_dir(cfg) / f"fields_used_{fips}.gpkg", driver="GPKG")
    return acq_fp
