"""Field-level Sentinel-2 acquisition QA: whole field-dates are kept or dropped.

    python -m data.s2qa            (also called by data.prepare; cached)

Per field and acquisition date (cache/s2_qa_<fips>.parquet):
  clear_share  share of the field's pixels (final CDL mask) that are SCL-clear
  aot          field-median Sen2Cor aerosol optical thickness (L2A AOT band / 1000), read at ~40 m.
               An independent haze / smoke estimate from the processor, not derived from the canopy.
  hot          Haze Optimized Transform (Zhang et al. 2002, Remote Sens. Environ. 82): distance of the
               field-median (blue, red) point above the county clear-sky line.  Haze scatters blue more
               than red and lifts points above the line; frost-damaged canopy brightens red more than
               blue and moves them below it, so HOT separates haze from damage, unlike an EVI2 drop.
  hot_anom     hot minus the field's own median hot over its clearest dates (removes soil / canopy
               offsets of the field)
HOT is only meaningful over canopy (bright bare soil sits above the clear line too), so the line
and the field reference use vegetated dates (field-median EVI2 >= hot_min_evi2) only.
A field-date is usable when clear_share >= min_field_clear_share and
  vegetated:      hot_anom <= hot_anom_max   (AOT alone is not used: L2A reflectance is already
                  corrected for the AOT Sen2Cor estimated; 2024-07-23, AOT 0.45 smoke, HOT 0, EVI2 fine)
  not vegetated:  aot <= aot_max             (fallback, HOT not usable)
otherwise drop_reason says why and data.store hides that date
(all its EVI2 pixels set to NaN) so every used acquisition shows the same, whole field.
Blue / red come from the cached 8-bit quicklook (reflectance = DN * 0.3 / 255, step 0.0012).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
from shapely.geometry import box, shape

from .common import cache_dir, load_config
from .fields import field_mask
from .s2 import _read, clusters, search_items

AOT_STEP = 4   # read AOT on a 40 m grid (COG overviews); it is a smooth field anyway


def _aot_item(item, cl_list, fields):
    """[(field index, date, field-median AOT)] for one S2 item."""
    item = pc.sign(item)
    p = item.properties
    if (p.get("proj:code") or f"EPSG:{p['proj:epsg']}") != "EPSG:32612":
        return []
    foot = shape(item.geometry)
    d = item.datetime.date().isoformat()
    out = []
    for bounds, idx in cl_list:
        bb = gpd.GeoSeries([box(*bounds)], crs="EPSG:32612").to_crs(4326).iloc[0]
        if not foot.intersects(bb):
            continue
        H = int(round((bounds[3] - bounds[1]) / 10)); W = int(round((bounds[2] - bounds[0]) / 10))
        a = _read(item.assets["AOT"].href, bounds,
                  (max(H // AOT_STEP, 1), max(W // AOT_STEP, 1))).astype("float32")
        a[a == 0] = np.nan   # nodata outside the swath
        for i in idx:
            r = fields.iloc[i]
            ox = int(round((r.x0 - bounds[0]) / 10)) // AOT_STEP
            oy = int(round((bounds[3] - r.y0) / 10)) // AOT_STEP
            v = a[oy:oy + max(int(r.ny) // AOT_STEP, 1), ox:ox + max(int(r.nx) // AOT_STEP, 1)]
            if np.isfinite(v).any():
                out.append((i, d, float(np.nanmedian(v)) / 1000.0))
    return out


def _aot(cfg, fips, fields, workers=6, log=print):
    cl = clusters(fields)
    items = search_items(cfg, fips)
    log(f"S2 QA: AOT for {len(fields)} fields, {len(items)} items")
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for n, res in enumerate(ex.map(lambda it: _aot_item(it, cl, fields), items), 1):
            rows += res
            if n % 20 == 0:
                log(f"  AOT items {n}/{len(items)}")
    df = pd.DataFrame(rows, columns=["i", "date", "aot"])
    # two tiles on one date: keep the lower value (the overlap is the same sky)
    df = df.groupby(["i", "date"], as_index=False).aot.min()
    df["field_id"] = fields.field_id.values[df.i]
    return df.drop(columns="i")


def _field_rows(cfg, fields):
    rows = []
    for r in fields.itertuples():
        m = field_mask(cfg, r)
        if not m.any():
            continue
        z = np.load(cache_dir(cfg, "s2") / f"{r.field_id}.npz")
        ev, rgb = z["evi2"], z["rgb"]
        for t, d in enumerate(z["dates"]):
            c = np.isfinite(ev[t]) & m
            cs = float(c.sum() / m.sum())
            red = blue = e = np.nan
            if c.any():
                red = float(np.median(rgb[t, 0][c])) * 0.3 / 255
                blue = float(np.median(rgb[t, 2][c])) * 0.3 / 255
                e = float(np.median(ev[t][c]))
            rows.append((r.field_id, str(d), cs, blue, red, e))
    return pd.DataFrame(rows, columns=["field_id", "date", "clear_share", "blue", "red", "evi2"])


def clear_line(df, min_clear, min_evi2):
    """County clear-sky line blue = a + b * red: lower-envelope fit (5th percentile of blue in
    red bins) over the clearest vegetated field-dates."""
    s = df[(df.clear_share >= min_clear) & (df.evi2 >= min_evi2) & df.blue.notna()]
    bins = np.quantile(s.red, np.linspace(0, 1, 21))
    x, y = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = s[(s.red >= lo) & (s.red <= hi)]
        if len(b) >= 20:
            x.append(b.red.median()); y.append(b.blue.quantile(0.05))
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def build(cfg, fips=None, log=print, force=False):
    fips = fips or cfg["study"]["county_fips"]
    q = cfg["s2_qa"]
    fp = cache_dir(cfg) / f"s2_qa_{fips}.parquet"
    if fp.exists() and not force:
        return pd.read_parquet(fp)
    fields = gpd.read_file(cache_dir(cfg) / f"fields_used_{fips}.gpkg")
    df = _field_rows(cfg, fields)
    aot_fp = cache_dir(cfg) / f"s2_aot_{fips}.parquet"
    if not aot_fp.exists():
        _aot(cfg, fips, fields, log=log).to_parquet(aot_fp)
    df = df.merge(pd.read_parquet(aot_fp), on=["field_id", "date"], how="left")

    veg = df.evi2 >= q["hot_min_evi2"]
    a, b = clear_line(df, q["min_field_clear_share"], q["hot_min_evi2"])
    th = np.arctan(b)   # HOT: perpendicular distance above the line blue = a + b * red
    df["hot"] = (df.blue - a - b * df.red) * np.cos(th)
    ref = (df[(df.clear_share >= q["min_field_clear_share"]) & veg]
           .groupby("field_id").hot.median().rename("hot_ref"))
    df = df.join(ref, on="field_id")
    df["hot_anom"] = df.hot - df.hot_ref

    reason = np.where(df.clear_share < q["min_field_clear_share"], "cloud",
             np.where(veg & (df.hot_anom > q["hot_anom_max"]), "haze",
             np.where(~veg & (df.aot > q["aot_max"]), "aot", "")))
    df["drop_reason"] = reason
    df["usable"] = reason == ""
    df.to_parquet(fp)
    log(f"S2 QA: clear line blue = {a:.4f} + {b:.3f} red; "
        + ", ".join(f"{k or 'usable'} {v}" for k, v in df.drop_reason.value_counts().items()))
    return df


if __name__ == "__main__":
    cfg = load_config()
    build(cfg, force="--force" in sys.argv)
