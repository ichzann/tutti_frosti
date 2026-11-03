"""Sentinel-2 L2A surface reflectance, all vegetation bands, per field (for multi-index methods).

    python -m data.s2bands [n]        # writes cache/s2bands/<field_id>.npz (skips existing; n: first n only)

Same acquisitions, cluster windows and SCL clear rule as data.s2; dates aligned with
cache/s2/<field_id>.npz.  Per field: refl (T, 10, H, W) uint16 = reflectance x 10000 after the
BOA offset (0 where not clear), band order BANDS.  20 m bands (B05-B07, B8A, B11, B12) are read
onto the 10 m grid with nearest resampling.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import planetary_computer as pc
from shapely.geometry import box, shape
import geopandas as gpd

from .common import cache_dir, load_config
from .fields import polygon_mask
from .s2 import _offset, _read, clusters, search_items

# a stalled HTTP read must fail (and be retried) instead of hanging the whole download
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_LOW_SPEED_TIME", "30")
os.environ.setdefault("GDAL_HTTP_LOW_SPEED_LIMIT", "1000")

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def _process_item(item, cl_list, fields, masks, clear_scl):
    item = pc.sign(item)
    foot = shape(item.geometry)
    p = item.properties
    if (p.get("proj:code") or f"EPSG:{p['proj:epsg']}") != "EPSG:32612":
        return item, []
    res = []
    for bounds, idx in cl_list:
        bb = gpd.GeoSeries([box(*bounds)], crs="EPSG:32612").to_crs(4326).iloc[0]
        if not foot.intersects(bb):
            continue
        H = int(round((bounds[3] - bounds[1]) / 10)); W = int(round((bounds[2] - bounds[0]) / 10))
        clear = np.isin(_read(item.assets["SCL"].href, bounds, (H, W)), clear_scl)
        sub = {}
        for i in idx:
            r = fields.iloc[i]
            ox = int(round((r.x0 - bounds[0]) / 10)); oy = int(round((bounds[3] - r.y0) / 10))
            sub[i] = (slice(oy, oy + int(r.ny)), slice(ox, ox + int(r.nx)))
        if not any((clear[sub[i]] & masks[i]).any() for i in idx):
            continue
        off = _offset(item)
        stack = np.stack([np.clip(_read(item.assets[b].href, bounds, (H, W)).astype("float32") + off, 0, 65535)
                          for b in BANDS]).astype("uint16")
        for i in idx:
            c = clear[sub[i]] & masks[i]
            if c.any():
                res.append((i, c, stack[(slice(None),) + sub[i]]))
    return item, res


def _process_retry(item, *a, tries=3):
    for k in range(tries):
        try:
            return _process_item(item, *a)
        except Exception as e:            # timeout / expired signature: re-sign and retry
            if k == tries - 1:
                raise
            print(f"  retry {item.id}: {e!r}", flush=True)


def build(cfg, fips: str | None = None, workers=8, log=print, limit=None, batch_fields=40):
    """Clusters are processed in batches of about batch_fields fields; each batch is written when done,
    so an interrupted build resumes at the next unwritten field."""
    fips = fips or cfg["study"]["county_fips"]
    out_dir = cache_dir(cfg, "s2bands")
    fields = gpd.read_file(cache_dir(cfg) / f"fields_used_{fips}.gpkg")
    todo = [i for i, r in enumerate(fields.itertuples()) if not (out_dir / f"{r.field_id}.npz").exists()]
    if not todo:
        return out_dir
    if limit:
        todo = todo[:limit]
    sub = fields.iloc[todo].reset_index(drop=True)
    masks = [polygon_mask(cfg, r) for r in sub.itertuples()]
    cl_all = clusters(sub)
    items = search_items(cfg, fips)
    batches, cur = [], []
    for cl in cl_all:
        cur.append(cl)
        if sum(len(c[1]) for c in cur) >= batch_fields:
            batches.append(cur); cur = []
    if cur:
        batches.append(cur)
    log(f"S2 bands: {len(sub)} fields in {len(cl_all)} clusters, {len(batches)} batches, {len(items)} items")
    clear_scl = cfg["data"]["s2_clear_scl"]
    for nb, cl_list in enumerate(batches, 1):
        ids = sorted(i for _, idx in cl_list for i in idx)
        dates = {i: [str(d) for d in np.load(cache_dir(cfg, "s2") / f"{sub.iloc[i].field_id}.npz")["dates"]]
                 for i in ids}
        refl = {i: np.zeros((len(dates[i]), len(BANDS), int(sub.iloc[i].ny), int(sub.iloc[i].nx)), "uint16")
                for i in ids}
        filled = {i: np.zeros((len(dates[i]), int(sub.iloc[i].ny), int(sub.iloc[i].nx)), bool) for i in ids}
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(_process_retry, it, cl_list, sub, masks, clear_scl) for it in items]
            for f in futs:
                item, res = f.result()
                d = item.datetime.date().isoformat()
                for i, c, st in res:
                    if d not in dates[i]:
                        continue
                    t = dates[i].index(d)
                    new = c & ~filled[i][t]
                    refl[i][t][:, new] = st[:, new]
                    filled[i][t] |= new
        for i in ids:
            np.savez_compressed(out_dir / f"{sub.iloc[i].field_id}.npz", dates=np.array(dates[i]),
                                bands=np.array(BANDS), refl=refl[i])
        log(f"  batch {nb}/{len(batches)}: {len(ids)} fields written", flush=True)
    return out_dir

if __name__ == "__main__":
    build(load_config(), log=lambda *a, **k: print(*a, flush=True),
          limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
