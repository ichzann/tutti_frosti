"""Swiss benchmark inputs (Canton Zurich 2021, maize) in the layout data.store.FieldStore reads.

    python -m validation.swiss_hail.data        # fields -> Sentinel-2 (SCL, 10 bands, AOT) -> field QA -> nights

Everything goes to cache/swiss_hail/zh2021/ (cfg["paths"]["cache"] of ch_config), so the method caches
(model, calibration) of this benchmark never mix with the Bingham ones.

Data rules follow the Bingham pipeline (data.fields / data.s2 / data.s2bands / data.s2qa) except where the
US source has no Swiss counterpart:
  fields   Canton Zurich OGD WFS, Nutzungsflaechen 2021 (farmer declarations, MGDM 153.1), BLW codes 0508
           Koernermais + 0521 Silo- und Gruenmais; 10 m inward buffer as csb_inward_buffer_m.  Size floor
           1 ha after buffering (= frost_recovery min_valid_px, 100 px) instead of min_field_area_ha 4 ha:
           Zurich maize parcels have a median of 0.9 ha, only 16 are >= 4 ha.  No CDL step: the declared
           crop is the crop, mask = polygon.
  grid     10 m, snapped to the Sentinel-2 UTM 32N grid (EPSG:32632; Zurich lies in tile 32TMT).
  S2       Planetary Computer L2A, SCL 4/5 clear, BOA offset, 20 m bands nearest to 10 m (as data.s2bands).
  QA       data.s2qa rules (clear share, HOT haze on closed canopy, AOT otherwise) with the Zurich clear line.
  nights   no RTMA: one row per (radar pixel, day) with trigger = POH > 0 within HAIL_RADIUS_M on one of the
           12 radar hail days (validation.swiss_hail.truth), tmin_c a constant above min_tmin_c.  This makes
           frost_recovery's "frost-free" training / calibration intervals "radar-hail-free" intervals.
"""
from __future__ import annotations

import copy
import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import requests
from scipy.spatial import cKDTree
from shapely.geometry import box, shape

from data.common import ROOT, cache_dir, load_config
from data.fields import PAD, polygon_mask
from data.s2 import _offset, _read, clusters
from data.s2qa import clear_line
from data.s2bands import BANDS
from data.store import FieldStore

from . import truth

FIPS = "ZH2021mais"
CRS = "EPSG:32632"
SEASON = ("2021-04-15", "2021-10-31")
MAIZE = {"0508", "0521"}
MIN_HA = 1.0
HAIL_RADIUS_M = 2000
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
WFS = ("https://maps.zh.ch/wfs/OGDZHWFS?service=WFS&version=2.0.0&request=GetFeature"
       "&typeNames=ms:ogd-0170_giszhpub_lw_nutzungsflaechen_2021_f&outputFormat=geojson")
AOT_STEP = 4

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")


def ch_config(cfg=None):
    cfg = copy.deepcopy(cfg or load_config())
    cfg["paths"]["cache"] = "cache/swiss_hail/zh2021"
    cfg["study"].update(county_fips=FIPS, year=2021, season_start=SEASON[0], season_end=SEASON[1],
                        timezone="Europe/Zurich")
    cfg["data"]["work_crs"] = CRS
    return cfg


# ------------------------------------------------------------------------------------- fields --
def fields(cfg):
    fp = cache_dir(cfg) / f"fields_used_{FIPS}.gpkg"
    if fp.exists():
        return gpd.read_file(fp)
    raw_fp = cache_dir(cfg) / "nutzungsflaechen_2021.geojson"
    if not raw_fp.exists():
        r = requests.get(WFS, timeout=1800)
        r.raise_for_status()
        raw_fp.write_bytes(r.content)
    g = gpd.read_file(raw_fp)
    g = g[g.blw_nr.isin(MAIZE)].to_crs(CRS)
    f = g[["gis_nr", "blw_nr", "blw_name", "bfsnr", "gembez"]].copy()
    f["area_raw_ha"] = g.area / 1e4
    f["geometry"] = g.buffer(-cfg["data"]["csb_inward_buffer_m"])
    f = gpd.GeoDataFrame(f, geometry="geometry", crs=CRS)
    f = f[~f.geometry.is_empty]
    f["area_ha"] = f.area / 1e4
    f = f[f.area_ha >= MIN_HA].copy()
    f["field_id"] = "ZH" + f.gis_nr.astype(str)
    f["GEOID"] = FIPS
    res = cfg["data"]["s2_resolution_m"]
    b = f.bounds
    f["x0"] = np.floor(b.minx / res) * res - PAD * res
    f["y0"] = np.ceil(b.maxy / res) * res + PAD * res
    f["nx"] = ((np.ceil(b.maxx / res) * res + PAD * res - f.x0) / res).astype(int)
    f["ny"] = ((f.y0 - (np.floor(b.miny / res) * res - PAD * res)) / res).astype(int)
    f = f[["field_id", "gis_nr", "blw_nr", "blw_name", "bfsnr", "gembez", "GEOID", "area_raw_ha", "area_ha",
           "x0", "y0", "nx", "ny", "geometry"]].reset_index(drop=True)
    f.to_file(fp, driver="GPKG")
    return f


# ----------------------------------------------------------------------------------- sentinel-2 --
def search_items(f):
    bb = gpd.GeoSeries([box(*f.total_bounds)], crs=CRS).to_crs(4326).iloc[0]
    cat = pystac_client.Client.open(STAC)
    items = cat.search(collections=["sentinel-2-l2a"], intersects=bb.__geo_interface__,
                       datetime=f"{SEASON[0]}/{SEASON[1]}").items()
    best = {}
    for it in items:
        p = it.properties
        if (p.get("proj:code") or f"EPSG:{p.get('proj:epsg')}") != CRS:
            continue
        k = (p["s2:mgrs_tile"], it.datetime)
        if k not in best or p.get("s2:processing_baseline", "") > best[k].properties.get("s2:processing_baseline", ""):
            best[k] = it
    return sorted(best.values(), key=lambda i: i.datetime)


def _item(item, cl_list, f, masks, clear_scl):
    """[(field index, clear (H, W), refl (10, H, W) uint16 or None, aot)] for one item."""
    item = pc.sign(item)
    foot = shape(item.geometry)
    out = []
    for bounds, idx in cl_list:
        bb = gpd.GeoSeries([box(*bounds)], crs=CRS).to_crs(4326).iloc[0]
        if not foot.intersects(bb):
            continue
        H = int(round((bounds[3] - bounds[1]) / 10)); W = int(round((bounds[2] - bounds[0]) / 10))
        scl = _read(item.assets["SCL"].href, bounds, (H, W))
        clear = np.isin(scl, clear_scl)
        sub = {}
        for i in idx:
            r = f.iloc[i]
            ox = int(round((r.x0 - bounds[0]) / 10)); oy = int(round((bounds[3] - r.y0) / 10))
            sub[i] = (slice(oy, oy + int(r.ny)), slice(ox, ox + int(r.nx)))
        covered = {i: ((scl[sub[i]] > 0) & masks[i]).any() for i in idx}
        stack = aot = None
        if any((clear[sub[i]] & masks[i]).any() for i in idx):
            off = _offset(item)
            stack = np.stack([np.clip(_read(item.assets[b].href, bounds, (H, W)).astype("float32") + off, 0, 65535)
                              for b in BANDS]).astype("uint16")
            aot = _read(item.assets["AOT"].href, bounds,
                        (max(H // AOT_STEP, 1), max(W // AOT_STEP, 1))).astype("float32")
            aot[aot == 0] = np.nan
        for i in idx:
            if not covered[i]:
                continue
            c = clear[sub[i]] & masks[i]
            a = np.nan
            if stack is not None and c.any():
                r = f.iloc[i]
                ox = int(round((r.x0 - bounds[0]) / 10)) // AOT_STEP
                oy = int(round((bounds[3] - r.y0) / 10)) // AOT_STEP
                v = aot[oy:oy + max(int(r.ny) // AOT_STEP, 1), ox:ox + max(int(r.nx) // AOT_STEP, 1)]
                a = float(np.nanmedian(v)) / 1000.0 if np.isfinite(v).any() else np.nan
                out.append((i, c, stack[(slice(None),) + sub[i]], a))
            else:
                out.append((i, np.zeros_like(c), None, np.nan))
    return item, out


def _item_retry(item, *a, tries=3):
    for k in range(tries):
        try:
            return _item(item, *a)
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  retry {item.id}: {e!r}", flush=True)


def sentinel2(cfg, workers=8, log=print):
    """cache/.../s2/<fid>.npz (dates, evi2, rgb, covered, poly) and s2bands/<fid>.npz (refl), plus a field-date
    table for QA (clear share, field-median blue / red / EVI2, AOT)."""
    rows_fp = cache_dir(cfg) / f"s2_rows_{FIPS}.parquet"
    if rows_fp.exists():
        return pd.read_parquet(rows_fp)
    f = fields(cfg)
    masks = [polygon_mask(cfg, r) for r in f.itertuples()]
    cl_list = clusters(f)
    items = search_items(f)
    log(f"S2: {len(f)} fields in {len(cl_list)} clusters, {len(items)} items")
    per = {i: {} for i in range(len(f))}               # i -> date -> (clear, refl, aot)
    clear_scl = cfg["data"]["s2_clear_scl"]
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_item_retry, it, cl_list, f, masks, clear_scl) for it in items]
        for n, fu in enumerate(futs, 1):
            item, res = fu.result()
            d = item.datetime.date().isoformat()
            for i, c, st, a in res:
                old = per[i].get(d)
                if old is None or (st is not None and (old[1] is None or c.sum() > old[0].sum())):
                    per[i][d] = (c, st, a)
            if n % 10 == 0:
                log(f"  S2 items {n}/{len(items)}", flush=True)
    s2_dir, b_dir = cache_dir(cfg, "s2"), cache_dir(cfg, "s2bands")
    rows = []
    for i, r in enumerate(f.itertuples()):
        dates = sorted(per[i])
        H, W = int(r.ny), int(r.nx)
        T = len(dates)
        refl = np.zeros((T, len(BANDS), H, W), "uint16")
        ev = np.full((T, H, W), np.nan, "float32")
        rgb = np.zeros((T, 3, H, W), "uint8")
        cov = np.zeros((T, H, W), bool)
        m = masks[i]
        for t, d in enumerate(dates):
            c, st, a = per[i][d]
            cov[t] = m
            cs = float(c.sum() / m.sum())
            blue = red = e = np.nan
            if st is not None and c.any():
                refl[t][:, c] = st[:, c]
                x = st.astype("float32") / 10000.0
                ev[t][c] = (2.5 * (x[6] - x[2]) / (x[6] + 2.4 * x[2] + 1.0))[c]
                rgb[t] = np.clip(x[[2, 1, 0]] / 0.3 * 255, 0, 255).astype("uint8")
                blue, red, e = float(np.median(x[0][c])), float(np.median(x[2][c])), float(np.median(ev[t][c]))
            rows.append((r.field_id, d, cs, blue, red, e, a))
        np.savez_compressed(s2_dir / f"{r.field_id}.npz", dates=np.array(dates), evi2=ev,
                            rgb=rgb, covered=cov, poly=m)
        np.savez_compressed(b_dir / f"{r.field_id}.npz", dates=np.array(dates), bands=np.array(BANDS), refl=refl)
    df = pd.DataFrame(rows, columns=["field_id", "date", "clear_share", "blue", "red", "evi2", "aot"])
    df.to_parquet(rows_fp)
    return df


def qa(cfg, log=print):
    """data.s2qa rules on the Zurich field-dates -> cache/.../s2_qa_<FIPS>.parquet."""
    fp = cache_dir(cfg) / f"s2_qa_{FIPS}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    q = cfg["s2_qa"]
    df = sentinel2(cfg, log=log).copy()
    veg = df.evi2 >= q["hot_min_evi2"]
    a, b = clear_line(df, q["min_field_clear_share"], q["hot_min_evi2"])
    th = np.arctan(b)
    df["hot"] = (df.blue - a - b * df.red) * np.cos(th)
    ref = (df[(df.clear_share >= q["min_field_clear_share"]) & veg].groupby("field_id").hot.median().rename("hot_ref"))
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


# ------------------------------------------------------------------------------------- nights --
def nights(cfg):
    """Radar pixel per field + (cell_id, night) table; trigger = POH > 0 within HAIL_RADIUS_M on a radar
    hail day.  Days outside the 12 radar days count as hail-free (the insurer data cover only these days)."""
    fp = cache_dir(cfg) / "nights.parquet"
    cf = cache_dir(cfg) / "field_cell.parquet"
    if fp.exists() and cf.exists():
        return pd.read_parquet(fp), pd.read_parquet(cf)
    truth.download()
    lat, lon, (hd, hi, hp), hdates = truth.read_hazard("POH")
    f = fields(cfg)
    pts = f.representative_point().to_crs(4326)
    xy = np.column_stack([lat, lon])
    tree = cKDTree(xy)
    cell = tree.query(np.column_stack([pts.y, pts.x]))[1]
    fc = pd.DataFrame({"field_id": f.field_id, "cell_id": cell.astype(int)})
    ucell = np.unique(cell)
    # local metric neighbourhood around each used radar pixel
    k = np.cos(np.radians(47.4))
    m_xy = np.column_stack([xy[:, 0] * 111_320, xy[:, 1] * 111_320 * k])
    near = cKDTree(m_xy).query_ball_point(m_xy[ucell], HAIL_RADIUS_M)
    hail = {}
    for e, d in enumerate(hdates):
        grid = np.zeros(len(lat), "float32")
        grid[hi[hp[e]:hp[e + 1]]] = hd[hp[e]:hp[e + 1]]
        hail[d] = {c: bool(grid[nb].max() > 0) for c, nb in zip(ucell, near)}
    days = pd.date_range(SEASON[0], SEASON[1], freq="D").date
    rows = [(int(c), d.isoformat(), bool(hail.get(d, {}).get(c, False))) for c in ucell for d in days]
    n = pd.DataFrame(rows, columns=["cell_id", "night", "trigger"])
    n["tmin_c"] = 20.0                   # no temperature gate: only the radar hail days define "event"
    n["near_trigger"] = False
    n["regime"] = "hail"
    n["dose_degree_hours"] = 0.0
    n.to_parquet(fp)
    fc.to_parquet(cf)
    return n, fc


# -------------------------------------------------------------------------------------- store --
class SwissStore(FieldStore):
    """FieldStore over the Zurich cache: polygon mask (no CDL), radar-hail nights, no gridMET."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.fips = FIPS
        c = cache_dir(cfg)
        f = gpd.read_file(c / f"fields_used_{FIPS}.gpkg")
        self._mask = {r.field_id: polygon_mask(cfg, r) for r in f.itertuples()}
        self.fields = f[[self._mask[i].any() for i in f.field_id]].reset_index(drop=True)
        self.nights, fc = nights(cfg)
        self.field_cell = dict(zip(fc.field_id, fc.cell_id))
        self._n = self.nights.set_index(["cell_id", "night"])
        qa_fp = c / f"s2_qa_{FIPS}.parquet"
        q = pd.read_parquet(qa_fp) if cfg.get("s2_qa", {}).get("enabled", True) and qa_fp.exists() else None
        self._drop = ({} if q is None else
                      q[~q.usable].groupby("field_id").apply(lambda g: dict(zip(g.date, g.drop_reason))).to_dict())

    def daily_temp(self, field_id):
        """No stage clock in the benchmark (one generic template set): a constant placeholder series."""
        d = pd.date_range(SEASON[0], SEASON[1], freq="D").date
        return pd.DataFrame({"tmax_c": 25.0, "tmin_c": 12.0}, index=[x.isoformat() for x in d])

    def night_meta(self, field_id, night):
        cell = self.field_cell[field_id]
        r = self._n.loc[(cell, dt.date.fromisoformat(str(night)).isoformat())]
        return dict(cell_id=int(cell), dose=0.0, tmin_c=float("nan"), trigger=bool(r.trigger),
                    near_trigger=False, regime="hail")


def build(log=print):
    cfg = ch_config()
    f = fields(cfg)
    log(f"fields: {len(f)} maize parcels >= {MIN_HA} ha after the inward buffer")
    sentinel2(cfg, log=log)
    qa(cfg, log=log)
    n, fc = nights(cfg)
    log(f"nights: {fc.cell_id.nunique()} radar cells, {int(n.trigger.sum())} hail cell-days")
    return cfg


if __name__ == "__main__":
    build(log=lambda *a, **k: print(*a, flush=True))
