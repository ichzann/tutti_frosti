"""Brazil PSR benchmark inputs: insured policies (SISSER) -> windows around the property point -> Sentinel-2 ->
crop mask -> ERA5-Land nights, in the layout data.store.FieldStore reads.

    python -m validation.brazil_psr.data <case>     # case: wheat2019 | maize2021 | barley2021

Exploration round (user decisions 2026-11-04): frost_recovery only, no fitted calibration; compare the
method's field signal with end-of-season loss (indemnity / insured value) per policy.

Data rules (differences from Bingham / Zurich, where the US / Swiss source has no counterpart):
  policies  data/dados_abertos_psr_2016a2024csv.csv (partial browser download, 117,682 rows).  Personal
            columns (name, CPF / CNPJ) are never read.  Location = property point from the DMS columns
            (integer seconds, ~30 m).  Only points in UTM 22S (lon -54..-48) so one grid serves a case.
  fields    no polygons: a WINDOW_M x WINDOW_M window centred on the point (policies sharing a point share a
            window); the crop mask is derived from the season's Sentinel-2 EVI2 per pixel: low in the sowing
            window (max <= SOW_MAX) and green in the peak window (max >= PEAK_MIN).  Forest / pasture stay green
            all season and drop out; neighbouring fields of the same season crop are kept.
  S2        Planetary Computer L2A as validation.swiss_hail.data (SCL 4/5 clear, BOA offset, 20 m bands nearest).
  QA        data.s2qa rules on the crop-mask pixels.
  nights    ERA5-Land hourly 2 m temperature (Open-Meteo archive API) per 0.1 deg cell; night = local 18:00 on
            the date to 09:00 next day, tmin_c = minimum; trigger = tmin_c <= TRIGGER_C.  No RTMA regimes.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
from shapely.geometry import Point, box, shape

from data.common import ROOT, cache_dir, load_config
from data.s2 import _offset, _read
from data.s2bands import BANDS
from data.s2qa import clear_line
from data.store import FieldStore

RAW = ROOT / "data" / "dados_abertos_psr_2016a2024csv.csv"
CRS = "EPSG:32722"
LON_RANGE = (-54.0, -48.0)
WINDOW_M = 500
TRIGGER_C = 2.0
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
AOT_STEP = 4

# crop, policy year, S2 season, sowing window (low EVI2), peak window (green), max policies per claim class
CASES = {
    "wheat2019": dict(crop="Trigo", year="2019", season=("2019-03-15", "2019-10-15"),
                      sow=("2019-03-15", "2019-05-31"), peak=("2019-07-01", "2019-09-30")),
    "maize2021": dict(crop="Milho 2ª safra", year="2021", season=("2021-01-15", "2021-09-15"),
                      sow=("2021-01-15", "2021-03-31"), peak=("2021-04-01", "2021-06-30"),
                      cap={"other": 400}, states=("PR", "SP", "MS")),
    "barley2021": dict(crop="Cevada", year="2021", season=("2021-04-01", "2021-11-15"),
                       sow=("2021-04-01", "2021-06-30"), peak=("2021-07-15", "2021-10-15")),
}
SOW_MAX, PEAK_MIN, MIN_PX = 0.35, 0.50, 100
KEEP = ["ID_PROPOSTA", "SG_UF_PROPRIEDADE", "NM_MUNICIPIO_PROPRIEDADE", "CD_GEOCMU", "NR_GRAU_LAT", "NR_MIN_LAT",
        "NR_SEG_LAT", "NR_GRAU_LONG", "NR_MIN_LONG", "NR_SEG_LONG", "LATITUDE", "LONGITUDE",
        "NM_CLASSIF_PRODUTO", "NM_CULTURA_GLOBAL", "NR_AREA_TOTAL", "NR_PRODUTIVIDADE_ESTIMADA",
        "NR_PRODUTIVIDADE_SEGURADA", "NivelDeCobertura", "VL_LIMITE_GARANTIA", "DT_PROPOSTA",
        "DT_INICIO_VIGENCIA", "DT_FIM_VIGENCIA", "ANO_APOLICE", "VALOR_INDENIZAÇÃO", "EVENTO_PREPONDERANTE"]
NUM = ["NR_GRAU_LAT", "NR_MIN_LAT", "NR_SEG_LAT", "NR_GRAU_LONG", "NR_MIN_LONG", "NR_SEG_LONG", "NR_AREA_TOTAL",
       "NR_PRODUTIVIDADE_ESTIMADA", "NR_PRODUTIVIDADE_SEGURADA", "NivelDeCobertura", "VL_LIMITE_GARANTIA",
       "VALOR_INDENIZAÇÃO"]

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")


def br_config(case, cfg=None):
    c = CASES[case]
    cfg = copy.deepcopy(cfg or load_config())
    cfg["paths"]["cache"] = f"cache/brazil_psr/{case}"
    cfg["study"].update(county_fips=case, year=int(c["year"]), season_start=c["season"][0],
                        season_end=c["season"][1], timezone="America/Sao_Paulo")
    cfg["data"]["work_crs"] = CRS
    return cfg


# ----------------------------------------------------------------------------------- policies --
def policies(raw=RAW, fp=None):
    """All policies with a point, numbers parsed (decimal comma), no personal columns (only KEEP is read)."""
    fp = fp or ROOT / "cache" / "brazil_psr" / "policies.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    with open(raw, "rb") as f:                       # partial download is UTF-8, the full 2016-2024 file Latin-1
        try:
            f.readline().decode("utf-8")
            enc = "utf-8"
        except UnicodeDecodeError:
            enc = "latin-1"
    d = pd.read_csv(raw, sep=";", usecols=KEEP, dtype=str, on_bad_lines="skip", encoding=enc)
    for c in NUM:
        d[c] = pd.to_numeric(d[c].str.strip().str.replace(",", ".", regex=False).replace("-", np.nan),
                             errors="coerce")
    d["lat"] = -(d.NR_GRAU_LAT + d.NR_MIN_LAT / 60 + d.NR_SEG_LAT / 3600)
    d["lon"] = -(d.NR_GRAU_LONG + d.NR_MIN_LONG / 60 + d.NR_SEG_LONG / 3600)
    d["event"] = d.EVENTO_PREPONDERANTE.str.strip().replace({"-": np.nan, "": np.nan})
    d["indemnity"] = d["VALOR_INDENIZAÇÃO"].fillna(0.0)
    d["loss_ratio"] = d.indemnity / d.VL_LIMITE_GARANTIA
    d["crop"] = d.NM_CULTURA_GLOBAL.str.strip()
    d["claim_class"] = np.where(d.event == "GEADA", np.where(d.indemnity > 0, "frost_paid", "frost_unpaid"),
                                np.where(d.event.notna(), "other", "none"))
    d = d.drop(columns=["VALOR_INDENIZAÇÃO", "EVENTO_PREPONDERANTE", "NM_CULTURA_GLOBAL"])
    d = d.drop_duplicates("ID_PROPOSTA")
    fp.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(fp)
    return d


def case_policies(case):
    c = CASES[case]
    p = policies()
    p = p[(p.crop == c["crop"]) & (p.ANO_APOLICE == c["year"]) & p.lat.between(-34, -5)
          & p.lon.between(*LON_RANGE)].copy()
    if "states" in c:
        p = p[p.SG_UF_PROPRIEDADE.isin(c["states"])]
    rng = np.random.default_rng(0)
    for cls, n in c.get("cap", {}).items():
        idx = p.index[p.claim_class == cls]
        if len(idx) > n:
            p = p.drop(rng.choice(idx, len(idx) - n, replace=False))
    return p.reset_index(drop=True)


def fields(cfg, case):
    """One window per distinct point; policies -> window via window_id."""
    fp = cache_dir(cfg) / f"fields_used_{case}.gpkg"
    pp = cache_dir(cfg) / "policies.parquet"
    if fp.exists() and pp.exists():
        return gpd.read_file(fp), pd.read_parquet(pp)
    p = case_policies(case)
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(p.lon, p.lat)], crs=4326).to_crs(CRS)
    p["x"], p["y"] = pts.x.round(0), pts.y.round(0)
    key = p.x.astype(int).astype(str) + "_" + p.y.astype(int).astype(str)
    p["field_id"] = ["BR%05d" % k for k in pd.factorize(key)[0]]
    w = p.drop_duplicates("field_id")[["field_id", "x", "y"]].copy()
    h = WINDOW_M / 2
    w["x0"] = np.floor((w.x - h) / 10) * 10
    w["y0"] = np.ceil((w.y + h) / 10) * 10
    w["nx"] = w["ny"] = int(WINDOW_M / 10)
    w["GEOID"] = case
    g = gpd.GeoDataFrame(w, geometry=[box(r.x0, r.y0 - r.ny * 10, r.x0 + r.nx * 10, r.y0) for r in w.itertuples()],
                         crs=CRS)
    g.to_file(fp, driver="GPKG")
    p.to_parquet(pp)
    return g, p


# ----------------------------------------------------------------------------------- sentinel-2 --
def search_items(f, season, tries=5):
    bb = gpd.GeoSeries([box(*f.total_bounds)], crs=CRS).to_crs(4326).iloc[0]
    for k in range(tries):                                       # timeout + retry: a hung/broken search stalled run_b4 twice
        try:
            cat = pystac_client.Client.open(STAC, timeout=(30, 120))
            items = list(cat.search(collections=["sentinel-2-l2a"], intersects=bb.__geo_interface__,
                                    datetime=f"{season[0]}/{season[1]}").items())
            break
        except pystac_client.exceptions.APIError as e:
            if k == tries - 1:
                raise
            print(f"  STAC search failed ({e}); retry {k + 1}", flush=True)
            time.sleep(30 * (k + 1))
    best = {}
    for it in items:
        p = it.properties
        if (p.get("proj:code") or f"EPSG:{p.get('proj:epsg')}") != CRS:
            continue
        k = (p["s2:mgrs_tile"], it.datetime)
        if k not in best or p.get("s2:processing_baseline", "") > best[k].properties.get("s2:processing_baseline", ""):
            best[k] = it
    return sorted(best.values(), key=lambda i: i.datetime)


def _item(item, f, clear_scl):
    """[(window index, clear (H, W), refl (10, H, W) uint16 or None, aot)] for one item."""
    item = pc.sign(item)
    foot = gpd.GeoSeries([shape(item.geometry)], crs=4326).to_crs(CRS).iloc[0]
    out = []
    for i, r in enumerate(f.itertuples()):
        bounds = (r.x0, r.y0 - r.ny * 10, r.x0 + r.nx * 10, r.y0)
        if not foot.contains(box(*bounds)):
            continue
        H, W = int(r.ny), int(r.nx)
        scl = _read(item.assets["SCL"].href, bounds, (H, W))
        if not (scl > 0).any():
            continue
        clear = np.isin(scl, clear_scl)
        if clear.mean() < 0.05:
            out.append((i, clear, None, np.nan))
            continue
        off = _offset(item)
        st = np.stack([np.clip(_read(item.assets[b].href, bounds, (H, W)).astype("float32") + off, 0, 65535)
                       for b in BANDS]).astype("uint16")
        aot = _read(item.assets["AOT"].href, bounds, (max(H // AOT_STEP, 1), max(W // AOT_STEP, 1))).astype("float32")
        aot[aot == 0] = np.nan
        a = float(np.nanmedian(aot)) / 1000.0 if np.isfinite(aot).any() else np.nan
        out.append((i, clear, st, a))
    return item, out


def _item_retry(item, *a, tries=4):
    for k in range(tries):
        try:
            return _item(item, *a)
        except Exception as e:
            if k == tries - 1:
                print(f"  FAILED {item.id}: {e!r}", flush=True)
                return item, []
            print(f"  retry {item.id}: {e!r}", flush=True)
            time.sleep(5 * (k + 1))


def sentinel2_raw(cfg, case, workers=16, log=print):
    """cache/.../raw/<fid>.npz: dates, clear (T, H, W), refl (T, 10, H, W) uint16, aot (T)."""
    out = cache_dir(cfg, "raw")
    done_fp = cache_dir(cfg) / "raw_done.json"
    if done_fp.exists():
        return
    f, _ = fields(cfg, case)
    items = search_items(f, CASES[case]["season"])
    log(f"S2 {case}: {len(f)} windows, {len(items)} items")
    tmp = cache_dir(cfg, "raw_tmp")                       # one file per (window, date), best clear count kept
    best = {}
    for p in tmp.glob("*.npz"):                           # resume
        fid, d, n = p.stem.split("__")
        best[(fid, d)] = (int(n), p)
    clear_scl = cfg["data"]["s2_clear_scl"]
    ids = list(f.field_id)
    with ThreadPoolExecutor(workers) as ex:
        pending = {ex.submit(_item_retry, it, f, clear_scl) for it in items}
        n = 0
        while pending:
            done = {fu for fu in pending if fu.done()}
            if not done:
                time.sleep(0.5)
                continue
            pending -= done
            for fu in done:
                item, res = fu.result()
                d = item.datetime.date().isoformat()
                for i, c, st, a in res:
                    k = (ids[i], d)
                    nc = int(c.sum()) if st is not None else -1
                    if k in best and best[k][0] >= nc:
                        continue
                    fp = tmp / f"{ids[i]}__{d}__{max(nc, 0)}.npz"
                    np.savez_compressed(fp, clear=c, refl=st if st is not None else np.zeros(0, "uint16"),
                                        aot=np.float32(a))
                    if k in best and best[k][1] != fp:
                        best[k][1].unlink(missing_ok=True)
                    best[k] = (nc, fp)
                n += 1
                if n % 25 == 0:
                    log(f"  S2 items {n}/{len(items)}", flush=True)
    by = {}
    for (fid, d), (_, p) in best.items():
        by.setdefault(fid, []).append((d, p))
    for r in f.itertuples():
        lst = sorted(by.get(r.field_id, []))
        H, W = int(r.ny), int(r.nx)
        clear = np.zeros((len(lst), H, W), bool)
        refl = np.zeros((len(lst), len(BANDS), H, W), "uint16")
        aot = np.full(len(lst), np.nan, "float32")
        for t, (d, p) in enumerate(lst):
            z = np.load(p)
            if z["refl"].size:
                clear[t], refl[t], aot[t] = z["clear"], z["refl"], z["aot"]
        np.savez_compressed(out / f"{r.field_id}.npz", dates=np.array([d for d, _ in lst]), clear=clear,
                            refl=refl, aot=aot)
    done_fp.write_text(json.dumps(dict(items=len(items), when=dt.datetime.now().isoformat())))


def _evi2(x):
    return 2.5 * (x[:, 6] - x[:, 2]) / (x[:, 6] + 2.4 * x[:, 2] + 1.0)


def crop_mask(dates, ev, c):
    """Pixels low in the sowing window and green in the peak window."""
    d = np.array([dt.date.fromisoformat(s) for s in dates])
    sow = (d >= dt.date.fromisoformat(c["sow"][0])) & (d <= dt.date.fromisoformat(c["sow"][1]))
    pk = (d >= dt.date.fromisoformat(c["peak"][0])) & (d <= dt.date.fromisoformat(c["peak"][1]))
    with np.errstate(all="ignore"):
        lo = np.nanmin(ev[sow], 0) if sow.any() else np.full(ev.shape[1:], np.nan)
        hi = np.nanmax(ev[pk], 0) if pk.any() else np.full(ev.shape[1:], np.nan)
    return (lo <= SOW_MAX) & (hi >= PEAK_MIN)


def build_fields(cfg, case, log=print):
    """raw -> crop mask -> s2/<fid>.npz, s2bands/<fid>.npz, s2 rows (QA input), mask table."""
    rows_fp = cache_dir(cfg) / f"s2_rows_{case}.parquet"
    if rows_fp.exists():
        return pd.read_parquet(rows_fp)
    f, _ = fields(cfg, case)
    c = CASES[case]
    s2_dir, b_dir, raw = cache_dir(cfg, "s2"), cache_dir(cfg, "s2bands"), cache_dir(cfg, "raw")
    rows, mrows = [], []
    for r in f.itertuples():
        z = np.load(raw / f"{r.field_id}.npz")
        dates, clear, refl, aot = z["dates"], z["clear"], z["refl"], z["aot"]
        x = refl.astype("float32") / 10000.0
        ev = np.where(clear, _evi2(x), np.nan).astype("float32")
        m = crop_mask(dates, ev, c) if len(dates) else np.zeros((int(r.ny), int(r.nx)), bool)
        mrows.append((r.field_id, int(m.sum())))
        if m.sum() < MIN_PX:
            continue
        T = len(dates)
        rgb = np.zeros((T, 3) + m.shape, "uint8")
        out_refl = np.zeros_like(refl)
        cov = np.broadcast_to(m, (T,) + m.shape).copy()
        for t in range(T):
            cc = clear[t] & m
            cs = float(cc.sum() / m.sum())
            blue = red = e = np.nan
            if cc.any():
                out_refl[t][:, cc] = refl[t][:, cc]
                rgb[t] = np.clip(x[t][[2, 1, 0]] / 0.3 * 255, 0, 255).astype("uint8")
                blue, red, e = (float(np.median(x[t, 0][cc])), float(np.median(x[t, 2][cc])),
                                float(np.median(ev[t][cc])))
            ev[t][~cc] = np.nan
            rows.append((r.field_id, str(dates[t]), cs, blue, red, e, float(aot[t])))
        np.savez_compressed(s2_dir / f"{r.field_id}.npz", dates=dates, evi2=ev, rgb=rgb, covered=cov, poly=m)
        np.savez_compressed(b_dir / f"{r.field_id}.npz", dates=dates, bands=np.array(BANDS), refl=out_refl)
    pd.DataFrame(mrows, columns=["field_id", "crop_px"]).to_parquet(cache_dir(cfg) / "crop_mask.parquet")
    df = pd.DataFrame(rows, columns=["field_id", "date", "clear_share", "blue", "red", "evi2", "aot"])
    df.to_parquet(rows_fp)
    log(f"crop mask: {sum(n >= MIN_PX for _, n in mrows)}/{len(mrows)} windows with >= {MIN_PX} crop px")
    return df


def qa(cfg, case, log=print):
    fp = cache_dir(cfg) / f"s2_qa_{case}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    q = cfg["s2_qa"]
    df = build_fields(cfg, case, log=log).copy()
    veg = df.evi2 >= q["hot_min_evi2"]
    a, b = clear_line(df, q["min_field_clear_share"], q["hot_min_evi2"])
    df["hot"] = (df.blue - a - b * df.red) * np.cos(np.arctan(b))
    ref = df[(df.clear_share >= q["min_field_clear_share"]) & veg].groupby("field_id").hot.median().rename("hot_ref")
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
def _openmeteo(lats, lons, start, end):
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=" + ",".join(f"{x:.2f}" for x in lats)
           + "&longitude=" + ",".join(f"{x:.2f}" for x in lons)
           + f"&start_date={start}&end_date={end}&hourly=temperature_2m&models=era5_land"
           + "&timezone=America%2FSao_Paulo")
    for k in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                j = json.loads(r.read())
            return j if isinstance(j, list) else [j]
        except Exception as e:
            print(f"  open-meteo retry {k}: {e!r}", flush=True)
            time.sleep(20 * (k + 1))
    raise RuntimeError("open-meteo failed")


def nights(cfg, case):
    """(cell_id, night) table from ERA5-Land hourly T2m; field -> nearest 0.1 deg cell."""
    fp = cache_dir(cfg) / "nights.parquet"
    cf = cache_dir(cfg) / "field_cell.parquet"
    if fp.exists() and cf.exists():
        return pd.read_parquet(fp), pd.read_parquet(cf)
    f, _ = fields(cfg, case)
    pts = gpd.GeoSeries([Point(r.x, r.y) for r in f.itertuples()], crs=CRS).to_crs(4326)
    clat, clon = np.round(pts.y.to_numpy(), 1), np.round(pts.x.to_numpy(), 1)
    cells = pd.DataFrame({"lat": clat, "lon": clon}).drop_duplicates().reset_index(drop=True)
    cells["cell_id"] = np.arange(len(cells))
    fc = pd.DataFrame({"field_id": f.field_id, "lat": clat, "lon": clon}).merge(cells, on=["lat", "lon"])
    s0, s1 = CASES[case]["season"]
    end = (dt.date.fromisoformat(s1) + dt.timedelta(days=1)).isoformat()
    rows = []
    for k in range(0, len(cells), 50):
        ch = cells.iloc[k:k + 50]
        js = _openmeteo(ch.lat, ch.lon, s0, end)
        for (cid, la, lo), j in zip(ch[["cell_id", "lat", "lon"]].itertuples(index=False), js):
            h = pd.Series(j["hourly"]["temperature_2m"], index=pd.to_datetime(j["hourly"]["time"]), dtype="float64")
            nd = (h.index - pd.Timedelta(hours=9)).normalize()     # 18:00 D .. 09:00 D+1 -> night D
            hh = h[(h.index.hour >= 18) | (h.index.hour <= 9)]
            mn = hh.groupby(nd[(h.index.hour >= 18) | (h.index.hour <= 9)]).min()
            for d, t in mn.items():
                if s0 <= d.date().isoformat() <= s1:
                    rows.append((int(cid), d.date().isoformat(), float(t), float(la), float(lo)))
        time.sleep(2)
    n = pd.DataFrame(rows, columns=["cell_id", "night", "tmin_c", "lat", "lon"])
    n["trigger"] = n.tmin_c <= TRIGGER_C
    n["near_trigger"] = False
    n["regime"] = "era5"
    n["dose_degree_hours"] = 0.0
    n.to_parquet(fp)
    fc[["field_id", "cell_id"]].to_parquet(cf)
    return n, fc[["field_id", "cell_id"]]


# -------------------------------------------------------------------------------------- store --
class BrazilStore(FieldStore):
    """FieldStore over a Brazil case cache: S2 crop mask, ERA5-Land nights, no stage clock."""

    def __init__(self, cfg, case):
        self.cfg, self.case, self.fips = cfg, case, case
        c = cache_dir(cfg)
        f = gpd.read_file(c / f"fields_used_{case}.gpkg")
        ok = {p.stem for p in cache_dir(cfg, "s2").glob("*.npz")}
        f = f[f.field_id.isin(ok)].reset_index(drop=True)
        self._mask = {fid: np.load(cache_dir(cfg, "s2") / f"{fid}.npz")["poly"] for fid in f.field_id}
        self.fields = f
        self.nights, fc = nights(cfg, case)
        self.field_cell = dict(zip(fc.field_id, fc.cell_id))
        self._n = self.nights.set_index(["cell_id", "night"])
        q = pd.read_parquet(c / f"s2_qa_{case}.parquet")
        self._drop = q[~q.usable].groupby("field_id").apply(lambda g: dict(zip(g.date, g.drop_reason))).to_dict()

    def daily_temp(self, field_id):
        s0, s1 = CASES[self.case]["season"]
        d = pd.date_range(s0, s1, freq="D").date
        return pd.DataFrame({"tmax_c": 25.0, "tmin_c": 12.0}, index=[x.isoformat() for x in d])

    def night_meta(self, field_id, night):
        cell = self.field_cell[field_id]
        r = self._n.loc[(cell, dt.date.fromisoformat(str(night)).isoformat())]
        return dict(cell_id=int(cell), dose=0.0, tmin_c=float(r.tmin_c), trigger=bool(r.trigger),
                    near_trigger=False, regime="era5")


def build(case, log=print):
    cfg = br_config(case)
    f, p = fields(cfg, case)
    log(f"{case}: {len(p)} policies, {len(f)} windows; classes {p.claim_class.value_counts().to_dict()}")
    n, fc = nights(cfg, case)
    log(f"nights: {fc.cell_id.nunique()} ERA5-Land cells, trigger cell-nights {int(n.trigger.sum())}")
    sentinel2_raw(cfg, case, log=log)
    qa(cfg, case, log=log)
    return cfg


if __name__ == "__main__":
    build(sys.argv[1], log=lambda *a, **k: print(*a, flush=True))
