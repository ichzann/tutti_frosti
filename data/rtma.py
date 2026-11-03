"""RTMA 2.5 km hourly analyses -> nights table (stream-and-discard).

Only the GRIB2 messages that are needed (TMP, WIND, GUST, TCDC every night hour;
DPT at the hour nearest sunset) are fetched with HTTP byte ranges from the NOAA
AWS bucket, decoded in memory, cut to a small window around the candidate
counties and discarded. The per-hour window subsets (a few tens of KB each) are
cached so an interrupted run can resume; the raw GRIB is never written to disk.
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import eccodes
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from astral import Observer
from astral.sun import sunrise, sunset
from pyproj import CRS, Transformer
from shapely.geometry import box

from .common import all_candidate_counties, cache_dir

BUCKET = "https://noaa-rtma-pds.s3.amazonaws.com"
HOURLY_VARS = ["TMP", "WIND", "GUST", "TCDC"]
SUNSET_VARS = ["DPT"]
# NDFD 2.5 km CONUS grid (checked against the GRIB headers on first fetch)
NX, NY, DX = 2345, 1597, 2539.703
LAT0, LON0 = 19.228976, 233.723448 - 360.0
RTMA_CRS = CRS.from_proj4(
    "+proj=lcc +lat_0=25 +lon_0=265 +lat_1=25 +lat_2=25 +R=6371200 +units=m +no_defs"
)
APPROX_MSG_BYTES = {"TMP": 6.09e6, "DPT": 6.09e6, "WIND": 5.62e6, "GUST": 5.62e6, "TCDC": 6.55e6}


# ---------------------------------------------------------------- grid -----
def _origin_xy():
    t = Transformer.from_crs("EPSG:4326", RTMA_CRS, always_xy=True)
    return t.transform(LON0, LAT0)


def window(cfg, margin_cells: int = 2):
    """(j0, j1, i0, i1) index window covering all candidate counties."""
    x0, y0 = _origin_xy()
    cnt = all_candidate_counties(cfg).to_crs(RTMA_CRS)
    minx, miny, maxx, maxy = cnt.total_bounds
    i0 = int(np.floor((minx - x0) / DX + 0.5)) - margin_cells
    i1 = int(np.ceil((maxx - x0) / DX + 0.5)) + margin_cells
    j0 = int(np.floor((miny - y0) / DX + 0.5)) - margin_cells
    j1 = int(np.ceil((maxy - y0) / DX + 0.5)) + margin_cells
    return j0, j1, i0, i1


def cells(cfg) -> gpd.GeoDataFrame:
    """RTMA cell polygons in the window, with the candidate county each touches."""
    x0, y0 = _origin_xy()
    j0, j1, i0, i1 = window(cfg)
    rows = []
    for j in range(j0, j1):
        for i in range(i0, i1):
            cx, cy = x0 + i * DX, y0 + j * DX
            rows.append((f"{j}_{i}", j, i, cx, cy,
                         box(cx - DX / 2, cy - DX / 2, cx + DX / 2, cy + DX / 2)))
    g = gpd.GeoDataFrame(rows, columns=["cell_id", "j", "i", "x", "y", "geometry"], crs=RTMA_CRS)
    cnt = all_candidate_counties(cfg).to_crs(RTMA_CRS)
    for _, c in cnt.iterrows():
        g[f"in_{c.GEOID}"] = g.intersects(c.geometry)
    g = g[g[[f"in_{f}" for f in cnt.GEOID]].any(axis=1)].reset_index(drop=True)
    ll = gpd.GeoSeries(gpd.points_from_xy(g.x, g.y), crs=RTMA_CRS).to_crs("EPSG:4326")
    g["lon"], g["lat"] = ll.x, ll.y
    return g


# ------------------------------------------------------------- timing -----
TZ = ZoneInfo("America/Denver")


def night_bounds(date: dt.date, lat: float, lon: float):
    """(local sunset on `date`, local sunrise on date+1), returned in UTC."""
    obs = Observer(latitude=lat, longitude=lon)
    ss = sunset(obs, date=date, tzinfo=TZ)
    sr = sunrise(obs, date=date + dt.timedelta(days=1), tzinfo=TZ)
    return ss.astimezone(dt.timezone.utc), sr.astimezone(dt.timezone.utc)


def season_dates(cfg):
    s = dt.date.fromisoformat(cfg["study"]["season_start"])
    e = dt.date.fromisoformat(cfg["study"]["season_end"])
    return [s + dt.timedelta(days=k) for k in range((e - s).days + 1)]


def needed_hours(cfg, cell_df):
    """{utc_hour: set(vars)} for the whole season."""
    need: dict[dt.datetime, set] = {}
    for d in season_dates(cfg):
        bounds = [night_bounds(d, la, lo) for la, lo in zip(cell_df.lat, cell_df.lon)]
        ss_min = min(b[0] for b in bounds)
        sr_max = max(b[1] for b in bounds)
        t = ss_min.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
        while t <= sr_max:
            need.setdefault(t, set()).update(HOURLY_VARS)
            t += dt.timedelta(hours=1)
        for ss, _ in bounds:
            h = _nearest_hour(ss)
            need.setdefault(h, set()).update(SUNSET_VARS)
    return need


def _nearest_hour(t: dt.datetime) -> dt.datetime:
    base = t.replace(minute=0, second=0, microsecond=0)
    return base + dt.timedelta(hours=1) if t.minute >= 30 else base


def estimate_volume(need) -> float:
    return sum(APPROX_MSG_BYTES[v] for vs in need.values() for v in vs)


# -------------------------------------------------------------- fetch -----
def _urls(t: dt.datetime):
    d = t.strftime("%Y%m%d")
    base = f"{BUCKET}/rtma2p5.{d}/rtma2p5.t{t:%H}z.2dvaranl_ndfd.grb2"
    return [base + "_wexp", base]


def _fetch_idx(session, t):
    for u in _urls(t):
        r = session.get(u + ".idx", timeout=60)
        if r.status_code == 200:
            return u, r.text
    return None, None


def fetch_hour(session, t: dt.datetime, variables, win):
    """Return {var: 2D float32 array over window} or None if the file is missing."""
    url, idx = _fetch_idx(session, t)
    if url is None:
        return None
    lines = [ln.split(":") for ln in idx.strip().splitlines()]
    offs = [int(ln[1]) for ln in lines]
    out = {}
    j0, j1, i0, i1 = win
    for k, ln in enumerate(lines):
        var = ln[3]
        if var not in variables or var in out:
            continue
        end = offs[k + 1] - 1 if k + 1 < len(offs) else ""
        r = session.get(url, headers={"Range": f"bytes={offs[k]}-{end}"}, timeout=120)
        r.raise_for_status()
        h = eccodes.codes_new_from_message(r.content)
        try:
            nx, ny = eccodes.codes_get(h, "Nx"), eccodes.codes_get(h, "Ny")
            assert (nx, ny) == (NX, NY) and eccodes.codes_get(h, "jScansPositively") == 1
            vals = eccodes.codes_get_values(h).reshape(ny, nx)
        finally:
            eccodes.codes_release(h)
        out[var] = vals[j0:j1, i0:i1].astype(np.float32)
    missing = set(variables) - set(out)
    if missing:
        raise RuntimeError(f"{t}: messages {missing} not in {url}")
    return out


def fetch_all(cfg, need, win, workers: int = 8, log=print):
    """Fetch (or load cached) window subsets for every needed hour."""
    cdir = cache_dir(cfg, "rtma_hours")
    todo, data = [], {}
    for t, vs in sorted(need.items()):
        fp = cdir / f"{t:%Y%m%d%H}.npz"
        if fp.exists():
            z = np.load(fp)
            if set(vs) <= set(z.files) or "_missing" in z.files:
                data[t] = None if "_missing" in z.files else {k: z[k] for k in z.files}
                continue
        todo.append((t, vs, fp))
    log(f"RTMA: {len(data)} hours cached, {len(todo)} to stream "
        f"(~{sum(APPROX_MSG_BYTES[v] for _, vs, _ in todo for v in vs) / 1e9:.1f} GB)")
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=workers)
    session.mount("https://", adapter)

    def job(t, vs, fp):
        for attempt in range(4):
            try:
                res = fetch_hour(session, t, vs, win)
                break
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise
        if res is None:
            np.savez(fp, _missing=np.zeros(1))
        else:
            np.savez(fp, **res)
        return t, res

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(job, *a) for a in todo]
        for n, f in enumerate(as_completed(futs), 1):
            t, res = f.result()
            data[t] = res
            if n % 200 == 0:
                log(f"  RTMA streamed {n}/{len(todo)} hours")
    return data


# ------------------------------------------------------------- nights -----
def neighbourhood_mean(a, n):
    """Mean over the n x n cells centred on each cell (NaN-aware at the window edge)."""
    if n <= 1:
        return a
    h = n // 2
    p = np.pad(a.astype("float64"), h, constant_values=np.nan)
    st = np.stack([p[dj:dj + a.shape[0], di:di + a.shape[1]] for dj in range(n) for di in range(n)])
    return np.nanmean(st, 0)


def thresholds(nc):
    """Trigger thresholds on RTMA hourly 2 m Tmin per night regime (see config.yaml: night)."""
    base = nc["leaf_critical_c"] + nc["rtma_margin_c"] + nc["hourly_miss_c"]
    return {"radiative": base + nc["leaf_offset_radiative_c"], "advective": base}


def flag(df, nc) -> pd.DataFrame:
    """Add regime / trigger / near_trigger / dose to a nights table (cheap; re-run for sweeps).

    Radiative = calm and clear in the hours up to Tmin; there the leaf runs colder than 2 m air,
    so the trigger is warmer.  Every other night is advective (wind / cloud keep the leaf near
    air temperature) and is still evaluated, against a colder threshold.
    """
    thr = thresholds(nc)
    # clear = the cell AND its n x n neighbourhood (a clear hole in a cloud deck still gets
    # back-radiation from the surrounding cloud; see config.yaml: cloud_neighbourhood_cells)
    cloud_nb = df.get("cloud_pre_tmin_nb_pct", df.cloud_pre_tmin_pct)
    rad = ((df.wind_pre_tmin_ms <= nc["wind_max_ms"]) & (df.cloud_pre_tmin_pct <= nc["cloud_max_pct"])
           & (cloud_nb <= nc["cloud_max_pct"]))
    out = df.copy()
    out["regime"] = np.where(df.tmin_c.isna(), None, np.where(rad, "radiative", "advective"))
    out["trigger_thr_c"] = np.where(rad, thr["radiative"], thr["advective"])
    out["trigger"] = (df.tmin_c <= out.trigger_thr_c).to_numpy()
    lo_t, hi_t = nc["near_trigger_tmin_c"]
    out["near_trigger"] = (rad & df.tmin_c.between(lo_t, hi_t)).to_numpy()
    out["dose_degree_hours"] = np.where(rad, df.dh_radiative, df.dh_advective)
    return out


def build_nights(cfg, cell_df, data, win) -> pd.DataFrame:
    """One row per RTMA cell x night with summary stats and flags."""
    nc = cfg["night"]
    pre = nc["pre_tmin_hours"]
    j0, _, i0, _ = win
    jj = (cell_df.j - j0).to_numpy()
    ii = (cell_df.i - i0).to_numpy()
    tz = ZoneInfo(cfg["study"]["timezone"])
    nb = nc.get("cloud_neighbourhood_cells", 1)
    tcdc_nb = {h: neighbourhood_mean(g["TCDC"], nb) for h, g in data.items() if "TCDC" in g}
    rows = []
    for d in season_dates(cfg):
        for c, (cid, la, lo, j, i) in enumerate(zip(cell_df.cell_id, cell_df.lat, cell_df.lon, jj, ii)):
            ss, sr = night_bounds(d, la, lo)
            hours = []
            t = ss.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
            while t <= sr:
                hours.append(t)
                t += dt.timedelta(hours=1)
            got = [data.get(h) for h in hours]
            ok_hours = [h for h, g in zip(hours, got) if g is not None]
            ok = [g for g in got if g is not None]
            tmp = np.array([g["TMP"][j, i] for g in ok]) - 273.15
            wind = np.array([g["WIND"][j, i] for g in ok])
            gust = np.array([g["GUST"][j, i] for g in ok])
            tcdc = np.array([g["TCDC"][j, i] for g in ok])
            tcdc_n = np.array([tcdc_nb[h][j, i] for h in ok_hours])
            hs = data.get(_nearest_hour(ss))
            dpt_ss = float(hs["DPT"][j, i] - 273.15) if hs is not None and "DPT" in hs else np.nan
            if len(ok):
                k = int(tmp.argmin())
                w = slice(max(0, k - pre + 1), k + 1)   # the `pre` hours up to and including Tmin
                # estimated leaf temperature per regime -> degree-hours below the dose base
                dh = lambda off: float(np.clip(nc["dose_base_c"] - (tmp - off), 0, None).sum())
                rec = dict(tmin_c=float(tmp[k]),
                           tmin_local=ok_hours[k].astimezone(tz).isoformat(timespec="minutes"),
                           wind_pre_tmin_ms=float(wind[w].mean()), cloud_pre_tmin_pct=float(tcdc[w].mean()),
                           cloud_pre_tmin_nb_pct=float(tcdc_n[w].mean()),
                           # evening cooling rate (first 3 h after sunset), C/h: clear calm nights
                           # cool fast; reported only, an independent check of the cloud input
                           cooling_evening_c_h=float((tmp[0] - tmp[3]) / 3) if len(tmp) > 3 else np.nan,
                           wind_mean_ms=float(wind.mean()), gust_max_ms=float(gust.max()),
                           cloud_mean_pct=float(tcdc.mean()),
                           dh_radiative=dh(nc["leaf_offset_radiative_c"]), dh_advective=dh(0.0))
            else:
                rec = dict(tmin_c=np.nan, tmin_local=None, wind_pre_tmin_ms=np.nan, cloud_pre_tmin_pct=np.nan,
                           cloud_pre_tmin_nb_pct=np.nan, cooling_evening_c_h=np.nan,
                           wind_mean_ms=np.nan, gust_max_ms=np.nan, cloud_mean_pct=np.nan,
                           dh_radiative=np.nan, dh_advective=np.nan)
            rows.append(dict(cell_id=cid, night=d.isoformat(),
                             sunset_local=ss.astimezone(tz).isoformat(timespec="minutes"),
                             sunrise_local=sr.astimezone(tz).isoformat(timespec="minutes"),
                             n_hours=len(hours), n_hours_missing=len(hours) - len(ok),
                             dewpoint_sunset_c=dpt_ss, **rec))
    return flag(pd.DataFrame(rows), nc)


def run(cfg, log=print, workers: int = 8, dry_run: bool = False):
    cell_df = cells(cfg)
    win = window(cfg)
    need = needed_hours(cfg, cell_df)
    if dry_run:
        return cell_df, need
    data = fetch_all(cfg, need, win, workers=workers, log=log)
    return cell_df, build_nights(cfg, cell_df, data, win)
