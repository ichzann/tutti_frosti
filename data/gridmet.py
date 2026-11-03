"""gridMET daily 2 m Tmax / Tmin (~4 km) for the study county, for thermal-time (P-day) crop stage.

    python -m data.gridmet            # writes cache/gridmet_<fips>.parquet (skips if it exists)

RTMA in cache/nights.parquet holds night hours only, so daytime maxima come from gridMET
(Abatzoglou 2013), subset server-side by the NKN THREDDS NetCDF subset service (county bounding box
+ margin, Apr 1 .. season end). Output: one row per grid cell and day: lat, lon, date, tmax_c, tmin_c.
"""
from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd
import requests
from scipy.io import netcdf_file

from .common import cache_dir, county, load_config

URL = "http://thredds.northwestknowledge.net:8080/thredds/ncss/agg_met_{v}_1979_CurrentYear_CONUS.nc"
VARS = {"tmmx": ("daily_maximum_temperature", "tmax_c"), "tmmn": ("daily_minimum_temperature", "tmin_c")}
MARGIN_DEG = 0.1


def _fetch(var, name, bounds, start, end):
    x0, y0, x1, y1 = bounds
    q = dict(var=name, north=y1 + MARGIN_DEG, south=y0 - MARGIN_DEG, west=x0 - MARGIN_DEG, east=x1 + MARGIN_DEG,
             disableProjSubset="on", horizStride=1, timeStride=1, accept="netcdf",
             time_start=f"{start}T00:00:00Z", time_end=f"{end}T00:00:00Z")
    r = requests.get(URL.format(v=var), params=q, timeout=300)
    r.raise_for_status()
    f = netcdf_file(io.BytesIO(r.content), mmap=False)
    v = f.variables[name]
    raw = v.data.astype("int64")
    if raw.min() < 0:                          # stored as unsigned int16 (_Unsigned = true)
        raw = raw % 65536
    a = np.where(raw == 32767, np.nan, raw * float(v.scale_factor) + float(v.add_offset)) - 273.15
    days = [dt.date(1900, 1, 1) + dt.timedelta(days=int(d)) for d in f.variables["day"].data]
    lat, lon = (f.variables[k].data.astype("float64") for k in ("lat", "lon"))   # big-endian in the file
    T, Y, X = a.shape                          # (day, lat, lon) despite the 'lon lat time' attribute
    assert (T, Y, X) == (len(days), len(lat), len(lon)), a.shape
    return pd.DataFrame(dict(lat=np.tile(np.repeat(lat, X), T), lon=np.tile(np.tile(lon, Y), T),
                             date=np.repeat([d.isoformat() for d in days], Y * X), value=a.ravel()))


def build(cfg, fips=None, log=print):
    fips = fips or cfg["study"]["county_fips"]
    fp = cache_dir(cfg) / f"gridmet_{fips}.parquet"
    if fp.exists():
        return fp
    bounds = county(cfg, fips).to_crs(4326).total_bounds
    year = int(cfg["study"]["year"])
    start, end = f"{year}-04-01", cfg["study"]["season_end"]
    parts = []
    for var, (name, col) in VARS.items():
        log(f"gridMET {var} {start}..{end}")
        parts.append(_fetch(var, name, bounds, start, end).rename(columns={"value": col}).set_index(
            ["lat", "lon", "date"]))
    df = parts[0].join(parts[1]).reset_index().dropna()
    df.to_parquet(fp)
    log(f"wrote {fp} ({df[['lat', 'lon']].drop_duplicates().shape[0]} cells, {df.date.nunique()} days)")
    return fp


if __name__ == "__main__":
    build(load_config())
