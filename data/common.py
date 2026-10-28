"""Shared helpers for the data layer: config, cache paths, county geometry."""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cache_dir(cfg: dict, *parts: str) -> Path:
    p = ROOT / cfg["paths"]["cache"]
    for part in parts:
        p = p / part
    p.mkdir(parents=True, exist_ok=True)
    return p


@lru_cache(maxsize=None)
def _all_counties(cache: str) -> gpd.GeoDataFrame:
    fp = Path(cache) / "cb_2022_us_county_500k.gpkg"
    if not fp.exists():
        r = requests.get(COUNTY_URL, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(Path(cache) / "_county_shp")
        gdf = gpd.read_file(Path(cache) / "_county_shp" / "cb_2022_us_county_500k.shp")
        gdf = gdf[gdf.STATEFP == "16"]  # Idaho only
        gdf.to_file(fp, driver="GPKG")
    return gpd.read_file(fp)


def county(cfg: dict, fips: str | None = None) -> gpd.GeoDataFrame:
    """County polygon (EPSG:4269 as published) for a 5-digit FIPS."""
    fips = fips or cfg["study"]["county_fips"]
    gdf = _all_counties(str(cache_dir(cfg)))
    out = gdf[gdf.GEOID == fips]
    if out.empty:
        raise ValueError(f"county {fips} not found")
    return out.reset_index(drop=True)


def all_candidate_counties(cfg: dict) -> gpd.GeoDataFrame:
    gdf = _all_counties(str(cache_dir(cfg)))
    return gdf[gdf.GEOID.isin(cfg["study"]["candidate_counties"].values())].reset_index(drop=True)
