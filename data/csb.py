"""USDA NASS Crop Sequence Boundaries (2018-2025 release) -> county field polygons.

The national archive is downloaded once, the Idaho candidate counties are cut
out and stored as a small GeoPackage, and the national files are deleted.
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile

import geopandas as gpd
import pyogrio

from .common import all_candidate_counties, cache_dir

URL = ("https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/"
       "datasets/{release}.zip")


def candidate_fields(cfg) -> gpd.GeoDataFrame:
    """All CSB polygons (every crop) in the candidate counties, cached."""
    fp = cache_dir(cfg) / "csb_candidates.gpkg"
    if fp.exists():
        return gpd.read_file(fp)
    rel = cfg["data"]["csb_release"]
    tmp = cache_dir(cfg, "_tmp")
    zp = tmp / "csb.zip"
    if not zp.exists() or zp.stat().st_size < 3e9:
        subprocess.run(["curl", "-s", "-L", "-C", "-", "-o", str(zp), URL.format(release=rel)],
                       check=True)
    gdb_dir = tmp / "csb_gdb"
    if not gdb_dir.exists():
        with zipfile.ZipFile(zp) as z:
            z.extractall(gdb_dir)
    gdb = next(gdb_dir.rglob("*.gdb"))
    layer = pyogrio.list_layers(gdb)[0][0]
    fips = list(cfg["study"]["candidate_counties"].values())
    st = {f[:2] for f in fips}
    cn = ",".join(f"'{f[2:]}'" for f in fips)
    where = f"STATEFIPS IN ({','.join(repr(s) for s in st)}) AND CNTYFIPS IN ({cn})"
    gdf = pyogrio.read_dataframe(gdb, layer=layer, where=where)
    gdf["GEOID"] = gdf["STATEFIPS"].astype(str) + gdf["CNTYFIPS"].astype(str)
    gdf.to_file(fp, driver="GPKG")
    shutil.rmtree(gdb_dir)
    zp.unlink()
    return gdf


def potato_fields(cfg, fips: str) -> gpd.GeoDataFrame:
    """CSB polygons whose crop attribute for the study year is potato."""
    gdf = candidate_fields(cfg)
    col = f"CDL{cfg['study']['year']}"
    out = gdf[(gdf.GEOID == fips) & (gdf[col].astype(int) == cfg["data"]["cdl_potato_class"])]
    return out.to_crs(cfg["data"]["work_crs"]).reset_index(drop=True)
