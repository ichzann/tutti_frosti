"""USGS 3DEP 10 m DEM (Planetary Computer `3dep-seamless`), clipped per county."""
from __future__ import annotations

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from .common import cache_dir, county

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def county_dem(cfg, fips: str | None = None):
    """Path to a GeoTIFF of the county DEM in the work CRS at 10 m (cached)."""
    fips = fips or cfg["study"]["county_fips"]
    fp = cache_dir(cfg, "dem") / f"dem_{fips}.tif"
    if fp.exists():
        return fp
    crs = cfg["data"]["work_crs"]
    res = cfg["data"]["dem_gsd_m"]
    c = county(cfg, fips)
    minx, miny, maxx, maxy = c.to_crs(crs).buffer(500).total_bounds
    minx, miny = np.floor([minx / res, miny / res]) * res
    maxx, maxy = np.ceil([maxx / res, maxy / res]) * res
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    items = cat.search(collections=[cfg["data"]["dem_collection"]],
                       bbox=list(c.to_crs(4326).total_bounds + np.array([-0.01, -0.01, 0.01, 0.01])),
                       query={"gsd": {"eq": res}}).items()
    srcs = []
    for it in items:
        src = rasterio.open(it.assets["data"].href)
        srcs.append(WarpedVRT(src, crs=crs, resampling=Resampling.bilinear))
    arr, transform = merge(srcs, bounds=(minx, miny, maxx, maxy), res=res,
                           nodata=np.nan, dtype="float32")
    with rasterio.open(fp, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=1, dtype="float32", crs=crs, transform=transform,
                       nodata=np.nan, compress="deflate", tiled=True, predictor=3) as dst:
        dst.write(arr[0], 1)
    return fp
