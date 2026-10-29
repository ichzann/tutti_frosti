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


def field_elevation(cfg, row) -> np.ndarray:
    """DEM on the field's 10 m grid (float32, NaN outside DEM). Field and DEM grids share 10 m snapping."""
    fp = cache_dir(cfg, "masks") / f"{row.field_id}_dem.npy"
    if fp.exists():
        return np.load(fp)
    with rasterio.open(county_dem(cfg, row.GEOID)) as src:
        w = rasterio.windows.from_bounds(row.x0, row.y0 - row.ny * 10, row.x0 + row.nx * 10, row.y0,
                                         transform=src.transform).round_offsets().round_lengths()
        a = src.read(1, window=w, boundless=True, fill_value=np.nan).astype("float32")
    np.save(fp, a)
    return a


def hillshade(cfg, fips: str | None = None, res=60, azimuth=315.0, altitude=45.0):
    """(uint8 hillshade in EPSG:3857, (west, south, east, north) lon/lat bounds), downsampled."""
    from rasterio.warp import calculate_default_transform, reproject
    with rasterio.open(county_dem(cfg, fips)) as src:
        f = int(round(res / src.res[0]))
        z = src.read(1, out_shape=(src.height // f, src.width // f), resampling=Resampling.average)
        tr = src.transform * src.transform.scale(src.width / z.shape[1], src.height / z.shape[0])
        crs = src.crs
    gy, gx = np.gradient(np.nan_to_num(z, nan=np.nanmedian(z)), res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = np.clip(hs, 0, 1).astype("float32")
    h, w = hs.shape
    dtr, dw, dh = calculate_default_transform(crs, "EPSG:3857", w, h,
                                              *rasterio.transform.array_bounds(h, w, tr))
    out = np.zeros((dh, dw), "float32")
    reproject(hs, out, src_transform=tr, src_crs=crs, dst_transform=dtr, dst_crs="EPSG:3857",
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=0)
    from rasterio.warp import transform_bounds
    b = transform_bounds("EPSG:3857", "EPSG:4326", *rasterio.transform.array_bounds(dh, dw, dtr))
    return (np.nan_to_num(out) * 255).astype("uint8"), b
