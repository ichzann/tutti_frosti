"""USDA Cropland Data Layer (class + confidence), clipped per county.

Class: CropScape by-FIPS service (small).  Confidence: only the GeoTIFF entry of
the national confidence zip is fetched (the overview file is skipped), inflated
to a temporary file, clipped to the county and deleted.
"""
from __future__ import annotations

import re
import struct
import subprocess
import zlib

import numpy as np
import rasterio
import requests
from rasterio.windows import from_bounds

from .common import cache_dir, county

CROPSCAPE = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year={y}&fips={f}"
CONF_URL = ("https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/"
            "{y}_30m_Confidence_Layer.zip")


def _range(url, a, b):
    return subprocess.run(["curl", "-s", "-L", "-r", f"{a}-{b}", url],
                          capture_output=True, check=True).stdout


def _zip_entry(url, suffix):
    """(local header offset, compressed size, method) of the entry ending in suffix."""
    hdr = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True).stdout
    n = int(re.findall(r"content-length:\s*(\d+)", hdr, re.I)[-1])
    tail = _range(url, n - 200000, n - 1)
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = tail[len(tail) - (n - cd_off):][:cd_size] if n - cd_off <= len(tail) else \
        _range(url, cd_off, cd_off + cd_size - 1)
    p = 0
    while cd[p:p + 4] == b"PK\x01\x02":
        meth = struct.unpack("<H", cd[p + 10:p + 12])[0]
        cs, us = struct.unpack("<II", cd[p + 20:p + 28])
        fl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + fl].decode()
        ex = cd[p + 46 + fl:p + 46 + fl + el]
        q = 0
        while q + 4 <= len(ex):
            hid, hl = struct.unpack("<HH", ex[q:q + 4])
            if hid == 1:
                vals = list(struct.unpack("<" + "Q" * (hl // 8), ex[q + 4:q + 4 + hl // 8 * 8]))
                if us == 0xFFFFFFFF: us = vals.pop(0)
                if cs == 0xFFFFFFFF: cs = vals.pop(0)
                if off == 0xFFFFFFFF: off = vals.pop(0)
            q += 4 + hl
        if name.endswith(suffix):
            return off, cs, meth
        p += 46 + fl + el + cl
    raise FileNotFoundError(suffix)


def _download_entry(url, suffix, out):
    off, cs, meth = _zip_entry(url, suffix)
    lh = _range(url, off, off + 29)
    fl, el = struct.unpack("<HH", lh[26:30])
    start = off + 30 + fl + el
    d = zlib.decompressobj(-15) if meth == 8 else None
    with requests.get(url, headers={"Range": f"bytes={start}-{start + cs - 1}",
                                    "User-Agent": "curl/8"}, stream=True, timeout=300) as r, \
            open(out, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 22):
            f.write(d.decompress(chunk) if d else chunk)
        if d:
            f.write(d.flush())


def county_cdl(cfg, fips: str | None = None):
    """(class_path, confidence_path) GeoTIFFs on the native CDL 30 m grid."""
    fips = fips or cfg["study"]["county_fips"]
    y = cfg["study"]["year"]
    cdir = cache_dir(cfg, "cdl")
    cls_fp, conf_fp = cdir / f"cdl_{y}_{fips}.tif", cdir / f"cdlconf_{y}_{fips}.tif"
    if not cls_fp.exists():
        xml = requests.get(CROPSCAPE.format(y=y, f=fips), timeout=300).text
        u = re.search(r"<returnURL>(.*?)</returnURL>", xml).group(1)
        cls_fp.write_bytes(requests.get(u, timeout=300).content)
    if not conf_fp.exists():
        nat = cache_dir(cfg, "_tmp") / f"{y}_30m_confidence_layer.tif"
        if not nat.exists():
            _download_entry(CONF_URL.format(y=y), "confidence_layer.tif", nat)
        with rasterio.open(cls_fp) as c:
            bounds, prof = c.bounds, c.profile
        with rasterio.open(nat) as src:
            w = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=w)
            prof.update(dtype=arr.dtype, transform=src.window_transform(w),
                        width=arr.shape[1], height=arr.shape[0], compress="deflate", nodata=None)
        with rasterio.open(conf_fp, "w", **prof) as dst:
            dst.write(arr, 1)
    return cls_fp, conf_fp


def cleanup(cfg):
    for p in cache_dir(cfg, "_tmp").glob("*_confidence_layer.tif"):
        p.unlink()
