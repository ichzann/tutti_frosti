"""USDA Cropland Data Layer (class + confidence), clipped per county.

Class: CropScape by-FIPS service (small).  Confidence: only the GeoTIFF entry of
the national confidence zip is fetched (the overview file is skipped, resumable),
inflated to a temporary file, clipped to the county and deleted.
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
# years published only as a 10 m national layer (~68 GB zip): streamed, see _stream_clip_10m
CONF_URL_10M = {2024: "https://storage.googleapis.com/temp_cdl_share/2024_10m_confidence_layer.zip"}


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
    """Fetch the entry's compressed bytes with resumable curl (restarts stalled transfers),
    then inflate to `out`; the compressed part file is deleted."""
    off, cs, meth = _zip_entry(url, suffix)
    lh = _range(url, off, off + 29)
    fl, el = struct.unpack("<HH", lh[26:30])
    start = off + 30 + fl + el
    part = out.with_suffix(".part")
    for _ in range(50):
        have = part.stat().st_size if part.exists() else 0
        if have >= cs:
            break
        with open(part, "ab") as f:
            subprocess.run(["curl", "-s", "-L", "-r", f"{start + have}-{start + cs - 1}",
                            "-y", "60", "-Y", "50000", url], stdout=f)
    if part.stat().st_size != cs:
        raise IOError(f"incomplete download of {suffix}: {part.stat().st_size}/{cs} bytes")
    d = zlib.decompressobj(-15) if meth == 8 else None
    with open(part, "rb") as src, open(out, "wb") as f:
        while chunk := src.read(1 << 24):
            f.write(d.decompress(chunk) if d else chunk)
        if d:
            f.write(d.flush())
    part.unlink()


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
    if not conf_fp.exists() and y in CONF_URL_10M:
        with rasterio.open(cls_fp) as c:
            bounds = c.bounds
        _stream_clip_10m(CONF_URL_10M[y], bounds, conf_fp, cache_dir(cfg, "_tmp"))
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


def _inflate_range(url, comp_start, comp_end, d, pos, sink, stop_at=None, checkpoints=None,
                   ckpt_every=1 << 30, log=print):
    """Inflate the deflate stream from compressed byte comp_start (decompressor `d` positioned at
    uncompressed offset `pos`), calling sink(pos, bytes) for every output chunk, until the
    compressed end or uncompressed offset stop_at. Restarts from the last checkpoint on network
    errors. Returns the final uncompressed offset."""
    ck = (comp_start, pos, d.copy())
    last_ck = comp_start
    off = comp_start
    while True:
        p = subprocess.Popen(["curl", "-s", "-L", "-r", f"{off}-{comp_end - 1}",
                              "-y", "60", "-Y", "50000", url], stdout=subprocess.PIPE)
        try:
            while chunk := p.stdout.read(1 << 22):
                buf = chunk
                while buf and not d.eof:  # bounded output: nodata runs inflate ~1000x
                    out = d.decompress(buf, 1 << 26)
                    buf = d.unconsumed_tail
                    if out:
                        sink(pos, out)
                        pos += len(out)
                off += len(chunk)
                if checkpoints is not None and off - last_ck >= ckpt_every:
                    checkpoints.append((off, pos, d.copy()))
                    last_ck = off
                    log(f"  CDL stream {off / 1e9:.1f}/{comp_end / 1e9:.1f} GB")
                if off - ck[0] >= ckpt_every:
                    ck = (off, pos, d.copy())
                if (stop_at is not None and pos >= stop_at) or d.eof:
                    return pos
        finally:
            p.kill()
            p.wait()
        if off >= comp_end:
            return pos
        log(f"  CDL stream interrupted at {off / 1e9:.2f} GB, resuming")
        off, pos, d = ck[0], ck[1], ck[2].copy()


def _bigtiff_ifd(buf_read, ifd_off):
    """Parse a little-endian BigTIFF IFD -> {tag: values}; buf_read(offset, n) -> bytes."""
    types = {1: "B", 3: "H", 4: "I", 16: "Q", 12: "d", 2: "s", 7: "B", 11: "f"}
    sizes = {"B": 1, "H": 2, "I": 4, "Q": 8, "d": 8, "s": 1, "f": 4}
    n = struct.unpack("<Q", buf_read(ifd_off, 8))[0]
    raw = buf_read(ifd_off + 8, 20 * n)
    tags = {}
    for k in range(n):
        tag, typ, cnt = struct.unpack("<HHQ", raw[20 * k:20 * k + 12])
        f = types.get(typ)
        if f is None:
            continue
        nb = sizes[f] * cnt
        data = raw[20 * k + 12:20 * k + 20] if nb <= 8 else \
            buf_read(struct.unpack("<Q", raw[20 * k + 12:20 * k + 20])[0], nb)
        tags[tag] = data[:nb] if f == "s" else struct.unpack(f"<{cnt}{f}", data[:nb])
    return tags


def _stream_clip_10m(url, bounds, out_fp, tmp, margin=256 << 20, log=print):
    """Clip a national 10 m CDL confidence GeoTIFF stored deflated inside a zip, without
    holding the archive or the inflated image on disk.

    The deflate stream is not seekable and the BigTIFF IFD sits at its end, so: pass 1 inflates
    the whole stream once (bytes are discarded) keeping in-memory decompressor checkpoints and
    the bytes from shortly before the IFD to the end; pass 2 re-inflates, from the nearest
    checkpoint, only the span holding the tiles of the window. Both go into a sparse temporary
    file at their original offsets, which GDAL reads as a normal GeoTIFF; it is deleted after
    the window is written."""
    off, cs, meth = _zip_entry(url, "confidence_layer.tif")
    assert meth == 8
    lh = _range(url, off, off + 29)
    fl, el = struct.unpack("<HH", lh[26:30])
    c0 = off + 30 + fl + el
    head = zlib.decompressobj(-15).decompress(_range(url, c0, c0 + 65535))
    assert head[:4] == b"II+\x00", "expected little-endian BigTIFF"
    ifd_off = struct.unpack("<Q", head[8:16])[0]
    sparse = tmp / "cdlconf_sparse.tif"
    sparse.unlink(missing_ok=True)
    f = open(sparse, "w+b")
    f.write(head[:16])
    keep_from = max(16, ifd_off - margin)

    def tail_sink(pos, b):
        if pos + len(b) > keep_from:
            s = max(0, keep_from - pos)
            f.seek(pos + s)
            f.write(b[s:])

    ckpts = [(c0, 0, zlib.decompressobj(-15))]
    log(f"CDL: streaming {cs / 1e9:.1f} GB (pass 1, nothing kept but the TIFF directory)")
    _inflate_range(url, c0, c0 + cs, ckpts[0][2].copy(), 0, tail_sink, checkpoints=ckpts, log=log)

    def rd(o, n):
        if o < keep_from:
            raise IOError(f"TIFF structure at {o} before kept tail {keep_from}; raise margin")
        f.seek(o)
        return f.read(n)

    tg = _bigtiff_ifd(rd, ifd_off)
    W, H = tg[256][0], tg[257][0]
    tw, th = (tg[322][0], tg[323][0]) if 322 in tg else (W, tg[278][0])
    offs, cnts = (tg[324], tg[325]) if 324 in tg else (tg[273], tg[279])
    x0, y0 = tg[33922][3], tg[33922][4]
    res = tg[33550][0]
    c_a = int((bounds.left - x0) // res); c_b = int((bounds.right - x0) // res) + 1
    r_a = int((y0 - bounds.top) // res); r_b = int((y0 - bounds.bottom) // res) + 1
    ntx = -(-W // tw)
    need = [ty * ntx + tx for ty in range(r_a // th, (r_b - 1) // th + 1)
            for tx in range(c_a // tw, (c_b - 1) // tw + 1)]
    lo = min(offs[i] for i in need)
    hi = max(offs[i] + cnts[i] for i in need)
    log(f"CDL: window {c_b - c_a}x{r_b - r_a} px, {len(need)} blocks in "
        f"{(hi - lo) / 1e9:.2f} GB of the image (pass 2)")
    ck = max((c for c in ckpts if c[1] <= lo), key=lambda c: c[1])

    def span_sink(pos, b):
        a, z = max(pos, lo), min(pos + len(b), hi)
        if a < z:
            f.seek(a)
            f.write(b[a - pos:z - pos])

    _inflate_range(url, ck[0], c0 + cs, ck[2].copy(), ck[1], span_sink, stop_at=hi, log=log)
    f.close()
    with rasterio.open(sparse) as src:
        w = rasterio.windows.Window(c_a, r_a, c_b - c_a, r_b - r_a)
        arr = src.read(1, window=w)
        prof = dict(driver="GTiff", dtype=arr.dtype, count=1, crs=src.crs,
                    transform=src.window_transform(w), width=arr.shape[1], height=arr.shape[0],
                    compress="deflate", tiled=True)
    with rasterio.open(out_fp, "w", **prof) as dst:
        dst.write(arr, 1)
    sparse.unlink()


def cleanup(cfg):
    for p in cache_dir(cfg, "_tmp").glob("*_confidence_layer.tif"):
        p.unlink()
