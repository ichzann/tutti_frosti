"""Fast test v2: insured area = FTW field segments grown from the PSR point by shared phenology.

    python -m validation.brazil_psr.field_segments ftw       # composites -> FTW inference (ftw conda env)
    python -m validation.brazil_psr.field_segments match     # segments, growth, outcome table
    python -m validation.brazil_psr.field_segments figures

Same 50 wheat 2019 policies as field_match (v1).  Whole season (frost ignored), S2 B02 B03 B04 B08 B11.
Pixels   only winter-crop pixels (pixel_winter) stay in a field, so tree lines / roads / pasture are cut out;
         holes enclosed by a field are filled back (weak or frost-damaged pixels must not leave the field).
Borders  FTW 3-class pretrained model (CC-BY checkpoint, ftw-tools 1.4.3) on two clear-median composites:
         A = early season (WIN_A), B = late season (WIN_B).  Segments = 4-connected FTW "interior" pixels,
         >= MIN_SEG_PX; segment area counted with a 1-pixel ring (FTW boundary pixels are part of the field).
Curves   per segment and date: median over clear segment pixels (date kept if >= half the segment is clear);
         EVI2, NDMI (B08 vs B11) and B11 interpolated on an 8-day grid.  green-up / green-down = first
         crossing of half the EVI2 amplitude before / after the season peak.
Winter   segment-median EVI2: min <= SOW_MAX in SOW window, then peak >= PEAK_MIN in PEAK window and rise >= AMP_MIN
         (v1 pixel rule peak >= 0.50 dropped 13/50 starts, 3 of them paid frost claims: low peaks = damage).
Start    winter segment containing the point, else nearest within SEED_MAX_M.
Grow     frontier = winter segments within ADJ_PX of the group; a segment joins if green-up and green-down
         are within DAYS_MAX of the start segment and mean |diff| of EVI2 and NDMI curves <= CURVE_MAX; the
         best-scoring candidate joins first.  matched = prefix of the join order closest to insured area.
Outcome  clean:   whole joined group within [1 - TOL, 1 + TOL] x insured
         subset:  group larger, a prefix is within the tolerance
         too_small: whole group < (1 - TOL) x insured;  too_big: start segment alone > (1 + TOL) x insured
         no_start: no winter segment near the point
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage

from data.common import ROOT
from validation.brazil_psr.data import CRS
from validation.brazil_psr.field_match import N_PX, WIN_M, sample

OUT = ROOT / "cache" / "brazil_psr" / "fieldmatch_wheat2019_v2"
FTW = "/opt/homebrew/Caskroom/miniforge/base/envs/ftw/bin/ftw"
CKPT = ROOT / "cache" / "brazil_psr" / "ftw_models" / "3_Class_CCBY_FTW_Pretrained.ckpt"
BANDS = ["B02", "B03", "B04", "B08", "B11"]
WIN_A = ("2019-05-01", "2019-06-30")
WIN_B = ("2019-08-01", "2019-09-30")
SEASON = ("2019-03-01", "2019-11-30")
SOW = ("2019-03-15", "2019-06-30")
PEAK = ("2019-06-15", "2019-10-15")
SOW_MAX, PEAK_MIN, AMP_MIN = 0.35, 0.35, 0.15
GRID_DAYS = 8
MIN_SEG_PX = 50
CORE_EDGE, MIN_CORE_PX, EDGE_DROP = 0.35, 20, 0.75
SEED_MAX_M = 200
ADJ_PX = 3
DAYS_MAX = 12
CURVE_MAX = 0.08
TOL = 0.25
SNAPS = [("2018-12-01", "2019-02-28"), ("2019-12-01", "2020-02-29"), ("2020-07-01", "2020-09-30")]
SNAP_CLOUD, SNAP_CLEAR, SNAP_MAX = 30, 0.9, 0.03
V = "v2"                                                      # "v3" = v2 + the 3 RGB snapshots
PICK = ["W006", "W012", "W004", "W025", "W035", "W032"]       # the 6 random cases shown for v1


def set_year(y, out=None):
    """Point the season windows (and optionally OUT) at wheat season y; 2019 values are the defaults above."""
    global WIN_A, WIN_B, SEASON, SOW, PEAK, SNAPS, OUT
    WIN_A, WIN_B = (f"{y}-05-01", f"{y}-06-30"), (f"{y}-08-01", f"{y}-09-30")
    SEASON, SOW, PEAK = (f"{y}-03-01", f"{y}-11-30"), (f"{y}-03-15", f"{y}-06-30"), (f"{y}-06-15", f"{y}-10-15")
    SNAPS = [(f"{y - 1}-12-01", f"{y}-02-28"), (f"{y}-12-01", f"{y + 1}-02-28"), (f"{y + 1}-07-01", f"{y + 1}-09-30")]
    if out is not None:
        OUT = out


def _load(w):
    z = np.load(OUT / "raw" / f"{w}.npz")
    return z["dates"], z["clear"], z["refl"].astype("float32") / 10000.0


def composite(dates, clear, x, lo, hi):
    sel = (dates >= lo) & (dates <= hi)
    v = np.where(clear[sel][:, None], x[sel], np.nan)
    with np.errstate(all="ignore"):
        c = np.nanmedian(v, 0)
        alt = np.nanmedian(np.where(clear[:, None], x, np.nan), 0)
    return np.where(np.isfinite(c), c, alt)                     # (5, H, W)


# -------------------------------------------------------------------------------------- FTW --
def ftw():
    s = sample()
    d = OUT / "ftw"
    d.mkdir(exist_ok=True)
    for r in s.itertuples():
        out = d / f"{r.wid}_pred.tif"
        if out.exists():
            continue
        dates, clear, x = _load(r.wid)
        a, b = composite(dates, clear, x, *WIN_A), composite(dates, clear, x, *WIN_B)
        order = [2, 1, 0, 3]                                   # B04 B03 B02 B08, as ftw-tools stacks
        st = np.concatenate([a[order], b[order]]) * 10000
        st = np.nan_to_num(st, nan=0).clip(0, 65535).astype("uint16")
        inp = d / f"{r.wid}_in.tif"
        with rasterio.open(inp, "w", driver="GTiff", width=N_PX, height=N_PX, count=8, dtype="uint16",
                           crs=CRS, transform=from_origin(r.x0, r.y1, 10, 10)) as dst:
            dst.write(st)
        subprocess.run([FTW, "inference", "run", str(inp), "-m", str(CKPT), "-o", str(out), "-f",
                        "--patch_size", "192", "--mps_mode"], check=True, capture_output=True)
        print(r.wid, "ftw done", flush=True)


# -------------------------------------------------------------------------------- snapshots --
def snapshots(s=None):
    """v3: per window, the least cloudy clear RGB scene in each SNAPS range -> OUT/snaps/<wid>.npz."""
    import geopandas as gpd
    import planetary_computer as pc
    import pystac_client
    from shapely.geometry import box, shape
    from validation.brazil_psr.data import STAC, _offset, _read
    from validation.brazil_psr.field_match import _bounds
    cat = pystac_client.Client.open(STAC)
    d = OUT / "snaps"
    d.mkdir(exist_ok=True)
    for r in (sample() if s is None else s).itertuples():
        fp = d / f"{r.wid}.npz"
        if fp.exists():
            continue
        b = _bounds(r)
        bb = gpd.GeoSeries([box(*b)], crs=CRS).to_crs(4326).iloc[0]
        rgb = np.full((len(SNAPS), 3, N_PX, N_PX), np.nan, "float32")
        got = []
        for i, (lo, hi) in enumerate(SNAPS):
            items = sorted(cat.search(collections=["sentinel-2-l2a"], intersects=bb.__geo_interface__,
                                      datetime=f"{lo}/{hi}", query={"eo:cloud_cover": {"lt": SNAP_CLOUD}}).items(),
                           key=lambda it: it.properties["eo:cloud_cover"])
            date = ""
            for it in items[:8]:
                foot = gpd.GeoSeries([shape(it.geometry)], crs=4326).to_crs(CRS).iloc[0]
                if not foot.contains(box(*b)):
                    continue
                it = pc.sign(it)
                clear = np.isin(_read(it.assets["SCL"].href, b, (N_PX, N_PX)), [4, 5])
                if clear.mean() < SNAP_CLEAR:
                    continue
                for j, k in enumerate(["B02", "B03", "B04"]):
                    v = (_read(it.assets[k].href, b, (N_PX, N_PX)).astype("float32") + _offset(it)) / 10000.0
                    rgb[i, j] = np.where(clear, v, np.nan)
                date = it.datetime.date().isoformat()
                break
            got.append(date)
        np.savez_compressed(fp, rgb=rgb, dates=np.array(got))
        print(r.wid, got, flush=True)


def _snaps(w):
    fp = OUT / "snaps" / f"{w}.npz"
    return np.load(fp)["rgb"] if V == "v3" and fp.exists() else None


def _grad_norm(chans):
    grads = []
    for v in chans:
        v = ndimage.gaussian_filter(np.nan_to_num(v, nan=float(np.nanmedian(v)) if np.isfinite(v).any() else 0), 0.7)
        g = np.hypot(ndimage.sobel(v, 0), ndimage.sobel(v, 1))
        grads.append(g / (np.median(g) + 1e-6))
    sg = np.mean(grads, 0)
    return np.clip(sg / np.percentile(sg, 95), 0, 1)


# ---------------------------------------------------------------------------------- segments --
def edge_map(w, dates, clear, x):
    """0..1 border strength: mean of FTW boundary probability and a multi-temporal spectral gradient."""
    with rasterio.open(OUT / "ftw" / f"{w}_prob.tif") as src:
        fb = src.read(3)
    with np.errstate(all="ignore"):
        full = np.nanmedian(np.where(clear[:, None], x, np.nan), 0)
    grads = []
    for mo in range(3, 12):
        lo, hi = f"{SEASON[0][:4]}-{mo:02d}-01", f"{SEASON[0][:4]}-{mo:02d}-31"
        c = composite(dates, clear, x, lo, hi)
        c = np.where(np.isfinite(c), c, full)
        for b in range(c.shape[0]):
            v = ndimage.gaussian_filter(np.nan_to_num(c[b]), 0.7)
            g = np.hypot(ndimage.sobel(v, 0), ndimage.sobel(v, 1))
            grads.append(g / (np.median(g) + 1e-6))
    sg = np.mean(grads, 0)
    sg = np.clip(sg / np.percentile(sg, 95), 0, 1)
    sn = _snaps(w)
    if sn is None:
        return 0.5 * fb + 0.5 * sg
    chans = [sn[i, j] for i in range(len(SNAPS)) for j in range(3) if np.isfinite(sn[i, j]).mean() > 0.5]
    return 0.4 * fb + 0.3 * sg + 0.3 * _grad_norm(chans) if chans else 0.5 * fb + 0.5 * sg


def pixel_winter(dates, clear, x):
    """Per pixel: bare in the SOW window (min clear EVI2 <= SOW_MAX), then a rise >= AMP_MIN to >= PEAK_MIN."""
    with np.errstate(all="ignore"):
        ev = np.where(clear, 2.5 * (x[:, 3] - x[:, 2]) / (x[:, 3] + 2.4 * x[:, 2] + 1.0), np.nan)
        lo = np.nanmin(ev[(dates >= SOW[0]) & (dates <= SOW[1])], 0)
        hi = np.nanmax(ev[(dates >= PEAK[0]) & (dates <= PEAK[1])], 0)
    return (lo <= SOW_MAX) & (hi >= PEAK_MIN) & (hi - lo >= AMP_MIN)


def segments(w, dates, clear, x):
    """Fields = watershed of the border map from low-border cores; labels 1..n, 0 = dropped (tiny)."""
    e = edge_map(w, dates, clear, x)
    core = ndimage.binary_erosion(e < CORE_EDGE)
    mk, n = ndimage.label(core)
    sz = ndimage.sum(np.ones_like(mk), mk, index=np.arange(1, n + 1))
    mk[np.isin(mk, np.nonzero(sz < MIN_CORE_PX)[0] + 1)] = 0
    lab = ndimage.watershed_ift((e * 255).astype("uint8"), mk.astype("int32"))
    lab = np.where(e > EDGE_DROP, 0, lab)                      # strong border pixels belong to no field
    lab = np.where(pixel_winter(dates, clear, x), lab, 0)      # trees / roads / pasture leave the field
    parts, _ = ndimage.label(lab > 0)                          # a field cut by that falls apart into pieces
    lab = lab * (parts.max() + 1) + parts
    sizes = np.bincount(lab.ravel())
    lab[np.isin(lab, np.nonzero(sizes < MIN_SEG_PX)[0])] = 0
    for k in np.unique(lab[lab > 0]):                           # holes inside a field (weak / damaged pixels)
        m = lab == k                                            # are part of it
        lab[ndimage.binary_fill_holes(m) & (lab == 0)] = k
    u, inv = np.unique(lab, return_inverse=True)               # relabel 1..n, keep 0 = no field
    return (inv.reshape(lab.shape) + (u[0] != 0)).astype("int32"), e


def _grid():
    t0, t1 = (dt.date.fromisoformat(s).toordinal() for s in SEASON)
    return np.arange(t0, t1 + 1, GRID_DAYS)


def seg_curves(lab, dates, clear, x):
    """{seg: dict(evi2, ndmi, b11 on the grid)} from per-date segment medians."""
    d = np.array([dt.date.fromisoformat(s).toordinal() for s in dates])
    g = _grid()
    b, gr, r, n, s1 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    with np.errstate(all="ignore"):
        ev = 2.5 * (n - r) / (n + 2.4 * r + 1.0)
        nd = (n - s1) / (n + s1)
    out = {}
    for k in range(1, lab.max() + 1):
        m = lab == k
        npx = m.sum()
        rows = []
        for t in range(len(d)):
            c = clear[t] & m
            if c.sum() >= 0.5 * npx:
                rows.append((d[t], np.median(ev[t][c]), np.median(nd[t][c]), np.median(s1[t][c])))
        if len(rows) < 6:
            continue
        a = np.array(rows)
        out[k] = {nm: np.interp(g, a[:, 0], a[:, j]) for j, nm in [(1, "evi2"), (2, "ndmi"), (3, "b11")]}
        out[k]["n_obs"] = len(rows)
    return out


def phenology(c):
    g = _grid()
    o = lambda s: dt.date.fromisoformat(s).toordinal()
    ev = c["evi2"]
    pk = (g >= o(PEAK[0])) & (g <= o(PEAK[1]))
    ip = int(np.nonzero(pk)[0][np.argmax(ev[pk])])
    sow = (g >= o(SOW[0])) & (g <= o(SOW[1])) & (np.arange(len(g)) < ip)
    if not sow.any():
        return dict(winter=False)
    ib = int(np.nonzero(sow)[0][np.argmin(ev[sow])])
    peak, base = ev[ip], ev[ib]
    half = base + 0.5 * (peak - base)
    up = next((i for i in range(ib, ip + 1) if ev[i] >= half), ip)
    after = ev[ip:]
    end_base = after.min()
    half_dn = end_base + 0.5 * (peak - end_base)
    dn = next((ip + i for i in range(len(after)) if after[i] <= half_dn), len(g) - 1)
    return dict(winter=bool(peak >= PEAK_MIN and base <= SOW_MAX and peak - base >= AMP_MIN), peak=float(peak), base=float(base),
                greenup=int(g[up]), greendown=int(g[dn]))


def match_one(r):
    dates, clear, x = _load(r.wid)
    lab, edge = segments(r.wid, dates, clear, x)
    cy, cx = int(round((r.y1 - r.y) / 10 - 0.5)), int(round((r.x - r.x0) / 10 - 0.5))
    res = dict(wid=r.wid, n_segments=int(lab.max()), point_edge=float(edge[cy, cx]),
               point_in_segment=bool(lab[cy, cx] > 0))
    cur = seg_curves(lab, dates, clear, x)
    ph = {k: phenology(c) for k, c in cur.items()}
    win = [k for k, p in ph.items() if p["winter"]]
    res["n_winter_segments"] = len(win)
    area = {k: float((lab == k).sum()) / 100 for k in cur}
    if not win:
        return dict(res, outcome="no_start")
    yy, xx = np.mgrid[:N_PX, :N_PX]
    dist = {k: float(np.hypot(yy[lab == k] - cy, xx[lab == k] - cx).min() * 10) for k in win}
    start = min(win, key=dist.get)
    res["start_dist_m"] = dist[start]
    if dist[start] > SEED_MAX_M:
        return dict(res, outcome="no_start")
    ref, refc = ph[start], cur[start]
    sn = _snaps(r.wid)
    snap_rgb = {}
    if sn is not None:                                         # per segment, per snapshot median RGB
        for k in win:
            m = lab == k
            snap_rgb[k] = [np.nanmedian(sn[i][:, m], 1) if np.isfinite(sn[i][0][m]).mean() >= 0.5
                           else None for i in range(len(SNAPS))]
    grown = {k: ndimage.binary_dilation(lab == k, iterations=ADJ_PX) for k in win}

    def score(k):
        p = ph[k]
        du, dd = abs(p["greenup"] - ref["greenup"]), abs(p["greendown"] - ref["greendown"])
        me = float(np.abs(cur[k]["evi2"] - refc["evi2"]).mean())
        mn = float(np.abs(cur[k]["ndmi"] - refc["ndmi"]).mean())
        ok = du <= DAYS_MAX and dd <= DAYS_MAX and me <= CURVE_MAX and mn <= CURVE_MAX
        sc = du / DAYS_MAX + dd / DAYS_MAX + me / CURVE_MAX + mn / CURVE_MAX
        det = dict(du=du, dd=dd, me=me, mn=mn)
        if snap_rgb:                                           # same colour in every other-season snapshot
            ds = [float(np.abs(a - b).mean()) for a, b in zip(snap_rgb[k], snap_rgb[start])
                  if a is not None and b is not None]
            det["snap"] = max(ds) if ds else np.nan
            if ds:
                ok = ok and det["snap"] <= SNAP_MAX
                sc += det["snap"] / SNAP_MAX
        return ok, sc, det

    group, order, rejected = {start}, [start], {}
    gmask = grown[start].copy()
    while True:
        front = [k for k in win if k not in group and k not in rejected and (gmask & (lab == k)).any()]
        cand = []
        for k in front:
            ok, sc, det = score(k)
            if ok:
                cand.append((sc, k))
            else:
                rejected[k] = det
        if not cand:
            break
        _, k = min(cand)
        group.add(k)
        order.append(k)
        gmask |= grown[k]
    cum = np.cumsum([area[k] for k in order])
    ins = float(r.NR_AREA_TOTAL)
    j = int(np.argmin(np.abs(cum - ins)))
    best = cum[j] / ins
    if abs(cum[-1] / ins - 1) <= TOL:
        out, j = "clean", len(order) - 1
    elif cum[-1] < (1 - TOL) * ins:
        out = "too_small"
    elif cum[0] > (1 + TOL) * ins:
        out = "too_big"
    elif abs(best - 1) <= TOL:
        out = "subset"
    else:
        out = "too_big"                                         # a single join overshoots the tolerance
    np.savez_compressed(_mdir() / f"{r.wid}.npz", lab=lab, edge=edge, order=np.array(order),
                        matched=np.array(order[:j + 1]), rejected=np.array(list(rejected)),
                        point=np.array([cy, cx]))
    pd.to_pickle(dict(cur=cur, ph=ph, area=area, rejected=rejected), _mdir() / f"{r.wid}.pkl")
    return dict(res, outcome=out, start_ha=area[start], group_ha=float(cum[-1]), matched_ha=float(cum[j]),
                n_group=len(order), n_matched=j + 1, n_rejected_adjacent=len(rejected),
                group_ratio=float(cum[-1] / ins), matched_ratio=float(cum[j] / ins),
                start_greenup=dt.date.fromordinal(ref["greenup"]).isoformat(),
                start_greendown=dt.date.fromordinal(ref["greendown"]).isoformat())


def _mdir():
    d = OUT / ("match" if V == "v2" else f"match_{V}")
    d.mkdir(exist_ok=True)
    return d


def _res_fp():
    return OUT / ("results.parquet" if V == "v2" else f"results_{V}.parquet")


def match():
    s = sample()
    rows = []
    for r in s.itertuples():
        m = match_one(r)
        rows.append(m)
        print(r.wid, r.size_class, r.NR_AREA_TOTAL, m["outcome"], round(m.get("group_ratio", np.nan), 2),
              round(m.get("matched_ratio", np.nan), 2), flush=True)
    res = s.merge(pd.DataFrame(rows), on="wid")
    res.to_parquet(_res_fp())
    print(pd.crosstab(res.size_class, res.outcome, margins=True))
    return res


# ---------------------------------------------------------------------------------- figures --
def figure(r, fp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dates, clear, x = _load(r.wid)
    m = np.load(_mdir() / f"{r.wid}.npz")
    k = pd.read_pickle(_mdir() / f"{r.wid}.pkl")
    lab = m["lab"]
    e = np.arange(N_PX) * 10 + 5
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1.15, 1.15]))
    for a, (lo, hi, t) in zip(ax[:2], [(*WIN_A, "early season composite (May-Jun)"),
                                       (*WIN_B, "late season composite (Aug-Sep)")]):
        c = composite(dates, clear, x, lo, hi)
        a.imshow(np.clip(np.nan_to_num(c[[2, 1, 0]].transpose(1, 2, 0)) / 0.15, 0, 1), extent=(0, WIN_M, WIN_M, 0))
        a.contour(e, e, lab.astype(float), np.arange(lab.max()) + 0.5, colors="white", linewidths=0.4, alpha=0.6)
        for kk in m["rejected"]:
            a.contour(e, e, (lab == kk).astype(float), [0.5], colors="#ff5050", linewidths=1.2)
        a.contour(e, e, np.isin(lab, m["order"]).astype(float), [0.5], colors="cyan", linewidths=1.5,
                  linestyles="--")
        a.contour(e, e, np.isin(lab, m["matched"]).astype(float), [0.5], colors="yellow", linewidths=2.2)
        a.plot(m["point"][1] * 10 + 5, m["point"][0] * 10 + 5, "rx", ms=12, mew=3)
        a.set_title(t)
        a.set_xticks([0, 500, 1000, 1500, 2000]); a.set_yticks([0, 500, 1000, 1500, 2000])
    g = [dt.date.fromordinal(int(v)) for v in _grid()]
    for a, key, nm in [(ax[2], "evi2", "EVI2 (greenness)"), (ax[3], "ndmi", "NDMI (NIR vs SWIR moisture)")]:
        for kk in m["rejected"]:
            a.plot(g, k["cur"][kk][key], color="#ff5050", lw=1, alpha=0.7)
        for i, kk in enumerate(m["order"]):
            a.plot(g, k["cur"][kk][key], color="goldenrod" if kk in m["matched"] else "c",
                   lw=2.6 if i == 0 else 1.3, alpha=1 if i == 0 else 0.8)
        a.set_title(nm)
        a.tick_params(axis="x", rotation=30)
    p0 = k["ph"][int(m["order"][0])]
    for v in (p0["greenup"], p0["greendown"]):
        ax[2].axvline(dt.date.fromordinal(v), color="k", ls=":", lw=1)
    ax[2].plot([], [], color="goldenrod", lw=2.6, label="start field (thick) / matched")
    ax[2].plot([], [], color="c", label="joined, beyond insured area")
    ax[2].plot([], [], color="#ff5050", label="adjacent, rejected")
    ax[2].plot([], [], "k:", label="start green-up / green-down")
    ax[2].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{r.wid}  {r.NM_MUNICIPIO_PROPRIEDADE}-{r.SG_UF_PROPRIEDADE}  {r.lat:.5f}, {r.lon:.5f}   "
                 f"insured {r.NR_AREA_TOTAL:.1f} ha   matched {r.matched_ha:.1f} ha ({r.n_matched} fields)   "
                 f"joined group {r.group_ha:.1f} ha   outcome {r.outcome}")
    fig.tight_layout()
    fig.savefig(fp, dpi=85)
    plt.close(fig)


def figures():
    res = pd.read_parquet(_res_fp())
    fd = OUT / ("fig" if V == "v2" else f"fig_{V}")
    fd.mkdir(exist_ok=True)
    for w in PICK:
        r = res[res.wid == w].iloc[0]
        if r.outcome != "no_start":
            figure(r, fd / f"{w}.png")
    print("figures:", PICK)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        V = sys.argv[2]
    {"ftw": ftw, "snapshots": snapshots, "match": match, "figures": figures}[sys.argv[1]]()
