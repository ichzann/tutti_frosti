"""c_mt shared helpers: settings, indices, field / pixel series, intervals, normal-change tables, z and persistence."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import uniform_filter

HERE = Path(__file__).parent
SETTINGS = HERE / "settings.yaml"
CFG = yaml.safe_load(SETTINGS.read_text())
CACHE = HERE / "cache"


def cfg_hash():
    return hashlib.md5(SETTINGS.read_bytes()).hexdigest()[:8]


def clock_fields():
    """Fields allowed into Module A / imagery tables: DEV split only (user decision 2026-11-09)."""
    from competition.field import Field, field_table
    t = field_table()
    assert CFG["clock_split"] == "dev"
    t = t[t.split == "dev"]
    return [Field(r.field_id, r.year) for r in t.itertuples()]


def indices(R):
    """R (..., 10) reflectance floored -> ndmi, re, vis, evi2."""
    R = np.maximum(R, CFG["refl_floor"])
    ndmi = (R[..., 7] - R[..., 8]) / (R[..., 7] + R[..., 8])
    re = np.log(R[..., 5] / R[..., 3])
    vis = np.log(R[..., 0:3].mean(-1))
    evi2 = 2.5 * (R[..., 6] - R[..., 2]) / (R[..., 6] + 2.4 * R[..., 2] + 1)
    return ndmi, re, vis, evi2


def doy_of(dates):
    return ((dates - dates.astype("datetime64[Y]")).astype(int) + 1).astype("float64")


def usable_dates(f):
    o = f.outline
    r = f.refl[:, :, o]
    use = f.usable & (np.isfinite(r[:, 6]).mean(1) >= CFG["usable_share"])
    return use, r


def field_series(f):
    """doy (Tu,), indices (Tu,) each, from the field-median bands on usable dates."""
    use, r = usable_dates(f)
    doy = doy_of(f.dates)[use]
    if use.sum() == 0:
        e = np.zeros(0)
        return doy, e, e, e, e
    R = np.nanmedian(r[use].astype("float64"), 2)
    return (doy, *indices(R))


def pixel_series(f, pix=None):
    """3x3 smoothing over clear outline pixels (sum value*mask / sum mask) on usable dates.
    Returns doy (Tu,), ndmi, re, vis, evi2 each (Tu, P) with NaN where the pixel is not clear; P = outline pixels
    (or the subset `pix`, indices into the outline pixel list)."""
    use, _ = usable_dates(f)
    doy = doy_of(f.dates)[use]
    o = f.outline
    x = f.refl[use]                                                     # (Tu, 10, H, W)
    m = np.isfinite(x[:, 6]) & o[None]                                  # clear outline pixels
    xv = np.where(m[:, None], np.nan_to_num(x), 0.0).astype("float64")
    num = uniform_filter(xv, size=(1, 1, 3, 3), mode="constant")
    den = uniform_filter(m.astype("float64"), size=(1, 3, 3), mode="constant")
    with np.errstate(all="ignore"):
        sm = num / den[:, None]
    sm = sm[:, :, o]                                                    # (Tu, 10, P)
    clr = m[:, o]
    if pix is not None:
        sm, clr = sm[:, :, pix], clr[:, pix]
    sm = np.where(clr[:, None], sm, np.nan)
    ndmi, re, vis, evi2 = indices(np.moveaxis(sm, 1, -1))
    for a in (ndmi, re, vis, evi2):
        a[~clr] = np.nan
    return doy, ndmi, re, vis, evi2


def run_median3(x):
    if len(x) < 3:
        return x.copy()
    m = x.copy()
    m[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), 0)
    return m


def heading(doy, re):
    if len(doy) < CFG["min_dates_heading"]:
        return np.nan
    m = run_median3(re)
    lo, hi = CFG["heading_doy"]
    w = np.flatnonzero((doy >= lo) & (doy <= hi))
    if len(w) == 0:
        return np.nan
    i = w[np.argmax(m[w])]
    return np.nan if i in (w[0], w[-1]) else float(doy[i])


def intervals(doy, ndmi, re, vis, evi2):
    """Consecutive clear dates per column.  Inputs (T, P) (NaN = not clear).  Returns dict of 1-D arrays over all
    intervals (a, b) with dt_min <= b - a <= dt_max: col, a, b, dt, canopy, dN, dR, dV, and c-values
    (dNc, dRc, dVc; NaN if no next clear date c within dt_max of b)."""
    ok = np.isfinite(ndmi) & np.isfinite(re) & np.isfinite(vis) & np.isfinite(evi2)
    T, P = ok.shape
    ti = np.arange(T)[:, None]
    nxt_incl = np.minimum.accumulate(np.where(ok, ti, T)[::-1], axis=0)[::-1]      # first clear index >= t
    nxt = np.vstack([nxt_incl[1:], np.full((1, P), T)])                            # first clear index > t
    tt, pp = np.nonzero(ok)
    bb = nxt[tt, pp]
    has = bb < T
    tt, pp, bb = tt[has], pp[has], bb[has]
    dt = doy[bb] - doy[tt]
    k = (dt >= CFG["dt_min"]) & (dt <= CFG["dt_max"])
    tt, pp, bb, dt = tt[k], pp[k], bb[k], dt[k]
    cc = nxt[bb, pp]
    hc = cc < T
    cc_s = np.where(hc, cc, 0)
    hc &= (doy[cc_s] - doy[bb]) <= CFG["dt_max"]
    out = dict(col=pp, a=doy[tt], b=doy[bb], dt=dt, canopy=evi2[tt, pp] >= CFG["canopy_evi2"])
    for nm, v in (("N", ndmi), ("R", re), ("V", vis)):
        out["d" + nm] = v[bb, pp] - v[tt, pp]
        out["d" + nm + "c"] = np.where(hc, v[cc_s, pp] - v[tt, pp], np.nan)
    return out


# ------------------------------------------------------------------------------ normal-change tables --
NB = 2 * CFG["stage_lim"] // CFG["stage_bin"]                           # 14 stage bins


def stage_bin(s):
    b = np.floor((s + CFG["stage_lim"]) / CFG["stage_bin"])
    return np.clip(np.nan_to_num(b, nan=0), 0, NB - 1).astype(int)


def dt_class(dt):
    c = CFG["dt_classes"]
    return np.where(dt <= c[0], 0, np.where(dt <= c[1], 1, 2))


def _loc_scale(x):
    mu = np.median(x, 0)
    return mu, 1.4826 * np.median(np.abs(x - mu), 0)


def build_table(s, dtc, D):
    """s stage (NaN = no anchor), dtc dt class, D (n, 3) changes dN dR dV of the model's intervals.
    Returns table dict: per dt class a bin->group map, group mu/sigma (3), pooled mu/sigma; sd_c and sd_cN."""
    hasH = np.isfinite(s)
    sb = stage_bin(s)
    T = dict(group=np.zeros((3, NB), int), mu=np.zeros((3, NB, 3)), sig=np.zeros((3, NB, 3)),
             pmu=np.zeros((3, 3)), psig=np.zeros((3, 3)))
    sigs = []
    for c in range(3):
        cnt = np.array([(hasH & (dtc == c) & (sb == b)).sum() for b in range(NB)])
        groups = [[b] for b in range(NB)]
        gc = list(cnt)
        while len(groups) > 1 and min(gc) < CFG["min_cell"]:
            i = int(np.argmin(gc))
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
            j = min(nb, key=lambda j: gc[j])
            lo, hi = min(i, j), max(i, j)
            groups[lo] = groups[lo] + groups[hi]; gc[lo] = gc[lo] + gc[hi]
            del groups[hi]; del gc[hi]
        for gi, g in enumerate(groups):
            sel = hasH & (dtc == c) & np.isin(sb, g)
            mu, sg = _loc_scale(D[sel]) if sel.any() else (np.zeros(3), np.full(3, np.nan))
            for b in g:
                T["group"][c, b] = gi
                T["mu"][c, b], T["sig"][c, b] = mu, sg
            sigs.append(sg)
        sel = dtc == c
        T["pmu"][c], T["psig"][c] = _loc_scale(D[sel])
        sigs.append(T["psig"][c])
    floor = np.nanquantile(np.array(sigs), CFG["sigma_floor_q"], axis=0)
    T["sig"] = np.maximum(np.where(np.isfinite(T["sig"]), T["sig"], floor), floor)
    T["psig"] = np.maximum(T["psig"], floor)
    T["sd_c"], T["sd_cN"] = 1.0, 1.0
    zs = zscore(T, s, dtc, D)
    T["sd_c"] = float(1.4826 * np.median(np.abs(zs["zm"] - np.median(zs["zm"]))))
    T["sd_cN"] = float(1.4826 * np.median(np.abs(zs["zN"] - np.median(zs["zN"]))))
    return T


def cell(T, s, dtc):
    hasH = np.isfinite(s)
    sb = stage_bin(s)
    mu = np.where(hasH[:, None], T["mu"][dtc, sb], T["pmu"][dtc])
    sg = np.where(hasH[:, None], T["sig"][dtc, sb], T["psig"][dtc])
    return mu, sg


def zscore(T, s, dtc, D, Dc=None):
    """Composite z (negative = damage-like), NDMI-only z, persistence pi (if Dc given)."""
    mu, sg = cell(T, s, dtc)
    zz = (D - mu) / sg
    zz[:, 2] *= -1                                                      # VIS: rise = damage-like
    out = dict(zm=zz.mean(1), zN=zz[:, 0])
    out["z"] = out["zm"] / T["sd_c"]
    out["zNc"] = out["zN"] / T["sd_cN"]
    if Dc is not None:
        w = np.array([-1.0, -1.0, 1.0]) / sg
        Db = (D * w).sum(1)
        Dcc = (Dc * w).sum(1)
        pi = np.where(np.isfinite(Dcc), np.where(Dcc >= CFG["persist_frac"] * Db, 1.0, 0.0), 0.5)
        out["pi"] = pi
    return out


def S_curve(s):
    k = np.array(CFG["S_knots"], float)
    s = np.asarray(s, float)
    v = np.interp(s, k[:, 0], k[:, 1], left=k[0, 1], right=0.0)
    return np.where(np.isfinite(s), v, CFG["S_nan"])
