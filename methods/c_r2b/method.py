"""c_r2b: canopy-scattering retention per pixel.

Per outline pixel: ln NIR (B08) interpolated on a fixed day-of-year grid, minus the pixel's own seasonal maximum of
that curve -> retention curve r(t) <= 0 (0 = as much NIR scattering as at the pixel's own best moment; -0.7 = half of it).
The retention curve is scale-free: it removes the pixel's absolute brightness (soil, sun / view geometry level) and
keeps how much canopy scattering is left at each date relative to the pixel's own peak.
Pixel loss = clip(b + w . r_px, 0, 1); w, b by ridge on field-mean retention curves (development labels only).
No weather input.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

SETTINGS = Path(__file__).with_name("settings.yaml")
CFG = yaml.safe_load(SETTINGS.read_text())
CACHE = Path(__file__).with_name("cache")          # label-free per-field mean retention curves (speed only)
STATE = {}


def _grid():
    return np.arange(CFG["grid_start_doy"], CFG["grid_end_doy"], CFG["grid_step_days"], dtype="float64")


def _interp_pixels(doy, v, g):
    """v (T, P) with NaN -> (G, P) linear interpolation between each pixel's valid dates, edge values held."""
    ok = np.isfinite(v)
    T, P = v.shape
    ti = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(ok, ti, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, ti, T)[::-1], axis=0)[::-1]
    firstv, lastv = nxt[0], prev[-1]
    cols = np.arange(P)
    res = np.full((len(g), P), np.nan)
    for k, gd in enumerate(g):
        j = np.searchsorted(doy, gd, side="right") - 1
        if j < 0:
            a = firstv.copy(); b = firstv.copy()
        else:
            a = prev[j]
            b = nxt[j + 1] if j + 1 < T else np.full(P, T)
        a = np.where(a < 0, firstv, a); b = np.where(b >= T, lastv, b)
        a = np.clip(a, 0, T - 1); b = np.clip(b, 0, T - 1)
        va, vb, da, db = v[a, cols], v[b, cols], doy[a], doy[b]
        wgt = np.clip(np.where(db > da, (gd - da) / np.where(db > da, db - da, 1), 0.0), 0, 1)
        res[k] = va + wgt * (vb - va)
    res[:, ok.sum(0) < CFG["min_clear_dates"]] = np.nan
    return res


def pixel_retention(f):
    """(G, H, W) retention curve ln NIR(t) - max_t ln NIR per outline pixel; NaN elsewhere / too few clear dates."""
    o = f.outline
    nir = np.exp(np.nanmean(np.log(np.maximum(np.stack([f.band(b) for b in CFG["bands"]]), CFG["nir_floor"])), axis=0)) \
        if len(CFG["bands"]) > 1 else f.band(CFG["bands"][0])
    nir[~np.isfinite(f.band("B08"))] = np.nan
    clr = np.isfinite(nir[:, o]).mean(1)
    use = f.usable & (clr >= CFG["usable_share"])
    doy = ((f.dates - f.dates.astype("datetime64[Y]")).astype(int) + 1.0)[use]
    g = _grid()
    out = np.full((len(g),) + o.shape, np.nan, dtype="float32")
    if len(doy) == 0:
        return out
    v = np.log(np.maximum(nir[use][:, o].astype("float64"), CFG["nir_floor"]))
    v[~np.isfinite(nir[use][:, o])] = np.nan
    c = _interp_pixels(doy, v, g)
    with np.errstate(all="ignore"):
        top = np.sort(np.where(np.isfinite(c), c, -np.inf), axis=0)[-CFG["ref_top_k"]:]
        c = c - np.mean(top, axis=0, keepdims=True)
    out[:, o] = c
    return out


def mean_curve(f):
    key = CACHE / f"{f.field_id}_{hashlib.md5(SETTINGS.read_bytes()).hexdigest()[:8]}.npy"
    if key.exists():
        return np.load(key)
    p = pixel_retention(f)[:, f.outline]
    m = np.nanmean(p, axis=1) if np.isfinite(p).any() else np.full(p.shape[0], np.nan)
    CACHE.mkdir(exist_ok=True)
    np.save(key, m)
    f.drop()
    return m



def _ridge(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return w / sd, y.mean() - (mu / sd) @ w


def fit(fields, labels):
    import multiprocessing as mp
    lab = labels.set_index("field_id")
    with mp.get_context("fork").Pool(8) as p:
        X = np.array(p.map(mean_curve, fields))
    lab = lab.loc[[f.field_id for f in fields]]
    keep = np.isfinite(X).all(1)
    X, lab = X[keep], lab[keep]
    claim = lab.claim.to_numpy(bool)
    loss = lab.loss.to_numpy(float)
    y = np.where(claim, np.where(np.isfinite(loss), loss, CFG["heavy_nan_target"]), CFG["none_target"])
    # paid with unknown loss: in (paid, none) pairs only -> rank above none, not comparable among paid
    lossc = np.where(claim, np.where(np.isfinite(loss), loss, np.nan), -1.0)
    years = lab.year.to_numpy()
    best = None
    for a in CFG["alphas"]:
        s = np.zeros(len(y))
        for yr in np.unique(years):
            tr = years != yr
            w, b = _ridge(X[tr], y[tr], a)
            s[~tr] = X[~tr] @ w + b
        c = _cindex_unknown(s, claim, lossc)
        if best is None or c > best[0]:
            best = (c, a)
    w, b = _ridge(X, y, best[1])
    STATE.update(w=w, b=b, alpha=best[1], inner_c=best[0])


def _cindex_unknown(s, claim, lossc):
    known = np.isfinite(lossc)
    l = np.nan_to_num(lossc)
    A = ((claim & known)[:, None] & (claim & known)[None, :] & (l[:, None] > l[None, :])) | (claim[:, None] & ~claim[None, :])
    K = A * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))
    return K.sum() / max(A.sum(), 1)


def predict(f):
    if "w" not in STATE:
        raise RuntimeError("fit first")
    p = pixel_retention(f)
    s = np.tensordot(STATE["w"], p, axes=1) + STATE["b"]
    s = np.clip(s, 0.0, 1.0).astype("float32")
    s[~f.outline] = np.nan
    return s
