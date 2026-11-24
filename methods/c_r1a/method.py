"""c_r1a: learned calendar EVI2 profile per pixel (grows from low_peak_evi2).

Champion low_peak_evi2 scores 1 - season maximum of the field-median EVI2 (one fixed weight on one date).
Here every outline pixel gets its EVI2 profile on a fixed day-of-year grid (linear interpolation between its clear
usable dates); a ridge regression on the profile, fitted on development labels, gives that pixel's loss.
Pixel loss = clip(b + w . profile_px, 0, 1).  Fit uses the field-mean pixel profile (what the scorer averages).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

SETTINGS = Path(__file__).with_name("settings.yaml")
CFG = yaml.safe_load(SETTINGS.read_text())
CACHE = Path(__file__).with_name("cache")          # label-free per-field mean profiles (speeds up fit)
STATE = {}


def _grid():
    return np.arange(CFG["grid_start_doy"], CFG["grid_end_doy"], CFG["grid_step_days"], dtype="float64")


def pixel_profiles(f):
    """(G, H, W) EVI2 interpolated on the doy grid per pixel; NaN for pixels with too few clear dates."""
    o = f.outline
    nir, red = f.band("B08"), f.band("B04")
    x = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)                     # (T, H, W)
    clr = np.isfinite(x[:, o]).mean(1)
    use = f.usable & (clr >= CFG["usable_share"])
    doy = (f.dates - f.dates.astype("datetime64[Y]")).astype(int) + 1.0
    x, doy = x[use], doy[use]
    g = _grid()
    H, W = o.shape
    out = np.full((len(g), H, W), np.nan, dtype="float32")
    if len(doy) == 0:
        return out
    v = x[:, o].astype("float64")                                        # (T, P)
    ok = np.isfinite(v)
    T, P = v.shape
    n = ok.sum(0)
    ti = np.arange(T)[:, None]
    # index of last valid date <= t and first valid date >= t, per pixel
    prev = np.maximum.accumulate(np.where(ok, ti, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, ti, T)[::-1], axis=0)[::-1]
    firstv = nxt[0]; lastv = prev[-1]
    res = np.full((len(g), P), np.nan)
    cols = np.arange(P)
    for k, gd in enumerate(g):
        j = np.searchsorted(doy, gd, side="right") - 1                  # last date <= gd
        if j < 0:
            a = firstv.copy(); b = firstv.copy()
        else:
            a = prev[j]
            b = nxt[j + 1] if j + 1 < T else np.full(P, T)
        a = np.where(a < 0, firstv, a)
        b = np.where(b >= T, lastv, b)
        a = np.clip(a, 0, T - 1); b = np.clip(b, 0, T - 1)
        va, vb = v[a, cols], v[b, cols]
        da, db = doy[a], doy[b]
        wgt = np.where(db > da, (gd - da) / np.where(db > da, db - da, 1), 0.0)
        wgt = np.clip(wgt, 0, 1)
        res[k] = va + wgt * (vb - va)
    res[:, n < CFG["min_clear_dates"]] = np.nan
    out[:, o] = res
    return out


def mean_profile(f):
    key = CACHE / f"{f.field_id}_{hashlib.md5(SETTINGS.read_bytes()).hexdigest()[:8]}.npy"
    if key.exists():
        return np.load(key)
    p = pixel_profiles(f)[:, f.outline]
    m = np.nanmean(p, axis=1) if np.isfinite(p).any() else np.full(p.shape[0], np.nan)
    CACHE.mkdir(exist_ok=True)
    np.save(key, m)
    f.drop()
    return m


def _cindex(s, claim, loss, year=None):
    A = (claim[:, None] & claim[None, :] & (loss[:, None] > loss[None, :])) | (claim[:, None] & ~claim[None, :])
    K = A * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))
    return K.sum() / max(A.sum(), 1)


def _ridge(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return w / sd, y.mean() - (mu / sd) @ w                              # raw-scale weights, intercept


def fit(fields, labels):
    import multiprocessing as mp
    lab = labels.set_index("field_id")
    with mp.get_context("fork").Pool(8) as p:
        X = np.array(p.map(mean_profile, fields))
    fid = [f.field_id for f in fields]
    lab = lab.loc[fid]
    keep = np.isfinite(X).all(1)
    X, lab = X[keep], lab[keep]
    claim = lab.claim.to_numpy(bool)
    loss = lab.loss.to_numpy(float)
    y = np.where(claim, np.where(np.isfinite(loss), loss, CFG["heavy_nan_target"]), CFG["none_target"])
    lossc = np.where(claim, np.where(np.isfinite(loss), loss, CFG["heavy_nan_target"]), -1.0)
    years = lab.year.to_numpy()
    best = None
    for a in CFG["alphas"]:
        s = np.zeros(len(y))
        for yr in np.unique(years):
            tr = years != yr
            w, b = _ridge(X[tr], y[tr], a)
            s[~tr] = X[~tr] @ w + b
        c = _cindex(s, claim, lossc)
        if best is None or c > best[0]:
            best = (c, a)
    w, b = _ridge(X, y, best[1])
    STATE.update(w=w, b=b, alpha=best[1], inner_c=best[0])


def predict(f):
    p = pixel_profiles(f)                                                # (G, H, W)
    if "w" not in STATE:
        raise RuntimeError("fit first")
    s = np.tensordot(STATE["w"], p, axes=1) + STATE["b"]
    s = np.clip(s, 0.0, 1.0).astype("float32")
    s[~f.outline] = np.nan
    return s
