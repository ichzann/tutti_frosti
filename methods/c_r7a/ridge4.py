"""Copy of c_r4a (round 4 improver) method code, used unchanged as the ridge part of c_r5a / c_r6a / c_r7a (own settings + cache).

Four ridges on field-mean features, applied per pixel, each clipped to 0..1, averaged with equal weight:
prof (EVI2 calendar profile, c_r1a), ret (ln B08 grid minus pixel max, c_r2b), res (10-band spectrum at the pixel's
lowest-EVI2 date after its EVI2 maximum, c_r3b), resn (10-band spectrum at the pixel's lowest-B08 date after its EVI2
maximum, new).  Field mean of the map = mean of the four models on field-mean features (up to the clips).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

SETTINGS = Path(__file__).with_name("ridge4_settings.yaml")
CFG = yaml.safe_load(SETTINGS.read_text())
CACHE = Path(__file__).with_name("cache")          # label-free per-field mean features (speed only)
STATE = {}
COMPONENTS = CFG["components"]


# ---------------------------------------------------------------------------------------- features --
def _use(f, o):
    nir, red = f.band("B08"), f.band("B04")
    x = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)                     # (T, H, W) EVI2
    clr = np.isfinite(x[:, o]).mean(1)
    use = f.usable & (clr >= CFG["usable_share"])
    doy = (f.dates - f.dates.astype("datetime64[Y]")).astype(int) + 1.0
    return x[use][:, o].astype("float64"), doy[use], use


def _grid():
    return np.arange(CFG["grid_start_doy"], CFG["grid_end_doy"], CFG["grid_step_days"], dtype="float64")


def _profiles(v, doy):
    """(G, P) EVI2 linearly interpolated on the doy grid per pixel (c_r1a)."""
    g = _grid()
    ok = np.isfinite(v)
    T, P = v.shape
    res = np.full((len(g), P), np.nan)
    if T == 0:
        return res
    n = ok.sum(0)
    ti = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(ok, ti, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, ti, T)[::-1], axis=0)[::-1]
    firstv = nxt[0]; lastv = prev[-1]
    cols = np.arange(P)
    for k, gd in enumerate(g):
        j = np.searchsorted(doy, gd, side="right") - 1
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
        res[k] = va + np.clip(wgt, 0, 1) * (vb - va)
    res[:, n < CFG["min_clear_dates"]] = np.nan
    return res


def _residues(f, o, v, use):
    """Two (10, P) residue spectra per pixel, both after the pixel's own EVI2 maximum:
    at its lowest-EVI2 date (c_r3b) and at its lowest-NIR (B08) date (new).  NaN if no post-peak date."""
    T, P = v.shape
    out = np.full((10, P), np.nan)
    out_n = np.full((10, P), np.nan)
    if T == 0:
        return out, out_n
    ok = np.isfinite(v)
    k = np.where(ok, v, -np.inf).argmax(0)
    post = ok & (np.arange(T)[:, None] > k[None])
    has = post.any(0) & (ok.sum(0) >= CFG["min_clear_dates"])
    r = f.refl[use][:, :, o]                                             # (T, 10, P)
    cols = np.arange(P)
    jm = np.where(post, v, np.inf).argmin(0)
    out[:, has] = r[jm[has], :, cols[has]].T
    jn = np.where(post, r[:, 6], np.inf).argmin(0)                      # B08 is finite where EVI2 is
    out_n[:, has] = r[jn[has], :, cols[has]].T
    return out, out_n


def _retention(f, o, doy, use):
    """(G, P) ln B08 on the doy grid minus the pixel's own maximum over the grid (c_r2b signal)."""
    nir = f.band("B08")[use][:, o].astype("float64")
    ln = np.log(np.maximum(nir, CFG["nir_floor"]))
    ln[~np.isfinite(nir)] = np.nan
    c = _profiles(ln, doy)
    with np.errstate(all="ignore"):
        mx = np.nanmax(np.where(np.isfinite(c), c, -np.inf), 0, keepdims=True)
        return np.where(np.isfinite(c), c - mx, np.nan)


def pixel_features(f):
    """dict of per-pixel feature arrays (d, P): prof (EVI2 grid), ret (ln NIR retention grid), res, resn."""
    o = f.outline
    v, doy, use = _use(f, o)
    res, resn = _residues(f, o, v, use)
    return dict(prof=_profiles(v, doy), ret=_retention(f, o, doy, use), res=res, resn=resn)


def mean_features(f):
    key = CACHE / f"{f.field_id}_{hashlib.md5(SETTINGS.read_bytes()).hexdigest()[:8]}.npz"
    if key.exists():
        z = np.load(key)
        return {k: z[k] for k in z.files}
    d = pixel_features(f)
    m = {}
    for k, x in d.items():
        with np.errstate(all="ignore"):
            m[k] = np.nanmean(x, 1) if np.isfinite(x).any() else np.full(x.shape[0], np.nan)
    CACHE.mkdir(exist_ok=True)
    np.savez(key, **m)
    f.drop()
    return m


# --------------------------------------------------------------------------------------------- fit --
def _cindex(s, claim, lossc, known):
    pp = (claim & known)[:, None] & (claim & known)[None, :] & (lossc[:, None] > lossc[None, :])
    A = pp | (claim[:, None] & ~claim[None, :])
    K = A * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))
    return K.sum() / max(A.sum(), 1)


def _ridge(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return w / sd, y.mean() - (mu / sd) @ w


def _loyo(X, y, years, alpha):
    s = np.zeros(len(y))
    for yr in np.unique(years):
        tr = years != yr
        if tr.sum() == 0:
            continue
        w, b = _ridge(X[tr], y[tr], alpha)
        s[~tr] = X[~tr] @ w + b
    return s


def fit(fields, labels):
    import multiprocessing as mp
    lab = labels.set_index("field_id")
    with mp.get_context("fork").Pool(8) as p:
        feats = p.map(mean_features, fields)
    F = {k: np.array([d[k] for d in feats]) for k in COMPONENTS}
    lab = lab.loc[[f.field_id for f in fields]]
    keep = np.isfinite(F["prof"]).all(1)
    F = {k: x[keep] for k, x in F.items()}
    lab = lab[keep]
    claim = lab.claim.to_numpy(bool)
    loss = lab.loss.to_numpy(float)
    known = np.isfinite(loss)
    y = np.where(claim, np.where(known, loss, CFG["heavy_nan_target"]), CFG["none_target"])
    lossc = np.where(claim & known, loss, -1.0)
    years = lab.year.to_numpy()
    c = lambda s: _cindex(s, claim, lossc, known)                       # noqa: E731
    STATE.clear()
    for k in COMPONENTS:
        X = F[k]
        fill = np.nanmean(X, 0)
        X = np.where(np.isfinite(X), X, fill)                           # field without a value (e.g. no post-peak date)
        if CFG["alpha"][k] == "inner":                                  # inner LOYO C over the grid (c_r1a / c_r2b rule)
            sc = {a: _loyo(X, y, years, a) for a in CFG["profile_alphas"]}
            a = max(sc, key=lambda a: c(sc[a]))
        else:
            a = CFG["alpha"][k]
        w, b = _ridge(X, y, a)
        STATE[k] = dict(w=w, b=b, alpha=a, fill=fill)


def predict(f):
    if not STATE:
        raise RuntimeError("fit first")
    o = f.outline
    d = pixel_features(f)
    s = 0.0
    for k in COMPONENTS:
        m = STATE[k]
        x = d[k]
        v = m["w"] @ x + m["b"]
        if CFG["fill_missing"][k]:                                      # residues: pixel without a post-peak date
            hv = np.isfinite(v)
            v = np.where(hv, v, np.nanmean(v[hv]) if hv.any() else m["w"] @ m["fill"] + m["b"])
        s = s + np.clip(v, 0.0, 1.0) / len(COMPONENTS)                  # each component clipped, equal weight
    out = np.full(o.shape, np.nan, dtype="float32")
    out[o] = s                                                          # NaN where the EVI2 / NIR grid has < 3 dates
    return out
