"""c_r3b: post-season residue spectrum per pixel, blended with the c_r1a EVI2 calendar profile.

Main idea: what is left on the ground once the canopy is gone.  For every outline pixel the date of its own EVI2
maximum is found, then the date of its lowest EVI2 after that maximum (the pixel's own post-season / post-harvest
state, no calendar used).  The 10-band reflectance spectrum on that date is the pixel's "residue spectrum".  A ridge
on the field-mean residue spectra, fitted on development labels, maps it to loss.
Second component: the c_r1a EVI2 calendar profile ridge (unchanged).  Pixel loss =
lam * clip(b_r + w_r . residue_px, 0, 1) + (1 - lam) * clip(b_e + w_e . profile_px, 0, 1); both parts are linear in
pixel features, so the field mean of the map equals the model on field-mean features (up to the clips).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

SETTINGS = Path(__file__).with_name("settings.yaml")
CFG = yaml.safe_load(SETTINGS.read_text())
CACHE = Path(__file__).with_name("cache")          # label-free per-field mean features (speed only)
STATE = {}


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


def _residue(f, o, v, use):
    """(10, P) reflectance at each pixel's lowest-EVI2 date after its EVI2 maximum; NaN if no such date."""
    T, P = v.shape
    out = np.full((10, P), np.nan)
    if T == 0:
        return out
    ok = np.isfinite(v)
    k = np.where(ok, v, -np.inf).argmax(0)
    post = ok & (np.arange(T)[:, None] > k[None])
    jm = np.where(post, v, np.inf).argmin(0)
    has = post.any(0) & (ok.sum(0) >= CFG["min_clear_dates"])
    r = f.refl[use][:, :, o]                                             # (T, 10, P)
    cols = np.arange(P)
    out[:, has] = r[jm[has], :, cols[has]].T
    return out


def pixel_features(f):
    o = f.outline
    v, doy, use = _use(f, o)
    return _profiles(v, doy), _residue(f, o, v, use)                    # (G, P), (10, P)


def mean_features(f):
    key = CACHE / f"{f.field_id}_{hashlib.md5(SETTINGS.read_bytes()).hexdigest()[:8]}.npz"
    if key.exists():
        z = np.load(key)
        return z["p"], z["r"]
    p, r = pixel_features(f)
    mp_ = np.nanmean(p, 1) if np.isfinite(p).any() else np.full(p.shape[0], np.nan)
    mr = np.nanmean(r, 1) if np.isfinite(r).any() else np.full(r.shape[0], np.nan)
    CACHE.mkdir(exist_ok=True)
    np.savez(key, p=mp_, r=mr)
    f.drop()
    return mp_, mr


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
    P = np.array([a for a, _ in feats]); R = np.array([b for _, b in feats])
    lab = lab.loc[[f.field_id for f in fields]]
    keep = np.isfinite(P).all(1)
    P, R, lab = P[keep], R[keep], lab[keep]
    rmean = np.nanmean(R, 0)
    R = np.where(np.isfinite(R), R, rmean)                              # field without a post-peak date
    claim = lab.claim.to_numpy(bool)
    loss = lab.loss.to_numpy(float)
    known = np.isfinite(loss)
    y = np.where(claim, np.where(known, loss, CFG["heavy_nan_target"]), CFG["none_target"])
    lossc = np.where(claim & known, loss, -1.0)
    years = lab.year.to_numpy()
    c = lambda s: _cindex(s, claim, lossc, known)                       # noqa: E731
    se = {a: _loyo(P, y, years, a) for a in CFG["profile_alphas"]}
    ae = max(se, key=lambda a: c(se[a]))                                # profile alpha: inner LOYO C (c_r1a)
    ar = CFG["residue_alpha"]                                           # residue alpha: fixed (settings.yaml)
    lam = CFG["blend_lam"]
    wr, br = _ridge(R, y, ar)
    we, be = _ridge(P, y, ae)
    STATE.update(wr=wr, br=br, we=we, be=be, lam=lam, alpha_r=ar, alpha_e=ae, rmean=rmean,
                 inner_c_profile=c(se[ae]), inner_c_residue=c(_loyo(R, y, years, ar)))


def predict(f):
    if "wr" not in STATE:
        raise RuntimeError("fit first")
    o = f.outline
    p, r = pixel_features(f)
    sr = np.full(p.shape[1], np.nan)
    hr = np.isfinite(r).all(0)
    if hr.any():
        sr[hr] = STATE["wr"] @ r[:, hr] + STATE["br"]
        sr[~hr] = np.nanmean(sr[hr])                                    # pixel without a post-peak date
    else:
        sr[:] = STATE["wr"] @ STATE["rmean"] + STATE["br"]
    se = STATE["we"] @ p + STATE["be"]
    lam = STATE["lam"]
    s = lam * np.clip(sr, 0.0, 1.0) + (1 - lam) * np.clip(se, 0.0, 1.0)     # each component clipped to 0..1
    out = np.full(o.shape, np.nan, dtype="float32")
    out[o] = s
    return out
