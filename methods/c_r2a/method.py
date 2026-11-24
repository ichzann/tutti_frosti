"""c_r2a: learned calendar profiles of several indices per pixel (grows from c_r1a).

c_r1a: every outline pixel gets its EVI2 profile on a fixed day-of-year grid, pixel loss = clip(b + w . profile, 0, 1),
ridge on field-mean profiles.  Here the pixel profile is the concatenation of several index profiles (settings
`indices`), interpolated the same way over the pixel's clear usable dates; the rest is as in c_r1a.
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

INDEX = {
    "evi2": lambda b: 2.5 * (b("B08") - b("B04")) / (b("B08") + 2.4 * b("B04") + 1.0),
    "cire": lambda b: b("B07") / b("B05") - 1.0,
    "ndmi": lambda b: (b("B08") - b("B11")) / (b("B08") + b("B11")),
    "ndre": lambda b: (b("B8A") - b("B05")) / (b("B8A") + b("B05")),
    "nbr2": lambda b: (b("B11") - b("B12")) / (b("B11") + b("B12")),
    "red": lambda b: b("B04"),
    "swir": lambda b: b("B11"),
}


def _grid():
    return np.arange(CFG["grid_start_doy"], CFG["grid_end_doy"], CFG["grid_step_days"], dtype="float64")


def _interp(v, doy, g):
    """v (T, P) values (NaN = not clear) on ascending doy -> (G, P) linear interpolation per pixel (flat ends)."""
    ok = np.isfinite(v)
    T, P = v.shape
    ti = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(ok, ti, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, ti, T)[::-1], axis=0)[::-1]
    firstv = nxt[0]; lastv = prev[-1]
    res = np.full((len(g), P), np.nan)
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
    res[:, ok.sum(0) < CFG["min_clear_dates"]] = np.nan
    return res


def pixel_profiles(f, indices=None):
    """(F, H, W): per index in `indices`, the profile on the doy grid; NaN for pixels with too few clear dates."""
    indices = indices or CFG["indices"]
    o = f.outline
    with np.errstate(all="ignore"):
        ev = INDEX["evi2"](f.band)
        clr = np.isfinite(ev[:, o]).mean(1)                              # clear share (same rule as c_r1a)
        use = f.usable & (clr >= CFG["usable_share"])
        doy = (f.dates - f.dates.astype("datetime64[Y]")).astype(int) + 1.0
        doy = doy[use]
        g = _grid()
        H, W = o.shape
        out = np.full((len(g) * len(indices), H, W), np.nan, dtype="float32")
        if len(doy) == 0:
            return out
        for i, name in enumerate(indices):
            x = ev if name == "evi2" else INDEX[name](f.band)
            v = x[use][:, o].astype("float64")
            v[~np.isfinite(v)] = np.nan
            out[i * len(g):(i + 1) * len(g)][:, o] = _interp(v, doy, g)
    return out


def mean_profile(f, indices=None):
    indices = indices or CFG["indices"]
    tag = hashlib.md5(repr((indices, CFG["grid_start_doy"], CFG["grid_end_doy"], CFG["grid_step_days"],
                            CFG["usable_share"], CFG["min_clear_dates"])).encode()).hexdigest()[:8]
    key = CACHE / f"{f.field_id}_{tag}.npy"
    if key.exists():
        return np.load(key)
    p = pixel_profiles(f, indices)[:, f.outline]
    m = np.nanmean(p, axis=1) if np.isfinite(p).any() else np.full(p.shape[0], np.nan)
    CACHE.mkdir(exist_ok=True)
    np.save(key, m)
    f.drop()
    return m


def _cindex(s, claim, loss):
    A = (claim[:, None] & claim[None, :] & (loss[:, None] > loss[None, :])) | (claim[:, None] & ~claim[None, :])
    K = A * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))
    return K.sum() / max(A.sum(), 1)


def _ridge(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return w / sd, y.mean() - (mu / sd) @ w


def fit_xy(X, lab):
    """Ridge fit from field-mean features X and labels (claim, loss, year); alpha by inner leave-one-year-out C
    with a tie rule (largest alpha within alpha_tie of the best)."""
    keep = np.isfinite(X).all(1)
    X, lab = X[keep], lab[keep]
    claim = lab.claim.to_numpy(bool)
    loss = lab.loss.to_numpy(float)
    y = np.where(claim, np.where(np.isfinite(loss), loss, CFG["heavy_nan_target"]), CFG["none_target"])
    lossc = np.where(claim, np.where(np.isfinite(loss), loss, CFG["heavy_nan_target"]), -1.0)
    years = lab.year.to_numpy()
    cs = []
    for a in CFG["alphas"]:
        s = np.zeros(len(y))
        for yr in np.unique(years):
            tr = years != yr
            w, b = _ridge(X[tr], y[tr], a)
            s[~tr] = X[~tr] @ w + b
        cs.append(_cindex(s, claim, lossc))
    cs = np.array(cs)
    # tie rule: the largest alpha whose inner C is within `alpha_tie` of the best inner C
    k = int(np.flatnonzero(cs >= cs.max() - CFG["alpha_tie"]).max())
    w, b = _ridge(X, y, CFG["alphas"][k])
    return dict(w=w, b=b, alpha=CFG["alphas"][k], inner_c=float(cs[k]), inner_c_best=float(cs.max()))


def fit(fields, labels):
    import multiprocessing as mp
    lab = labels.set_index("field_id")
    with mp.get_context("fork").Pool(8) as p:
        X = np.array(p.map(mean_profile, fields))
    lab = lab.loc[[f.field_id for f in fields]]
    STATE.update(fit_xy(X, lab))


def predict(f):
    if "w" not in STATE:
        raise RuntimeError("fit first")
    p = pixel_profiles(f)
    s = np.tensordot(STATE["w"], p, axes=1) + STATE["b"]
    s = np.clip(s, 0.0, 1.0).astype("float32")
    s[~f.outline] = np.nan
    return s
