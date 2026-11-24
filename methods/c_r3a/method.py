"""c_r3a: c_r1a EVI2 calendar ridge + pixel-sigmoid (multiple-instance) ln NIR retention model, maps averaged.

Grows from champion c_r1a. Component A is c_r1a unchanged: per-pixel EVI2 profile on the day-of-year grid,
pixel loss A = clip(b + w . profile_px, 0, 1), ridge on field-mean profiles.
Component B: per-pixel ln NIR (B08) retention curve on the same grid (ln B08(t) - pixel's max over the grid, as in
c_r2b), pixel loss B = sigmoid(v . z_px + c) with z = standardised curve.  v, c are fitted so that the FIELD MEAN of
the pixel sigmoids matches the field label (multiple-instance least squares, L2 penalty lambda; lambda by inner
leave-one-year-out C).  Pixel loss = 0.5 A + 0.5 B.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

SETTINGS = Path(__file__).with_name("settings.yaml")
CFG = yaml.safe_load(SETTINGS.read_text())
CACHE = Path(__file__).with_name("cache")          # label-free per-field features (speeds up fit)
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


def pixel_curves(f):
    """EVI2 profile (G, P) and ln NIR retention curve (G, P) for the outline pixels (NaN: too few clear dates)."""
    o = f.outline
    nir, red = f.band("B08"), f.band("B04")
    x = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)                     # (T, H, W)
    clr = np.isfinite(x[:, o]).mean(1)
    use = f.usable & (clr >= CFG["usable_share"])
    doy = ((f.dates - f.dates.astype("datetime64[Y]")).astype(int) + 1.0)[use]
    g = _grid()
    P = int(o.sum())
    if len(doy) == 0:
        return np.full((len(g), P), np.nan), np.full((len(g), P), np.nan)
    e = _interp_pixels(doy, x[use][:, o].astype("float64"), g)
    n = nir[use][:, o].astype("float64")
    ln = np.log(np.maximum(n, CFG["nir_floor"]))
    ln[~np.isfinite(n)] = np.nan
    r = _interp_pixels(doy, ln, g)
    with np.errstate(all="ignore"):
        r = r - np.nanmax(r, axis=0, keepdims=True)
    return e, r


def field_features(f):
    """Label-free: field-mean EVI2 profile (G,) and a fixed random sample of pixel retention curves (<= n, G)."""
    h = hashlib.md5(SETTINGS.read_bytes()).hexdigest()[:8]
    key = CACHE / f"{f.field_id}_{h}.npz"
    if key.exists():
        z = np.load(key)
        return z["emean"], z["rpix"]
    e, r = pixel_curves(f)
    em = np.nanmean(e, axis=1) if np.isfinite(e).any() else np.full(e.shape[0], np.nan)
    idx = np.where(np.isfinite(r).all(0))[0]
    rng = np.random.default_rng(int(hashlib.md5(f.field_id.encode()).hexdigest()[:8], 16))
    if 0 < CFG["mil_pixels_per_field"] < len(idx):
        idx = np.sort(rng.choice(idx, CFG["mil_pixels_per_field"], replace=False))
    rp = r[:, idx].T.astype("float32")
    CACHE.mkdir(exist_ok=True)
    np.savez(key, emean=em, rpix=rp)
    f.drop()
    return em, rp


def _cindex(s, claim, lossc):
    """Censored C as the scorer: paid with unknown loss (NaN) only in (paid, none) pairs."""
    known = np.isfinite(lossc)
    l = np.nan_to_num(lossc)
    A = ((claim & known)[:, None] & (claim & known)[None, :] & (l[:, None] > l[None, :])) | (claim[:, None] & ~claim[None, :])
    K = A * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))
    return K.sum() / max(A.sum(), 1)


# ---------------------------------------------------------------- component A: c_r1a ridge on EVI2 --
def _ridge(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return w / sd, y.mean() - (mu / sd) @ w


def _fit_ridge(X, y, claim, lossc, years):
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
    return dict(w=w, b=b, alpha=best[1], inner_c=best[0])


# ------------------------------------------------- component B: multiple-instance sigmoid on retention --
def _mil(Xs, y, lam):
    Z = np.vstack(Xs).astype("float64")
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    Z = (Z - mu) / sd
    n = np.array([len(x) for x in Xs])
    st = np.r_[0, np.cumsum(n)[:-1]]
    seg = np.repeat(np.arange(len(Xs)), n)
    d = Z.shape[1]

    def obj(th):
        w, b = th[:d], th[d]
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        r = np.add.reduceat(p, st) / n - y
        gp = (2 * r / n)[seg] * p * (1 - p)
        return (r ** 2).sum() + lam * (w ** 2).sum(), np.r_[Z.T @ gp + 2 * lam * w, gp.sum()]

    yb = float(np.clip(y.mean(), 0.05, 0.95))
    th = minimize(obj, np.r_[np.zeros(d), np.log(yb / (1 - yb))], jac=True, method="L-BFGS-B").x
    return dict(v=th[:d] / sd, c=th[d] - (mu / sd) @ th[:d])           # raw-scale: sigmoid(v . r + c)


def _mil_score(Xs, m):
    return np.array([np.mean(1.0 / (1.0 + np.exp(-(x.astype("float64") @ m["v"] + m["c"])))) for x in Xs])


def _fit_mil(Xs, y, claim, lossc, years):
    best = None
    for lam in CFG["mil_lambdas"]:
        s = np.zeros(len(y))
        for yr in np.unique(years):
            a = np.where(years != yr)[0]; b = np.where(years == yr)[0]
            m = _mil([Xs[i] for i in a], y[a], lam)
            s[b] = _mil_score([Xs[i] for i in b], m)
        c = _cindex(s, claim, lossc)
        if best is None or c > best[0]:
            best = (c, lam)
    m = _mil(Xs, y, best[1])
    m.update(lam=best[1], inner_c=best[0])
    return m


def fit(fields, labels):
    import multiprocessing as mp
    lab = labels.set_index("field_id")
    with mp.get_context("fork").Pool(8) as p:
        feats = p.map(field_features, fields)
    lab = lab.loc[[f.field_id for f in fields]]
    X = np.array([e for e, _ in feats])
    R = [r for _, r in feats]
    keep = np.isfinite(X).all(1) & np.array([len(r) > 0 for r in R])
    X, lab = X[keep], lab[keep]
    R = [r for r, k in zip(R, keep) if k]
    claim = lab.claim.to_numpy(bool)
    loss = lab.loss.to_numpy(float)
    y = np.where(claim, np.where(np.isfinite(loss), loss, CFG["heavy_nan_target"]), CFG["none_target"])
    lossc = np.where(claim, np.where(np.isfinite(loss), loss, np.nan), -1.0)
    years = lab.year.to_numpy()
    STATE["A"] = _fit_ridge(X, y, claim, lossc, years)
    STATE["B"] = _fit_mil(R, y, claim, lossc, years)


def predict(f):
    if "A" not in STATE:
        raise RuntimeError("fit first")
    e, r = pixel_curves(f)                                               # (G, P)
    A, B = STATE["A"], STATE["B"]
    with np.errstate(all="ignore"):
        pa = np.clip(A["w"] @ e + A["b"], 0.0, 1.0)
        pb = 1.0 / (1.0 + np.exp(-(B["v"] @ r + B["c"])))
    wa = CFG["weight_evi2_ridge"]
    pix = wa * pa + (1 - wa) * pb                                        # NaN where either curve is missing
    out = np.full(f.outline.shape, np.nan, dtype="float32")
    out[f.outline] = pix
    return out
