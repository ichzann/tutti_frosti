"""c_r5b: random convolutional kernel transform (MiniRocket-style) of pixel spectral time series + ridge.

Every outline pixel's season is a 10-band log reflectance series on a 4-day grid.  A fixed bank of random dilated
convolution kernels (each mixes 1-3 bands with random weights; MiniRocket kernel shapes, so they respond to the shape
of the series, not its level) is run along time.  Each kernel output is turned into PPV features: the share of the
season in which the output exceeds a threshold (thresholds = quantiles of the output on training pixels, label-free).
PPV counts how long a local temporal pattern is present, wherever in the season it occurs.  Pixel loss =
clip(b + w . ppv_px, 0, 1); w, b by ridge on field-mean PPV (= what the scorer averages before the clip).
"""
from __future__ import annotations

import multiprocessing as mp
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

CFG = yaml.safe_load(Path(__file__).with_name("settings.yaml").read_text())
STATE = {}
COMBOS = np.array(list(combinations(range(9), 3)))


# ---------------------------------------------------------------------------------------- series --
def _grid():
    return np.arange(CFG["grid_start_doy"], CFG["grid_end_doy"], CFG["grid_step_days"], dtype="float64")


def _interp(v, doy, g):
    ok = np.isfinite(v)
    T, P = v.shape
    res = np.full((len(g), P), np.nan)
    if T == 0:
        return res
    ti = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(ok, ti, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, ti, T)[::-1], axis=0)[::-1]
    firstv, lastv = nxt[0], prev[-1]
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
        w = np.where(db > da, (gd - da) / np.where(db > da, db - da, 1), 0.0)
        res[k] = va + np.clip(w, 0, 1) * (vb - va)
    res[:, ok.sum(0) < CFG["min_clear_dates"]] = np.nan
    return res


def pixel_series(f, sample=None):
    """(P, G, 10) log reflectance series of outline pixels (all, or a fixed random sample)."""
    o = f.outline
    idx = np.flatnonzero(o.ravel())
    if sample and len(idx) > sample:
        rng = np.random.default_rng(int(f.field_id[1:]))
        idx = np.sort(rng.choice(idx, sample, replace=False))
    r = f.refl.reshape(f.refl.shape[0], 10, -1)
    clr = np.isfinite(r[:, 6, o.ravel()]).mean(1)
    use = f.usable & (clr >= CFG["usable_share"])
    doy = ((f.dates - f.dates.astype("datetime64[Y]")).astype(int) + 1.0)[use]
    x = np.log(np.maximum(r[use][:, :, idx].astype("float64"), CFG["refl_floor"]))
    out = np.stack([_interp(x[:, b], doy, _grid()) for b in range(10)], -1)      # (G, P, 10)
    return np.transpose(out, (1, 0, 2)).astype("float32"), idx


# ---------------------------------------------------------------------------------------- kernels --
def make_kernels(G):
    rng = np.random.default_rng(CFG["seed"])
    maxd = int(np.log2((G - 1) / 8))
    ks = []
    for _ in range(CFG["n_kernels"]):
        d = int(2 ** rng.uniform(0, maxd + 0.999)) if maxd > 0 else 1
        nc = rng.integers(1, CFG["max_channels"] + 1)
        ch = rng.choice(10, nc, replace=False)
        cw = rng.normal(size=nc)
        if CFG["zero_mean_kernels"]:
            w = -np.ones(9); w[COMBOS[rng.integers(84)]] = 2.0
        else:
            w = rng.normal(size=9)
        ks.append((d, ch, cw, w, int(rng.integers(2))))
    return ks


def conv(X, k):
    d, ch, cw, w, pad = k
    x = (X[:, :, ch] * cw).sum(-1)
    G = x.shape[1]
    h = 4 * d
    xp = np.pad(x, ((0, 0), (h, h)), mode="edge")
    out = np.zeros_like(x)
    for j in range(9):
        out += w[j] * xp[:, j * d: j * d + G]
    if pad == 0 and G - 2 * h > 0:
        out = out[:, h:G - h]
    return out


def transform(X):
    """X (P, G, 10) standardised -> (P, F) PPV features."""
    out = []
    for k, b in zip(STATE["ks"], STATE["bs"]):
        o = conv(X, k)
        for bb in b:
            if "win3" in CFG["features"]:
                out += [p.mean(1) for p in np.array_split(o > bb, 3, axis=1)]
            else:
                out.append((o > bb).mean(1))
        if "max" in CFG["features"]:
            out.append(o.max(1))
    return np.stack(out, 1).astype("float32")


# -------------------------------------------------------------------------------------------- fit --
def _sample(f):
    x, _ = pixel_series(f, CFG["pixels_per_field_fit"])
    f.drop()
    return x[np.isfinite(x).all((1, 2))]


def _feat_mean(x):
    return transform((x - STATE["mu"]) / STATE["sd"]).mean(0) if len(x) else None


def _cindex(s, claim, loss):
    known = np.isfinite(loss)
    l = np.nan_to_num(loss)
    pp = (claim & known)[:, None] & (claim & known)[None, :] & (l[:, None] > l[None, :])
    A = pp | (claim[:, None] & ~claim[None, :])
    K = A * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))
    return K.sum() / max(A.sum(), 1)


def _ridge(X, y, a):
    m, v = X.mean(0), X.std(0) + 1e-6
    Z = (X - m) / v
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return w / v, ym - (m / v) @ w                       # raw-scale weights, intercept


def fit(fields, labels):
    lab = labels.set_index("field_id").loc[[f.field_id for f in fields]]
    with mp.get_context("fork").Pool(8) as p:
        xs = p.map(_sample, fields, chunksize=2)
    allx = np.concatenate([x[::4] for x in xs])
    STATE["mu"] = allx.mean((0, 1)); STATE["sd"] = allx.std((0, 1)) + 1e-9
    STATE["ks"] = make_kernels(allx.shape[1])
    rng = np.random.default_rng(1)
    sub = (allx[rng.choice(len(allx), min(2000, len(allx)), replace=False)] - STATE["mu"]) / STATE["sd"]
    STATE["bs"] = [np.quantile(conv(sub, k), CFG["bias_quantiles"]) for k in STATE["ks"]]
    with mp.get_context("fork").Pool(8) as p:
        F = p.map(_feat_mean, xs, chunksize=4)
    keep = np.array([v is not None for v in F])
    X = np.stack([v for v in F if v is not None])
    lab = lab[keep]
    claim, loss, year = lab.claim.to_numpy(bool), lab.loss.to_numpy(float), lab.year.to_numpy()
    y = np.where(claim, np.where(np.isfinite(loss), loss, CFG["heavy_nan_target"]), CFG["none_target"])
    best = None
    for a in CFG["alpha_grid"]:                           # inner leave-one-year-out C on the training years
        s = np.zeros(len(y))
        for yr in np.unique(year):
            tr = year != yr
            w, b = _ridge(X[tr], y[tr], a)
            s[~tr] = X[~tr] @ w + b
        c = _cindex(s, claim, loss)
        if best is None or c > best[0]:
            best = (c, a)
    w, b = _ridge(X, y, best[1])
    STATE.update(w=w, b=b, alpha=best[1], inner_c=best[0])


def predict(f):
    if "w" not in STATE:
        raise RuntimeError("fit first")
    o = f.outline
    x, _ = pixel_series(f)
    ok = np.isfinite(x).all((1, 2))
    s = np.full(len(x), np.nan)
    q = (x[ok] - STATE["mu"]) / STATE["sd"]
    ch = CFG["pixel_chunk"]
    vals = [transform(q[i:i + ch]) @ STATE["w"] + STATE["b"] for i in range(0, len(q), ch)]
    if vals:
        s[ok] = np.concatenate(vals)
    out = np.full(o.shape, np.nan, dtype="float32")
    out[o] = np.clip(s, 0.0, 1.0)
    return out
