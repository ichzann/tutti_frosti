"""c_r7b: season manifold regression.

Main idea: the training fields' season trajectories (log reflectance, 10 bands, doy grid) are laid out on a
label-free low-dimensional manifold: diffusion map (Coifman & Lafon) of the elastic (DTW) distances between the
trajectories.  Loss is a smooth (ridge) function of the first m diffusion coordinates, fitted to the field labels.
Each outline pixel is placed on the same manifold by the Nystrom out-of-sample extension of its DTW distances to the
training trajectories, and gets loss = ridge(coordinates), clipped 0..1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

SETTINGS = Path(__file__).with_name("settings.yaml")
CFG = yaml.safe_load(SETTINGS.read_text())
STATE = {}


# ------------------------------------------------------------------- trajectories (as c_r4b, copied) --
def _grid():
    return np.arange(CFG["grid_start_doy"], CFG["grid_end_doy"], CFG["grid_step_days"], dtype="float64")


def _interp(v, doy, g):
    """v (T, P) with NaN gaps -> (G, P) linear interpolation in time per pixel, flat beyond the ends."""
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


def pixel_curves(f):
    """(P, G, 10) log reflectance trajectories of the outline pixels (NaN rows: too few clear dates)."""
    o = f.outline
    r = f.refl[:, :, o]                                                  # (T, 10, P)
    clr = np.isfinite(r[:, 6]).mean(1)
    use = f.usable & (clr >= CFG["usable_share"])
    doy = ((f.dates - f.dates.astype("datetime64[Y]")).astype(int) + 1.0)[use]
    r = np.log(np.maximum(r[use].astype("float64"), CFG["refl_floor"]))
    g = _grid()
    out = np.stack([_interp(r[:, b], doy, g) for b in range(r.shape[1])], -1)   # (G, P, 10)
    return np.transpose(out, (1, 0, 2))


def field_curve(f):
    """(G, 10) reference trajectory: log of the field-mean reflectance per usable date, interpolated."""
    o = f.outline
    r = f.refl[:, :, o]
    clr = np.isfinite(r[:, 6]).mean(1)
    use = f.usable & (clr >= CFG["usable_share"])
    doy = ((f.dates - f.dates.astype("datetime64[Y]")).astype(int) + 1.0)[use]
    m = np.nanmean(r[use].astype("float64"), 2) if use.any() else np.zeros((0, r.shape[1]))
    m = np.log(np.maximum(m, CFG["refl_floor"]))
    c = _interp(m, doy, _grid())                                         # (G, 10)
    f.drop()
    return c


# --------------------------------------------------------------------------------------------- DTW --
def dtw(A, B, W):
    """A (n, G, c), B (m, G, c) -> (n, m) multivariate DTW cost (squared Euclidean, shared path, band W)."""
    n, G, _ = A.shape
    m = B.shape[0]
    a2 = (A ** 2).sum(-1)                                                # (n, G)
    b2 = (B ** 2).sum(-1)                                                # (m, G)
    inf = np.float32(np.inf)
    R0 = np.full((G + 1, n, m), inf, dtype="float32")
    R0[0] = 0
    for i in range(1, G + 1):
        R1 = np.full((G + 1, n, m), inf, dtype="float32")
        ai = A[:, i - 1]
        for j in range(max(1, i - W), min(G, i + W) + 1):
            c = a2[:, i - 1, None] + b2[None, :, j - 1] - 2.0 * (ai @ B[:, j - 1].T)
            R1[j] = np.maximum(c, 0) + np.minimum(np.minimum(R0[j], R1[j - 1]), R0[j - 1])
        R0 = R1
    return R0[G]


# ------------------------------------------------------------------------------------- diffusion map --
def _embed(Dtr):
    """Diffusion map of a symmetric distance matrix -> extension function (distances to refs -> coordinates)."""
    off = Dtr[~np.eye(len(Dtr), dtype=bool)]
    eps = float(np.quantile(off, CFG["eps_quantile"]))
    a = CFG["dm_alpha"]
    K = np.exp(-Dtr / eps)
    q = K.sum(1)
    Ka = K / np.outer(q, q) ** a
    d = Ka.sum(1)
    w, V = np.linalg.eigh(Ka / np.sqrt(np.outer(d, d)))
    o = np.argsort(-w)[1:CFG["n_coords"] + 1]                          # drop the trivial first eigenvector
    w, V = w[o], V[:, o]
    psi = V / np.sqrt(d)[:, None]
    return dict(eps=eps, q=q, w=w, psi=psi)


def _extend(E, Dq):
    """Nystrom extension: (n, N_ref) distances -> (n, m) diffusion coordinates."""
    Kq = np.exp(-Dq / E["eps"])
    Kqa = Kq / (Kq.sum(1)[:, None] * E["q"][None]) ** CFG["dm_alpha"]
    P = Kqa / np.maximum(Kqa.sum(1)[:, None], 1e-300)
    return (P @ E["psi"]) / E["w"]


def _ridge(X, y, lam):
    mu, ym = X.mean(0), y.mean()
    Xc = X - mu
    b = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return b, ym - mu @ b


def _cindex(s, claim, loss):
    known = claim & np.isfinite(loss)
    l0 = np.nan_to_num(loss)
    A = (known[:, None] & known[None, :] & (l0[:, None] > l0[None, :])) | (claim[:, None] & ~claim[None, :])
    K = (s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :])
    return float((A * K).sum() / A.sum())


# --------------------------------------------------------------------------------------------- fit --
def fit(fields, labels):
    import multiprocessing as mp
    lab = labels.set_index("field_id").loc[[f.field_id for f in fields]]
    with mp.get_context("fork").Pool(8) as p:
        ref = np.array(p.map(field_curve, fields))                       # (n, G, 10)
    keep = np.isfinite(ref).all((1, 2))
    ref, lab = ref[keep], lab[keep]
    claim = lab.claim.to_numpy(bool)
    loss = lab.loss.to_numpy(float)
    year = lab.year.to_numpy()
    y = np.where(claim, np.where(np.isfinite(loss), loss, CFG["heavy_nan_target"]), CFG["none_target"])
    sd = ref.reshape(-1, ref.shape[-1]).std(0) + 1e-9                    # per-band scale (label-free)
    R = (ref / sd).astype("float32")
    D = dtw(R, R, CFG["dtw_band"]).astype("float64")
    D = (D + D.T) / 2
    E = _embed(D)
    X = _extend(E, D)
    xs = X.std(0) + 1e-12
    X = X / xs
    # ridge strength by inner leave-one-year-out C over the training years (embedding fixed, label-free)
    best = None
    for lam in CFG["ridge_alphas"]:
        s = np.full(len(y), np.nan)
        for yr in np.unique(year):
            te = year == yr
            b, c = _ridge(X[~te], y[~te], lam)
            s[te] = X[te] @ b + c
        cv = _cindex(s, claim, loss)
        if best is None or cv > best[0] + 1e-12:
            best = (cv, lam)
    b, c = _ridge(X, y, best[1])
    STATE.update(ref=R, sd=sd, E=E, xs=xs, b=b, c=c, lam=best[1], inner_c=best[0])


def predict(f):
    if "ref" not in STATE:
        raise RuntimeError("fit first")
    o = f.outline
    pc = pixel_curves(f)                                                 # (P, G, 10)
    ok = np.isfinite(pc).all((1, 2))
    s = np.full(pc.shape[0], np.nan)
    q = (pc[ok] / STATE["sd"]).astype("float32")
    ch = CFG["pixel_chunk"]
    vals = []
    for i in range(0, len(q), ch):
        d = dtw(q[i:i + ch], STATE["ref"], CFG["dtw_band"]).astype("float64")
        X = _extend(STATE["E"], d) / STATE["xs"]
        vals.append(X @ STATE["b"] + STATE["c"])
    if vals:
        s[ok] = np.concatenate(vals)
    out = np.full(o.shape, np.nan, dtype="float32")
    out[o] = np.clip(s, 0.0, 1.0)
    return out
