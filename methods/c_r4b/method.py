"""c_r4b: season analogues by elastic (DTW) matching of pixel spectral trajectories to labelled fields.

Main idea: nearest-neighbour analogues.  Every training field is kept as one reference trajectory: its field-mean
log reflectance in all 10 bands on a day-of-year grid.  Each outline pixel of a new field gets its own log
reflectance trajectory on the same grid and is matched to every reference trajectory by multivariate dynamic time
warping (shared warping path for all bands, Sakoe-Chiba band of +-W grid steps, so seasons that run earlier or later
can still match).  Pixel loss = mean label target of the k reference fields with the smallest warped distance.
No weights or regression on the profile: the labels enter only as the values carried by the analogues.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

SETTINGS = Path(__file__).with_name("settings.yaml")
CFG = yaml.safe_load(SETTINGS.read_text())
STATE = {}


# ---------------------------------------------------------------------------------------- features --
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
    y = np.where(claim, np.where(np.isfinite(loss), loss, CFG["heavy_nan_target"]), CFG["none_target"])
    sd = ref.reshape(-1, ref.shape[-1]).std(0) + 1e-9                    # per-band scale of the reference set
    STATE.update(ref=(ref / sd).astype("float32"), y=y.astype("float64"), sd=sd)


def predict(f):
    if "ref" not in STATE:
        raise RuntimeError("fit first")
    o = f.outline
    pc = pixel_curves(f)                                                 # (P, G, 10)
    ok = np.isfinite(pc).all((1, 2))
    s = np.full(pc.shape[0], np.nan)
    q = (pc[ok] / STATE["sd"]).astype("float32")
    k = min(CFG["k_neighbours"], len(STATE["y"]))
    ch = CFG["pixel_chunk"]
    vals = []
    for i in range(0, len(q), ch):
        d = dtw(q[i:i + ch], STATE["ref"], CFG["dtw_band"])
        nn = np.argpartition(d, k - 1, axis=1)[:, :k]
        vals.append(STATE["y"][nn].mean(1))
    if vals:
        s[ok] = np.concatenate(vals)
    out = np.full(o.shape, np.nan, dtype="float32")
    out[o] = np.clip(s, 0.0, 1.0)
    return out
