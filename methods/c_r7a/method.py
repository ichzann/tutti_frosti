"""c_r7a: c_r6a (analogues 0.7 + c_r4a ridge 0.3) with DTW cell cost (squared distance) ** 1.25 and atypical-pixel replacement.

Code taken from c_r5a (round 5); only the blend weight differs (settings.yaml ridge_weight).

Part 1 (from c_r4b, one change): every training field is kept as one reference trajectory, now the log of its
field-MEDIAN reflectance per usable date (c_r4b: log of the field mean); pixel trajectories are matched by
multivariate DTW, pixel analogue loss = mean target of the k nearest reference fields.
Part 2 (borrowed from runner-up c_r4a, code copied to ridge4.py): mean of four clipped ridge maps.
Pixel loss = (1 - w) x part 1 + w x part 2, w = ridge_weight (nanmean where one part has no value).

c_r4b docstring follows.

Main idea: nearest-neighbour analogues.  Every training field is kept as one reference trajectory: its field-mean
log reflectance in all 10 bands on a day-of-year grid.  Each outline pixel of a new field gets its own log
reflectance trajectory on the same grid and is matched to every reference trajectory by multivariate dynamic time
warping (shared warping path for all bands, Sakoe-Chiba band of +-W grid steps, so seasons that run earlier or later
can still match).  Pixel loss = mean label target of the k reference fields with the smallest warped distance.
No weights or regression on the profile: the labels enter only as the values carried by the analogues.
"""
from __future__ import annotations

from pathlib import Path

import importlib

import numpy as np
import yaml

from methods.c_r7a import ridge4 as R4

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
    """(2, G, 10) log of the field-mean (c_r4b, used for the per-band scale only) and of the field-median
    (reference trajectory) reflectance per usable date, interpolated."""
    o = f.outline
    r = f.refl[:, :, o]
    clr = np.isfinite(r[:, 6]).mean(1)
    use = f.usable & (clr >= CFG["usable_share"])
    doy = ((f.dates - f.dates.astype("datetime64[Y]")).astype(int) + 1.0)[use]
    m = np.nanmean(r[use].astype("float64"), 2) if use.any() else np.zeros((0, r.shape[1]))
    m = np.log(np.maximum(m, CFG["refl_floor"]))
    md = np.nanmedian(r[use].astype("float64"), 2) if use.any() else np.zeros((0, r.shape[1]))
    md = np.log(np.maximum(md, CFG["refl_floor"]))
    c = np.stack([_interp(m, doy, _grid()), _interp(md, doy, _grid())])  # (2, G, 10)
    f.drop()
    return c


# --------------------------------------------------------------------------------------------- DTW --
def dtw(A, B, W):
    """A (n, G, c), B (m, G, c) -> (n, m) multivariate DTW cost (squared Euclidean, shared path, band W).
    Same result as c_r4b's dtw; only cells inside the band are stored (speed)."""
    n, G, _ = A.shape
    m = B.shape[0]
    a2 = (A ** 2).sum(-1)                                                # (n, G)
    b2 = (B ** 2).sum(-1)                                                # (m, G)
    R0 = {0: np.zeros((n, m), "float32")}
    for i in range(1, G + 1):
        R1 = {}
        ai = A[:, i - 1]
        for j in range(max(1, i - W), min(G, i + W) + 1):
            c = a2[:, i - 1, None] + b2[None, :, j - 1] - 2.0 * (ai @ B[:, j - 1].T)
            np.maximum(c, 0, out=c)
            if CFG["cost_power"] != 1:                                   # c_r7a: cell cost = (squared dist) ** p
                c = c ** CFG["cost_power"]
            best = None
            for prev in (R0.get(j), R1.get(j - 1), R0.get(j - 1)):
                if prev is not None:
                    best = prev if best is None else np.minimum(best, prev)
            R1[j] = c + best
        R0 = R1
    return R0[G]


# --------------------------------------------------------------------------------------------- fit --
def fit(fields, labels):
    import multiprocessing as mp
    lab = labels.set_index("field_id").loc[[f.field_id for f in fields]]
    with mp.get_context("fork").Pool(8) as p:
        both = np.array(p.map(field_curve, fields))                      # (n, 2, G, 10)
    keep = np.isfinite(both).all((1, 2, 3))
    both = both[keep]
    fmean, ref = both[:, 0], both[:, {"mean": 0, "median": 1}[CFG["reference"]]]                               # field-mean (scale), field-median (reference)
    lab = lab[keep]
    claim = lab.claim.to_numpy(bool)
    loss = lab.loss.to_numpy(float)
    y = np.where(claim, np.where(np.isfinite(loss), loss, CFG["heavy_nan_target"]), CFG["none_target"])
    sd = fmean.reshape(-1, fmean.shape[-1]).std(0) + 1e-9                # per-band scale (c_r4b: field-mean curves)
    STATE.update(ref=(ref / sd).astype("float32"), y=y.astype("float64"), sd=sd)
    importlib.reload(R4)                                                 # part 2: c_r4a ridges, fresh state
    R4.fit(fields, labels)


def predict(f):
    a, atyp = predict_analogue(f)
    if CFG["ridge_weight"] == 0:
        out = a
    else:
        b = R4.predict(f)
        w = CFG["ridge_weight"]
        out = np.where(np.isfinite(a) & np.isfinite(b), (1 - w) * a + w * b, np.where(np.isfinite(a), a, b))
    out[~f.outline] = np.nan
    # c_r7a: atypical pixels (log-reflectance trajectory far from the field's median trajectory) get the mean of the
    # field's typical pixels
    if atyp.any():
        typ = f.outline & ~atyp & np.isfinite(out)
        if typ.any():
            out[atyp] = np.nanmean(out[typ])
    return out.astype("float32")


def _atypical(pc, ok, o):
    """(H, W) bool: outline pixels whose (scaled) log trajectory is farther from the field-median trajectory than
    the field's atypical_quantile of that distance (plain Euclidean on the grid, no warping)."""
    at = np.zeros(o.shape, bool)
    q = CFG["atypical_quantile"]
    if q >= 1 or ok.sum() < 5:
        return at
    x = pc[ok] / STATE["sd"]
    med = np.median(x, 0)
    d = np.sqrt(((x - med) ** 2).sum((1, 2)))
    flag = np.zeros(len(pc), bool)
    flag[np.flatnonzero(ok)] = d > np.quantile(d, q)
    at[o] = flag
    return at


def predict_analogue(f):
    if "ref" not in STATE:
        raise RuntimeError("fit first")
    o = f.outline
    pc = pixel_curves(f)                                                 # (P, G, 10)
    ok = np.isfinite(pc).all((1, 2))
    atyp = _atypical(pc, ok, o)
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
    return out, atyp
