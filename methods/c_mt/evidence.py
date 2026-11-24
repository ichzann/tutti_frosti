"""Module C: stage-locked damage evidence per pixel (label-free) and field exposure.

    python -m methods.c_mt.evidence        # cache raw D_p (all switches) + X_k for the clock (dev) fields

Switch configs (doc 4.3, F1-F5 change the method only through these):
  true     : leave-self-out omega_hat, theory S, persistence, composite z (NDMI, RE, VIS)
  shift_m35 / shift_p35 : omega_hat moved 35 d earlier / later (F1 placebo clock)
  flat_clock : omega_hat == 1 (F2)
  flat_S     : S == 1 (F3)
  no_pi      : pi == 1 (F4)
  ndmi_only  : composite z = NDMI z only (F5; clock unchanged)
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from methods.c_mt import clock as K
from methods.c_mt import common as C

CFG = C.CFG
CONFIGS = ["true", "shift_m35", "shift_p35", "flat_clock", "flat_S", "no_pi", "ndmi_only"]
_CK = {}


def _clock():
    if "c" not in _CK:
        _CK["c"] = K.load()
    return _CK["c"]


def _shift(w, d):
    """omega moved by d days (d > 0: later), zero-filled."""
    out = np.zeros_like(w)
    if d > 0:
        out[d:] = w[:-d]
    else:
        out[:d] = w[-d:]
    return out


def exposure(H, w):
    if not np.isfinite(H) or w.sum() <= 0:
        return np.nan
    return float((w * C.S_curve(K.DAYS - H)).sum() / w.sum())


def field_evidence(f, configs=("true",)):
    """dict config -> D (P,) raw evidence per outline pixel (NaN: < 3 clear usable dates); plus 'X' exposure, 'H'."""
    ck = _clock()
    H = ck["H"].get(f.field_id)
    if H is None:                                     # not a clock field: own anchor from its own series
        doy, n, r, v, e = C.field_series(f)
        H = C.heading(doy, r)
    w = K.field_omega_hat(ck, f.field_id, f.year)     # leave-self-out for clock fields
    doy, n, r, v, e = C.pixel_series(f)
    P = n.shape[1]
    ok = np.isfinite(n) & np.isfinite(r) & np.isfinite(v) & np.isfinite(e)
    nclear = ok.sum(0)
    it = C.intervals(doy, n, r, v, e)
    m = it["canopy"]
    it = {k: x[m] for k, x in it.items()}
    s = (it["a"] + it["b"]) / 2 - H
    dtc = C.dt_class(it["dt"])
    D = np.stack([it["dN"], it["dR"], it["dV"]], 1)
    Dc = np.stack([it["dNc"], it["dRc"], it["dVc"]], 1)
    z = C.zscore(ck["Tpix"], s, dtc, D, Dc)
    S = C.S_curve(s)
    # omega_bar = max over d in (a, b] of the clock, per unique (a, b)
    ab = np.stack([it["a"], it["b"]], 1)
    uab, inv = np.unique(ab, axis=0, return_inverse=True) if len(ab) else (np.zeros((0, 2)), np.zeros(0, int))
    inv = np.asarray(inv).ravel()
    cov = K._cover(uab[:, 0], uab[:, 1]) if len(uab) else np.zeros((0, len(K.DAYS)), bool)

    def wbar(wc):
        if len(uab) == 0:
            return np.zeros(0)
        return np.where(cov, wc[None, :], 0.0).max(1)[inv]

    out = dict(H=np.float64(H), X=np.float64(exposure(H, w)))
    for cf in configs:
        wc = {"shift_m35": _shift(w, -35), "shift_p35": _shift(w, 35),
              "flat_clock": np.ones_like(w)}.get(cf, w)
        zz = z["zNc"] if cf == "ndmi_only" else z["z"]
        Sx = np.ones_like(S) if cf == "flat_S" else S
        pix_pi = np.ones_like(z["pi"]) if cf == "no_pi" else z["pi"]
        ev = wbar(wc) * Sx * np.maximum(0.0, -zz + CFG["evid_z"]) * pix_pi
        Dp = np.bincount(it["col"], weights=ev, minlength=P).astype("float64")
        Dp[nclear < 3] = np.nan
        out[cf] = Dp
    return out


def cache_file(fid):
    return C.CACHE / "evidence" / f"{fid}_{C.cfg_hash()}.npz"


def _build_one(f):
    p = cache_file(f.field_id)
    if not p.exists():
        d = field_evidence(f, CONFIGS)
        np.savez(p, **d)
    f.drop()
    return f.field_id


def load_field(f, configs=("true",)):
    p = cache_file(f.field_id)
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    return field_evidence(f, configs)


def build():
    (C.CACHE / "evidence").mkdir(parents=True, exist_ok=True)
    fields = C.clock_fields()
    with mp.get_context("fork").Pool(CFG["workers"]) as p:
        p.map(_build_one, fields, chunksize=1)


if __name__ == "__main__":
    build()
