"""c_r1b: crop-calendar position.  Each pixel's season centre (day of year) from its red-edge chlorophyll curve;
earlier season centre -> higher loss, through a monotone curve learned from development labels.

Per pixel (CIre = B07 / B05 - 1, clear pixel values on usable dates only, needs >= min_obs clear dates):
  peak = date of the season maximum; green-up date = interpolated crossing of lo_before + frac (peak - lo_before)
  before the peak; senescence date = crossing of lo_after + frac (peak - lo_after) after the peak;
  centre = (green-up + senescence) / 2, day of year (year is used as calendar only).
Pixel loss = hi / (1 + exp((centre - m) / s)); (m, s, hi) fitted on development fields: field score (mean of the
pixel map) vs label with censored squared error (paid: (score - loss)^2; no-claim: max(score - 0.35, 0)^2;
paid without a loss value: min(score - 0.35, 0)^2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "settings.yaml").read_text())
INDEX = {"cire": lambda f: f.band("B07") / f.band("B05") - 1.0,
         "ndre": lambda f: (f.band("B8A") - f.band("B05")) / (f.band("B8A") + f.band("B05")),
         "evi2": lambda f: 2.5 * (f.band("B08") - f.band("B04")) / (f.band("B08") + 2.4 * f.band("B04") + 1.0),
         "ndmi": lambda f: (f.band("B08") - f.band("B11")) / (f.band("B08") + f.band("B11"))}
CACHE = HERE / "_cache"
PARAMS = dict(CFG["default_params"])          # replaced by fit()


def _cross_down(d, e, i, thr):
    for j in range(i, len(e) - 1):
        if e[j + 1] < thr <= e[j]:
            return d[j] + (e[j] - thr) / (e[j] - e[j + 1]) * (d[j + 1] - d[j])
    return np.nan


def _cross_up(d, e, i, thr):
    for j in range(i, 0, -1):
        if e[j - 1] < thr <= e[j]:
            return d[j - 1] + (thr - e[j - 1]) / (e[j] - e[j - 1]) * (d[j] - d[j - 1])
    return np.nan


def centre_map(f) -> np.ndarray:
    """(H, W) season-centre day of year per outline pixel (NaN = no estimate); cached on disk per field."""
    key = f"{f.field_id}_{CFG['index']}_{CFG['frac']}_1_{CFG['min_obs']}.npy"
    p = CACHE / key
    if p.exists():
        return np.load(p)
    use = np.asarray(f.usable, bool)
    o = f.outline
    out = np.full(o.shape, np.nan, "float32")
    if use.sum() >= CFG["min_obs"]:
        with np.errstate(all="ignore"):
            x = INDEX[CFG["index"]](f)[use]
        doy = (f.dates[use] - np.datetime64(f"{f.year}-01-01")).astype(int).astype("float64")
        xs = x[:, o]                                            # (T, N)
        res = np.full(xs.shape[1], np.nan)
        fr = CFG["frac"]
        for n in range(xs.shape[1]):
            v = xs[:, n]
            ok = np.isfinite(v)
            if ok.sum() < CFG["min_obs"]:
                continue
            d, e = doy[ok], v[ok]
            i = int(np.argmax(e))
            pk = e[i]
            lb, la = e[:i + 1].min(), e[i:].min()
            u = _cross_up(d, e, i, lb + fr * (pk - lb))
            s = _cross_down(d, e, i, la + fr * (pk - la))
            res[n] = 0.5 * (u + s)
        out[o] = res
    CACHE.mkdir(exist_ok=True)
    np.save(p, out)
    return out


def _g(c, m, s, hi):
    with np.errstate(all="ignore"):
        return hi / (1.0 + np.exp((c - m) / s))


def fit(fields, labels):
    lab = labels.set_index("field_id")
    cs, loss, claim = [], [], []
    for f in fields:
        c = centre_map(f)
        v = c[f.outline]
        v = v[np.isfinite(v)]
        f.drop()
        if len(v) == 0:
            continue
        cs.append(v)
        loss.append(lab.loc[f.field_id, "loss"])
        claim.append(bool(lab.loc[f.field_id, "claim"]))
    loss, claim = np.array(loss, float), np.array(claim)

    def obj(p):
        m, ls, lhi = p
        s, hi = np.exp(ls), 1.0 / (1.0 + np.exp(-lhi))
        sc = np.array([_g(v, m, s, hi).mean() for v in cs])
        err = np.where(~claim, np.maximum(sc - CFG["none_cap"], 0.0),            # no claim: loss < 0.35
                       np.where(np.isfinite(loss), sc - np.nan_to_num(loss),       # paid with loss value
                                np.minimum(sc - CFG["none_cap"], 0.0)))            # paid, loss missing: >= 0.35
        return float((err ** 2).mean())

    d = CFG["default_params"]
    x0 = [d["m"], np.log(d["s"]), np.log(d["hi"] / (1 - d["hi"]))]
    r = minimize(obj, x0, method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-3, fatol=1e-7))
    m, ls, lhi = r.x
    PARAMS.update(m=float(m), s=float(np.exp(ls)), hi=float(1.0 / (1.0 + np.exp(-lhi))))


def predict(f):
    c = centre_map(f)
    y = _g(c, PARAMS["m"], PARAMS["s"], PARAMS["hi"]).astype("float32")
    return np.where(f.outline & np.isfinite(c), y, np.float32(np.nan))
