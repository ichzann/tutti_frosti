"""Module D: fusion of the Module B prior with the Module C evidence (c_mt_full), and the stand-alone
evidence map (c_mt_core).  Doc section 3.5.  Per-fold beta / b are appended to methods/c_mt/report/fit_<variant>.jsonl.
"""
from __future__ import annotations

import json
import time

import numpy as np

from methods.c_mt import clock as K
from methods.c_mt import common as C
from methods.c_mt import evidence as E
from methods.c_mt import prior as P

STATE = {}
B1_GRID = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
B2_GRID = [0, 0.1, 0.2, 0.3, 0.5]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 0.02, 0.98)
    return np.log(p / (1 - p))


def _evidence_stats(fields):
    raw = [E.load_field(f) for f in fields]
    for f in fields:
        f.drop()
    Dm = np.array([np.nanmean(r["true"]) if np.isfinite(r["true"]).any() else np.nan for r in raw])
    kappa = float(np.nanmedian(Dm[Dm > 0]))
    with np.errstate(all="ignore"):
        le = np.array([np.nanmean(np.log1p(r["true"] / kappa)) if np.isfinite(r["true"]).any() else np.nan
                       for r in raw])
    X = np.array([float(r["X"]) for r in raw])
    st = dict(kappa=kappa, mE=float(np.nanmean(le)), sdE=float(np.nanstd(le)),
              mX=float(np.nanmean(X)), sdX=float(np.nanstd(X)))
    Ebar = np.nan_to_num((le - st["mE"]) / st["sdE"], nan=0.0)
    Xt = np.nan_to_num((X - st["mX"]) / st["sdX"], nan=0.0)
    return st, Ebar, Xt


def _targets(lab, cfg):
    claim = lab.claim.to_numpy(bool)
    loss = lab.loss.to_numpy(float)
    known = np.isfinite(loss)
    y = np.where(claim, np.where(known, loss, cfg["heavy_nan_target"]), cfg["none_target"])
    return y, claim, np.where(claim & known, loss, -1.0), known


def _log(variant, rec):
    with open(C.HERE / "report" / f"fit_{variant}.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def fit(fields, labels, cfg, variant):
    t0 = time.time()
    STATE.clear()
    STATE["cfg"] = cfg
    lab = labels.set_index("field_id").loc[[f.field_id for f in fields]].reset_index()
    held = sorted(set(K.load()["Y"]) - set(lab.year.unique()))
    st, Ebar, Xt = _evidence_stats(fields)
    STATE["st"] = st
    y, claim, lossc, known = _targets(lab, cfg)
    rec = dict(variant=variant, held_out=[int(h) for h in held], n_train=len(lab), **st)
    if cfg["mode"] == "core":
        Xm = np.c_[np.ones(len(y)), Ebar, Xt]
        use = [0, 1, 2]
        while True:
            b = np.zeros(3)
            b[use] = np.linalg.lstsq(Xm[:, use], y, rcond=None)[0]
            neg = [j for j in use if j > 0 and b[j] < 0]
            if not neg:
                break
            use = [j for j in use if j not in neg]
        STATE["b"] = b
        rec["b"] = b.tolist()
    else:
        P.set_delta(K.delta_table(cfg["time_axis"]), cfg["time_axis"])
        P.fit(fields, labels)
        try:
            oof = P.oof_field()
            A = np.array([oof.get(fid, np.nan) for fid in lab.field_id])
            ok = np.isfinite(A)
            from methods.c_mt.ridge4 import _cindex
            best, Cs = None, {}
            for b1 in B1_GRID:
                for b2 in B2_GRID:
                    s = _sig(_logit(A[ok]) + b1 * Ebar[ok] + b2 * Xt[ok])
                    Cs[(b1, b2)] = _cindex(s, claim[ok], lossc[ok], known[ok])
            cmax = max(Cs.values())
            beta = min((g for g in Cs if Cs[g] >= cmax - 0.002), key=lambda g: (sum(g), g))
            rec.update(beta=list(beta), beta_source="fit", oof_C_best=cmax, oof_C_beta0=Cs[(0, 0)],
                       oof_C_chosen=Cs[beta], n_oof=int(ok.sum()))
        except Exception as ex:                                       # noqa: BLE001 - pre-declared fallback
            beta = (0.25, 0.0)
            rec.update(beta=list(beta), beta_source=f"fallback ({type(ex).__name__}: {ex})")
        STATE["beta"] = beta
    rec["seconds"] = round(time.time() - t0, 1)
    _log(variant, rec)


def _maps(f):
    st = STATE["st"]
    r = E.load_field(f)
    with np.errstate(all="ignore"):
        Ep = (np.log1p(r["true"] / st["kappa"]) - st["mE"]) / st["sdE"]
    X = float(r["X"])
    Xt = (X - st["mX"]) / st["sdX"] if np.isfinite(X) else 0.0
    return Ep, Xt


def predict(f):
    o = f.outline
    Ep, Xt = _maps(f)
    out = np.full(o.shape, np.nan, dtype="float32")
    if STATE["cfg"]["mode"] == "core":
        b = STATE["b"]
        v = np.clip(b[0] + b[1] * Ep + b[2] * Xt, 0, 1)             # NaN where E_p is NaN
        out[o] = v
        return out
    A = P.predict(f)
    b1, b2 = STATE["beta"]
    a = A[o].astype("float64")
    L = _sig(_logit(a) + b1 * Ep + b2 * Xt)
    out[o] = np.where(np.isfinite(Ep), L, a)                     # E_p NaN -> prior; prior NaN -> NaN
    return out
