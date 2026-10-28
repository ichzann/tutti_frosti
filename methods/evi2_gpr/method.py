"""Baseline method: per-pixel EVI2 GPR expected trajectory -> z-score damage.

Plugin interface (see README):
    m = Method(cfg, store)            # store: methods.base.FieldStore
    m.damage(field_id, night) -> {"score", "valid", "meta", "layers"}
"""
from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

NAME = "evi2_gpr"

# valid-mask reason codes (0 = valid)
REASONS = {0: "valid", 1: "outside field / CDL", 2: "outside growth phase",
           3: "too few clear obs before", 4: "last obs before too old",
           5: "no clear obs after in time", 6: "GPR failed"}


def running_median(x, w):
    out = np.full_like(x, np.nan)
    h = w // 2
    for k in range(len(x)):
        seg = x[max(0, k - h):k + h + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[k] = np.median(seg)
    return out


def growth_phase(dates, evi2, mask, cfg):
    """(emergence_date, peak_date, smoothed field-median series) for a field."""
    med = np.array([np.nanmedian(e[mask]) if np.isfinite(e[mask]).any() else np.nan
                    for e in evi2])
    ok = np.isfinite(med)
    d_ok = [d for d, o in zip(dates, ok) if o]
    sm = running_median(med[ok], cfg["growth"]["smoothing_window_obs"])
    series = (d_ok, med[ok], sm)
    if not len(sm):
        return None, None, series
    above = np.where(sm > cfg["growth"]["emergence_evi2"])[0]
    if not len(above):
        return None, None, series
    emerg = d_ok[above[0]]
    peak = d_ok[int(np.nanargmax(sm))]
    if peak < emerg:
        return emerg, None, series
    return emerg, peak, series


def _kernel(cfg):
    ls = cfg["gpr"].get("length_scale_init_days", 20.0)
    return ConstantKernel(0.05, (1e-4, 1.0)) * RBF(ls, (5.0, 200.0)) + WhiteKernel(1e-3, (1e-6, 1e-1))


class Method:
    name = NAME

    def __init__(self, cfg, store):
        self.cfg, self.store = cfg, store
        self._gp = {}

    def growth(self, field_id):
        if field_id not in self._gp:
            f = self.store.field(field_id)
            self._gp[field_id] = growth_phase(f["dates"], f["evi2"], f["mask"], self.cfg)
        return self._gp[field_id]

    def damage(self, field_id, night):
        """Evaluate one field on one night (night = date of the evening, str or date)."""
        cfg = self.cfg
        g = cfg["gpr"]
        night = dt.date.fromisoformat(str(night))
        f = self.store.field(field_id)
        dates, ev, mask = f["dates"], f["evi2"].astype("float64"), f["mask"]
        T, H, W = ev.shape
        emerg, peak, _ = self.growth(field_id)
        nmeta = self.store.night_meta(field_id, night)

        reason = np.where(mask, 0, 1).astype("uint8")
        in_phase = emerg is not None and peak is not None and emerg <= night <= peak
        stage = ("pre-emergence" if emerg is None or night < emerg else
                 "growth" if peak is None or night <= peak else "post-peak")
        if not in_phase:
            reason[mask] = 2

        clear = np.isfinite(ev) & mask[None]
        t_days = np.array([(d - night).days for d in dates], dtype=float)  # night = 0
        # observations strictly before the night; the night's evening is day 0 so
        # acquisitions on the night date (late morning) count as before
        before = t_days <= 0
        after = t_days > 0
        since_em = np.array([emerg is not None and d >= emerg for d in dates])

        score = np.full((H, W), np.nan, "float32")
        z = np.full((H, W), np.nan, "float32")
        pred = np.full((H, W), np.nan, "float32")
        pstd = np.full((H, W), np.nan, "float32")
        obs = np.full((H, W), np.nan, "float32")
        pre_idx = np.full((H, W), -1, int)
        post_idx = np.full((H, W), -1, int)

        for y, x in zip(*np.nonzero(reason == 0)):
            c = clear[:, y, x]
            tb = np.nonzero(c & before)[0]          # all clear obs before -> GPR fit
            ta = np.nonzero(c & after)[0]
            if (c & before & since_em).sum() < g["min_clear_obs_since_emergence"]:
                reason[y, x] = 3; continue
            if -t_days[tb[-1]] > g["max_gap_before_days"]:
                reason[y, x] = 4; continue
            if not len(ta) or t_days[ta[0]] > g["max_gap_after_days"]:
                reason[y, x] = 5; continue
            Xb = t_days[tb][:, None]
            yb = ev[tb, y, x]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gp = GaussianProcessRegressor(_kernel(cfg), normalize_y=True,
                                                  n_restarts_optimizer=0)
                    gp.fit(Xb, yb)
                    m, s = gp.predict(np.array([[t_days[ta[0]]]]), return_std=True)
            except Exception:
                reason[y, x] = 6; continue
            o = ev[ta[0], y, x]
            zz = (o - m[0]) / s[0] if s[0] > 0 else np.nan
            if not np.isfinite(zz):
                reason[y, x] = 6; continue
            z[y, x], pred[y, x], pstd[y, x], obs[y, x] = zz, m[0], s[0], o
            score[y, x] = max(0.0, -zz)
            pre_idx[y, x], post_idx[y, x] = tb[-1], ta[0]

        valid = reason == 0
        # representative dates = most common per-pixel pre/post acquisition
        def mode(a):
            v = a[valid]
            return int(np.bincount(v).argmax()) if v.size else None
        pi, qi = mode(pre_idx), mode(post_idx)
        if pi is None:  # still report the field-level nearest acquisitions
            cb = np.nonzero(before & clear.any((1, 2)))[0]
            ca = np.nonzero(after & clear.any((1, 2)))[0]
            pi = int(cb[-1]) if len(cb) else None
            qi = int(ca[0]) if len(ca) else None
        nxt = None
        if qi is not None:
            later = np.nonzero(clear.any((1, 2)) & (np.arange(T) > qi))[0]
            nxt = int(later[0]) if len(later) else None
        iso = lambda i: dates[i].isoformat() if i is not None else None
        days = lambda i: int(t_days[i]) if i is not None else None
        meta = dict(field_id=field_id, night=night.isoformat(), method=NAME,
                    pre_date=iso(pi), post_date=iso(qi), next_date=iso(nxt),
                    gap_days_before=-days(pi) if pi is not None else None,
                    gap_days_after=days(qi), next_days_after=days(nxt),
                    growth_stage=stage, emergence=emerg and emerg.isoformat(),
                    peak=peak and peak.isoformat(), **nmeta)
        return {"score": score, "valid": valid, "meta": meta,
                "layers": {"z": z, "predicted": pred, "pred_std": pstd, "observed": obs,
                           "reason": reason, "pre_idx": pi, "post_idx": qi, "next_idx": nxt}}
