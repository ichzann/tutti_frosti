"""Baseline method: per-pixel EVI2 GPR expected trajectory -> z-score damage.

Plugin interface (see README):
    m = Method(cfg, store)            # store: data.store.FieldStore
    m.damage(field_id, night) -> {"score", "valid", "meta", "curve", "layers"}
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
        # fit cache: the fit uses all clear obs before the night and predicts at the first
        # clear obs after it, so it depends only on (pixel, last-before idx, first-after idx);
        # nights between the same two acquisitions share it.  Times are absolute days, so a
        # stationary kernel gives the same result as times relative to the night.
        self._fit = {}
        s0 = dt.date.fromisoformat(cfg["study"]["season_start"])
        e0 = dt.date.fromisoformat(cfg["study"]["season_end"])
        self.day0 = s0
        self.curve_days = np.arange(0, (e0 - s0).days + 31, 2, dtype=float)

    def clear_cache(self):
        self._fit.clear()

    def _fit_pixel(self, field_id, y, x, tb, ta, ad, yb):
        key = (field_id, y, x, int(tb[-1]), int(ta[0]))
        if key not in self._fit:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp = GaussianProcessRegressor(_kernel(self.cfg), normalize_y=True,
                                              n_restarts_optimizer=0)
                gp.fit(ad[tb][:, None], yb)
                m, s = gp.predict(np.array([[ad[ta[0]]]]), return_std=True)
                cm, cs = gp.predict(self.curve_days[:, None], return_std=True)
            self._fit[key] = (m[0], s[0], cm.astype("float32"), cs.astype("float32"))
        return self._fit[key]

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
        abs_days = np.array([(d - self.day0).days for d in dates], dtype=float)
        curves_m, curves_s = [], []
        # observations strictly before the night; the night's evening is day 0 so
        # acquisitions on the night date (late morning) count as before
        before = t_days <= 0
        after = t_days > 0
        since_em = np.array([emerg is not None and d >= emerg for d in dates])
        win = g.get("obs_window_days")
        in_window = before & (t_days >= -win if win is not None else True)

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
            if ((c & before & since_em).sum() < g["min_clear_obs_since_emergence"]
                    or (c & in_window).sum() < g.get("min_clear_obs_before", 0)):
                reason[y, x] = 3; continue
            if -t_days[tb[-1]] > g["max_gap_before_days"]:
                reason[y, x] = 4; continue
            if not len(ta) or t_days[ta[0]] > g["max_gap_after_days"]:
                reason[y, x] = 5; continue
            try:
                m, s, cm, cs = self._fit_pixel(field_id, y, x, tb, ta, abs_days, ev[tb, y, x])
            except Exception:
                reason[y, x] = 6; continue
            o = ev[ta[0], y, x]
            zz = (o - m) / s if s > 0 else np.nan
            if not np.isfinite(zz):
                reason[y, x] = 6; continue
            z[y, x], pred[y, x], pstd[y, x], obs[y, x] = zz, m, s, o
            curves_m.append(cm); curves_s.append(cs)
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
        # field-level prediction band: median over valid pixels of the per-pixel GPR mean / std
        if curves_m:
            curve = dict(days=self.curve_days, mean=np.median(curves_m, 0), std=np.median(curves_s, 0))
        else:
            curve = None
        return {"score": score, "valid": valid, "meta": meta, "curve": curve,
                "layers": {"z": z, "predicted": pred, "pred_std": pstd, "observed": obs,
                           "reason": reason, "pre_idx": pi, "post_idx": qi, "next_idx": nxt}}
