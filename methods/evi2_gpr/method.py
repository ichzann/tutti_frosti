"""Baseline method: per-pixel EVI2 GPR expected trajectory -> z-score damage.

Variants (config `gpr.hyper`, `gpr.mean`; baseline = pixel + constant):
  hyper  pixel: kernel hyperparameters fitted per pixel (sklearn, normalize_y)
         field: one set per field and acquisition interval (joint marginal likelihood over a pixel
                sample), then exact GP conditioning per pixel with them fixed
  mean   constant: GP extrapolated from the last obs before the night to the first obs after
         reference: GP smooths the pixel to its last obs before the night, then adds the EVI2 change
                over the gap expected from frost-free intervals of other fields (`GrowthReference`)

Plugin interface (see README):
    m = Method(cfg, store)            # store: data.store.FieldStore
    m.damage(field_id, night) -> {"score", "valid", "meta", "curve", "layers"}
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neighbors import NearestNeighbors

from data.store import FieldStore
from validation.cases import is_holdout

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


# ------------------------------------------------ shared-hyperparameter GP (gpr.hyper: field) --
# amplitude (EVI2^2), length scale (days), noise (EVI2^2); same ranges as _kernel, but on raw EVI2
HYPER_BOUNDS = np.log([[1e-5, 1.0], [5.0, 200.0], [1e-6, 1e-1]])


def _rbf(a, b, amp, ls):
    return amp * np.exp(-0.5 * ((a[:, None] - b[None, :]) / ls) ** 2)


def _chol(t, amp, ls, noise):
    return np.linalg.cholesky(_rbf(t, t, amp, ls) + (noise + 1e-9) * np.eye(len(t)))


def _neg_lml(logp, groups):
    """Summed negative log marginal likelihood of pixel series centred on their own mean."""
    amp, ls, noise = np.exp(logp)
    tot = 0.0
    for t, yc in groups:
        try:
            low = _chol(t, amp, ls, noise)
        except np.linalg.LinAlgError:
            return 1e12
        a = np.linalg.solve(low, yc)
        tot += 0.5 * (a ** 2).sum() + yc.shape[1] * np.log(np.diag(low)).sum()
    return tot


def _pattern_groups(cb, ys, xs):
    """Pixels grouped by their set of clear acquisitions: {tuple of date indices: pixel positions}."""
    out = {}
    for k, row in enumerate(cb[:, ys, xs].T):
        out.setdefault(tuple(np.nonzero(row)[0]), []).append(k)
    return {key: np.array(v) for key, v in out.items()}


def fit_hyper(groups, ls0):
    """groups: [(t (n,), centred y (n, m))] -> (amp, ls, noise) maximising the joint likelihood."""
    best = None
    for x0 in ([1e-2, ls0, 1e-3], [1e-2, 2 * ls0, 1e-3]):
        r = minimize(_neg_lml, np.log(x0), args=(groups,), method="L-BFGS-B", bounds=HYPER_BOUNDS)
        if best is None or r.fun < best.fun:
            best = r
    return tuple(np.exp(best.x))


# ------------------------------------------------------ growth reference (gpr.mean: reference) --
REF_FEATURES = ["e0", "slope", "gap", "doy"]


def _fold(field_id, n):
    return int(hashlib.md5(f"ref:{field_id}".encode()).hexdigest(), 16) % n


def field_median_series(f, min_share):
    """Field-median EVI2 per date, NaN where < min_share of the field's pixels are clear."""
    m = f["mask"]
    med = np.full(len(f["dates"]), np.nan)
    for i, e in enumerate(f["evi2"]):
        v = e[m]
        if np.isfinite(v).mean() >= min_share:
            med[i] = np.nanmedian(v)
    return med


def field_slope(dates, med, i):
    """EVI2 per day from the previous usable date to date i (field median); 0 if none."""
    prev = [j for j in range(i) if np.isfinite(med[j])]
    if not prev or not np.isfinite(med[i]):
        return 0.0
    j = prev[-1]
    return float((med[i] - med[j]) / (dates[i] - dates[j]).days)


def reference_table(cfg, store):
    """Frost-free field-median EVI2 changes between usable acquisitions (non-lockbox fields).

    One row per (field, acquisition k, later acquisition k + j), j <= max_step, gap <= max_gap_days,
    and on every night [date_k, date_k+j - 1] in the field's RTMA cell: no trigger and Tmin >= min_tmin_c."""
    rc = cfg["reference"]
    n = store.nights
    cold = n.trigger.copy()
    if rc.get("min_tmin_c") is not None:
        cold |= n.tmin_c < rc["min_tmin_c"]
    trig = set(zip(n.cell_id[cold], n.night[cold]))
    rows = []
    for fid in store.field_ids():
        if is_holdout(fid, rc["lockbox_share"]):
            continue
        f = store.field(fid)
        med = field_median_series(f, rc["min_clear_share"])
        use = np.nonzero(np.isfinite(med))[0]
        d = [f["dates"][i] for i in use]
        cell = store.field_cell.get(fid)
        for k in range(1, len(use) - 1):
            slope = field_slope(f["dates"], med, use[k])
            for j in range(1, rc["max_step"] + 1):
                if k + j >= len(use):
                    break
                gap = (d[k + j] - d[k]).days
                if gap > rc["max_gap_days"] or any(
                        (cell, (d[k] + dt.timedelta(days=i)).isoformat()) in trig for i in range(gap)):
                    break        # a longer pair would contain the same gap / trigger night
                rows.append(dict(field_id=fid, date=d[k].isoformat(), e0=med[use[k]], slope=slope, gap=gap,
                                 doy=d[k].timetuple().tm_yday, delta=med[use[k + j]] - med[use[k]]))
    return pd.DataFrame(rows)


class GrowthReference:
    """Expected EVI2 change over a gap: k nearest frost-free intervals in (e0, slope, gap, doy),
    standardised. Leave-fold-out: a field is predicted only from intervals of other field folds."""

    def __init__(self, table, rc):
        self.table, self.folds, k = table, rc["folds"], rc["k_neighbors"]
        fold = table.field_id.map(lambda i: _fold(i, self.folds)).to_numpy()
        X, y = table[REF_FEATURES].to_numpy(float), table.delta.to_numpy(float)
        self.models = {}
        for q in range(self.folds):
            tr = fold != q
            mu, sd = X[tr].mean(0), X[tr].std(0)
            self.models[q] = (mu, sd, NearestNeighbors(n_neighbors=k).fit((X[tr] - mu) / sd), y[tr])

    def predict(self, field_id, X):
        """X (n, 4) -> mean, variance of the expected change (n,)."""
        mu, sd, nn, y = self.models[_fold(field_id, self.folds)]
        _, idx = nn.kneighbors((np.asarray(X, float) - mu) / sd)
        v = y[idx]
        return v.mean(1), v.var(1, ddof=1)


_REF = {}


def growth_reference(cfg, store):
    """Built once per process and config, always from the unmodified store (never an injected copy)."""
    key = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    if key not in _REF:
        if not isinstance(store, FieldStore):
            raise RuntimeError("growth reference must first be built from a FieldStore "
                               "(construct Method on the unmodified store before any wrapped one)")
        _REF[key] = GrowthReference(reference_table(cfg, store), cfg["reference"])
    return _REF[key]


class Method:
    name = NAME

    def __init__(self, cfg, store):
        self.cfg, self.store = cfg, store
        self.score_threshold = -cfg["damage"]["z_damaged"]   # damaged if score >= this (z <= z_damaged)
        self._gp = {}
        # fit cache: the fit uses all clear obs before the night and predicts at the first
        # clear obs after it, so it depends only on (pixel, last-before idx, first-after idx);
        # nights between the same two acquisitions share it.  Times are absolute days, so a
        # stationary kernel gives the same result as times relative to the night.
        self._fit = {}
        g = cfg["gpr"]
        self.hyper, self.mean = g.get("hyper", "pixel"), g.get("mean", "constant")
        self._hyp = {}                                         # (field, n dates <= night) -> hyperparameters
        self.ref = growth_reference(cfg, store) if self.mean == "reference" else None
        s0 = dt.date.fromisoformat(cfg["study"]["season_start"])
        e0 = dt.date.fromisoformat(cfg["study"]["season_end"])
        self.day0 = s0
        self.curve_days = np.arange(0, (e0 - s0).days + 31, 2, dtype=float)

    def clear_cache(self):
        self._fit.clear()
        self._hyp.clear()

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

    def _hyper_params(self, field_id, cb, ev, abs_days, nb):
        """Field hyperparameters from all clear obs before the night of a pixel sample; they depend
        only on which acquisitions lie before the night (nb), so nights in one interval share them."""
        key = (field_id, nb)
        if key not in self._hyp:
            ys, xs = np.nonzero(cb.sum(0) >= 3)
            hp = None
            if len(ys):
                rng = np.random.default_rng(int(hashlib.md5(str(field_id).encode()).hexdigest()[:8], 16))
                pick = rng.choice(len(ys), min(self.cfg["gpr"]["hyper_sample_px"], len(ys)), replace=False)
                ys, xs = ys[pick], xs[pick]
                groups = []
                for idx, pos in _pattern_groups(cb, ys, xs).items():
                    yv = ev[list(idx)][:, ys[pos], xs[pos]]
                    groups.append((abs_days[list(idx)], yv - yv.mean(0)))
                hp = fit_hyper(groups, self.cfg["gpr"].get("length_scale_init_days", 20.0))
            self._hyp[key] = hp
        return self._hyp[key]

    def _predict_shared(self, field_id, f, ev, cb, post_idx, reason, abs_days, nb,
                        z, pred, pstd, obs, score):
        """Joint prediction for all candidate pixels (reason 0) with field hyperparameters.
        Fills z / pred / pstd / obs / score in place, sets reason 6 on failure; returns the
        per-pixel curves (mean, std) for the field-level band."""
        ys, xs = np.nonzero(reason == 0)
        if not len(ys):
            return [], []
        hp = self._hyper_params(field_id, cb, ev, abs_days, nb)
        if hp is None:
            reason[ys, xs] = 6
            return [], []
        amp, ls, noise = hp
        med = field_median_series(f, self.cfg["reference"]["min_clear_share"]) if self.ref else None
        cd = self.curve_days
        curves_m, curves_s = [], []
        # pixels sharing (clear obs before, first obs after) share the GP algebra
        key = np.concatenate([cb[:, ys, xs].T.astype(int), post_idx[ys, xs][:, None]], 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        for gi, k in enumerate(uniq):
            pos = np.nonzero(inv == gi)[0]
            gy, gx = ys[pos], xs[pos]
            idx, ia = np.nonzero(k[:-1])[0], int(k[-1])
            t, yv = abs_days[idx], ev[idx][:, gy, gx]
            mu = yv.mean(0)
            try:
                low = _chol(t, amp, ls, noise)
            except np.linalg.LinAlgError:
                reason[gy, gx] = 6
                continue
            alpha = cho_solve((low, True), yv - mu)

            def cond(tq):
                """latent GP mean (q, m) and variance (q,) at times tq"""
                kq = _rbf(np.atleast_1d(np.asarray(tq, float)), t, amp, ls)
                v = amp - np.einsum("qn,nq->q", kq, cho_solve((low, True), kq.T))
                return mu + kq @ alpha, np.maximum(v, 0.0)

            cm, cv = cond(cd)
            tl, ta = t[-1], abs_days[ia]
            if self.ref is None:          # extrapolate the GP to the first obs after
                m, v = cond(ta)
                m, v = m[0], np.full(len(gy), v[0] + noise)
                cmean, cstd = cm, np.broadcast_to(np.sqrt(cv + noise)[:, None], cm.shape)
            else:                          # smoothed level at the last obs before + expected change
                s, vs = cond(tl)
                s, vs = s[0], vs[0]
                il = idx[-1]
                X = np.column_stack([s, np.full_like(s, field_slope(f["dates"], med, il)),
                                     np.full_like(s, ta - tl),
                                     np.full_like(s, f["dates"][il].timetuple().tm_yday)])
                d, vd = self.ref.predict(field_id, X)
                m, v = s + d, vs + vd + noise
                frac = np.clip((cd - tl) / (ta - tl), 0, 1)[:, None]
                ahead = (cd > tl)[:, None]
                cmean = np.where(ahead, s + frac * d, cm)
                cstd = np.sqrt(np.where(ahead, vs + frac * vd, cv[:, None]) + noise)
            o = ev[ia, gy, gx]
            zz = (o - m) / np.sqrt(v)
            ok = np.isfinite(zz)
            reason[gy[~ok], gx[~ok]] = 6
            gy, gx = gy[ok], gx[ok]
            z[gy, gx], pred[gy, gx], pstd[gy, gx], obs[gy, gx] = zz[ok], m[ok], np.sqrt(v)[ok], o[ok]
            score[gy, gx] = np.maximum(0.0, -zz[ok])
            curves_m += list(cmean[:, ok].T.astype("float32"))
            curves_s += list(cstd[:, ok].T.astype("float32"))
        return curves_m, curves_s

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
            if self.hyper != "pixel" or self.mean != "constant":
                pre_idx[y, x], post_idx[y, x] = tb[-1], ta[0]
                continue                            # predicted jointly below
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

        if self.hyper != "pixel" or self.mean != "constant":
            curves_m, curves_s = self._predict_shared(
                field_id, f, ev, clear & before[:, None, None], post_idx, reason, abs_days,
                int(before.sum()), z, pred, pstd, obs, score)
            pre_idx[reason != 0] = -1
            post_idx[reason != 0] = -1

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
