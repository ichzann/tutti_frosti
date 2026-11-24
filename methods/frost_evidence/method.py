"""frost_evidence: satellite-first frost damage; weather and terrain only for attribution.

Damage evidence (the score; air temperature is never used in it):
  1. expected EVI2 without frost, as evi2_gpr with hyper field + mean reference: s = GP level of the pixel
     at its last clear obs before the night, plus the expected change over the gap from frost-free
     intervals of other fields
  2. persistence: residuals at up to `n_post` clear obs after the night within `persist_days`, combined
     by their GLS mean (they share the error of s): z_raw = mean residual / its std
  3. spatial support: z_raw averaged over a `smooth_px` window of valid pixels
  4. empirical calibration: z = Phi^-1(rank of the smoothed statistic among frost-free pixel-intervals of
     other fields, leave-fold-out, same stage and number of post obs); damaged if z <= z_damaged
Attribution (layers / meta only, never in the score):
  p_frost      probability that the leaf reached night.leaf_critical_c on this night: RTMA Tmin minus the
               median hourly-sampling miss and, on radiative nights, the leaf offset and a cold-air pooling
               term from the DEM, with a normal error of sigma_c
  terrain_rho  rank correlation inside the field between z and the pooling index (> 0: lower z in hollows)
  attribution  "no damage" / "damage, frost plausible" / "damage, frost implausible on this night"
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve
from scipy.stats import norm, spearmanr

from data import dem
from data.common import cache_dir
from data.store import FieldStore
from methods.evi2_gpr import method as base
from validation.cases import is_holdout

NAME = "frost_evidence"
REASONS = dict(base.REASONS)


def cold_pool_index(cfg, row, radius_m):
    """Pixel elevation minus the mean elevation within radius_m, on the field grid (m; < 0 = hollow)."""
    pad = int(np.ceil(radius_m / 10))
    with rasterio.open(dem.county_dem(cfg, row.GEOID)) as src:
        w = rasterio.windows.from_bounds(
            row.x0 - pad * 10, row.y0 - (row.ny + pad) * 10, row.x0 + (row.nx + pad) * 10, row.y0 + pad * 10,
            transform=src.transform).round_offsets().round_lengths()
        a = src.read(1, window=w, boundless=True, fill_value=np.nan).astype("float64")
    yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1]
    disk = ((yy ** 2 + xx ** 2) <= pad ** 2).astype(float)
    ok = np.isfinite(a)
    num = fftconvolve(np.where(ok, a, 0.0), disk, mode="same")
    den = fftconvolve(ok.astype(float), disk, mode="same")
    tpi = np.where(ok, a - num / np.maximum(den, 1.0), np.nan)
    return tpi[pad:pad + row.ny, pad:pad + row.nx].astype("float32")


def attribute(cfg, fe, nmeta, cp, mask, z, valid, z_damaged):
    """Weather + terrain plausibility of frost on one night; never changes the damage evidence.
    Returns (p_frost (H, W), meta dict: p_frost_max, p_frost_median, terrain_rho, share_damaged, attribution)."""
    ncfg = cfg["night"]
    leaf = nmeta["tmin_c"] - fe["hourly_miss_median_c"]
    if nmeta["regime"] == "radiative":
        pool = np.minimum(fe["pool_c_per_m"] * np.maximum(0.0, -np.nan_to_num(cp)), fe["pool_max_c"])
        leaf = leaf - ncfg["leaf_offset_radiative_c"] - pool
    p_frost = np.where(mask, norm.cdf((ncfg["leaf_critical_c"] - leaf) / fe["sigma_c"]), np.nan)
    share = float((valid & (z <= z_damaged)).sum() / valid.sum()) if valid.any() else np.nan
    rho = np.nan
    vv = valid & np.isfinite(cp)
    if vv.sum() >= 30 and np.ptp(cp[vv]) > 0:
        rho = float(spearmanr(z[vv], cp[vv])[0])
    pmax = float(np.nanmax(p_frost)) if mask.any() else np.nan
    attribution = ("no damage" if not (share >= fe["field_flag_share"]) else
                   "damage, frost plausible" if pmax >= fe["p_frost_min"] else
                   "damage, frost implausible on this night")
    return p_frost, dict(p_frost_max=pmax,
                         p_frost_median=float(np.nanmedian(p_frost[mask])) if mask.any() else np.nan,
                         terrain_rho=rho, share_damaged=share, attribution=attribution)


def _cfg_key(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------ calibration --
def calibration_intervals(m, store):
    """(field_id, night) of frost-free intervals of non-lockbox fields for empirical calibration.

    Night on a usable acquisition date d inside the field's growth phase [emergence, peak), with every
    night in [d, d + persist_days) in the field's RTMA cell free of triggers and >= the reference Tmin
    floor (the post obs used as evidence fall inside that window). Fields in a fixed hashed order,
    the first m.fe['cal_fields']."""
    fe, rc = m.fe, m.cfg["reference"]
    n = store.nights
    cold = n.trigger.copy()
    if rc.get("min_tmin_c") is not None:
        cold |= n.tmin_c < rc["min_tmin_c"]
    coldset = set(zip(n.cell_id[cold], n.night[cold]))
    md5 = lambda i: hashlib.md5(f"cal:{i}".encode()).hexdigest()
    ids = sorted((i for i in store.field_ids() if not is_holdout(i, rc["lockbox_share"])), key=md5)
    for fid in ids[:fe["cal_fields"]]:
        f = store.field(fid)
        emerg, peak, _ = m.growth(fid)
        if emerg is None or peak is None:
            continue
        med = base.field_median_series(f, rc["min_clear_share"])
        cell = store.field_cell.get(fid)
        for i in np.nonzero(np.isfinite(med))[0]:
            d0 = f["dates"][i]
            if not (emerg <= d0 < peak):
                continue
            if any((cell, (d0 + dt.timedelta(days=k)).isoformat()) in coldset for k in range(fe["persist_days"])):
                continue
            yield fid, d0
        m.clear_cache()


def calibration_table(m, store):
    """Smoothed statistic (z_smooth) of a pixel sample of every calibration interval."""
    rng = np.random.default_rng(0)
    parts = []
    for fid, d0 in calibration_intervals(m, store):
        r = m._evidence(fid, d0)
        v = r["valid"]
        if not v.any():
            continue
        k = np.nonzero(v.ravel())[0]
        k = rng.choice(k, min(m.fe["cal_px_per_case"], len(k)), replace=False)
        parts.append(pd.DataFrame(dict(
            field_id=fid, date=d0.isoformat(), z=r["z_smooth"].ravel()[k],
            stage=r["stage"].ravel()[k].astype("int8"), npost=r["npost"].ravel()[k].astype("int8"))))
    return pd.concat(parts, ignore_index=True)


class Calibrator:
    """Leave-fold-out empirical CDF of the smoothed statistic per (stage, number of post obs)."""

    def __init__(self, table, folds, min_n):
        self.folds = folds
        fold = table.field_id.map(lambda i: base._fold(i, folds)).to_numpy()
        self.ref = {}
        for q in range(folds):
            t = table[fold != q]
            for st, g in t.groupby("stage"):
                self.ref[(q, st, None)] = np.sort(g.z.to_numpy())
                for npo, h in g.groupby("npost"):
                    if len(h) >= min_n:
                        self.ref[(q, st, npo)] = np.sort(h.z.to_numpy())

    def __call__(self, field_id, z, stage, npost):
        q = base._fold(field_id, self.folds)
        out = np.full(z.shape, np.nan)
        for st in np.unique(stage[np.isfinite(z)]):
            for npo in np.unique(npost[np.isfinite(z) & (stage == st)]):
                sel = np.isfinite(z) & (stage == st) & (npost == npo)
                a = self.ref.get((q, int(st), int(npo)), self.ref.get((q, int(st), None)))
                if a is None:
                    continue
                p = (np.searchsorted(a, z[sel], side="right") + 0.5) / (len(a) + 1)
                out[sel] = norm.ppf(p)
        return out


_CAL = {}


def calibrator(m, store):
    """Built once per config from the unmodified store, cached in cache/ and per process."""
    key = _cfg_key(m.cfg)
    if key not in _CAL:
        fp = cache_dir(m.cfg) / f"frost_evidence_cal_{store.fips}_{key}.parquet"
        if not fp.exists():
            if not isinstance(store, FieldStore):
                raise RuntimeError("frost_evidence calibration must first be built from a FieldStore")
            tmp = fp.with_name(f"{fp.name}.{os.getpid()}.tmp")
            calibration_table(m, store).to_parquet(tmp)
            os.replace(tmp, fp)
        fe = m.fe
        _CAL[key] = Calibrator(pd.read_parquet(fp), m.cfg["reference"]["folds"], fe["cal_min_n"])
    return _CAL[key]


# ----------------------------------------------------------------------------------- method --
class Method(base.Method):
    name = NAME

    def __init__(self, cfg, store):
        cfg = copy.deepcopy(cfg)
        cfg["gpr"].update(hyper="field", mean="reference")     # the expected-growth model of evi2_gpr v3
        super().__init__(cfg, store)
        self.fe = cfg["frost_evidence"]
        self.score_threshold = -self.fe["z_damaged"]
        self._cp = {}
        self.cal = calibrator(self, store) if self.fe["calibrate"] else None

    def cold_pool(self, field_id):
        if field_id not in self._cp:
            row = self.store.fields[self.store.fields.field_id == field_id].iloc[0]
            self._cp[field_id] = cold_pool_index(self.cfg, row, self.fe["pool_radius_m"])
        return self._cp[field_id]

    def _evidence(self, field_id, night):
        """Uncalibrated satellite evidence for one field-night (see module docstring, steps 1-3)."""
        cfg, g, fe = self.cfg, self.cfg["gpr"], self.fe
        night = dt.date.fromisoformat(str(night))
        f = self.store.field(field_id)
        dates, ev, mask = f["dates"], f["evi2"].astype("float64"), f["mask"]
        T, H, W = ev.shape
        emerg, peak, _ = self.growth(field_id)
        reason = np.where(mask, 0, 1).astype("uint8")
        in_phase = emerg is not None and peak is not None and emerg <= night <= peak
        stage_txt = ("pre-emergence" if emerg is None or night < emerg else
                     "growth" if peak is None or night <= peak else "post-peak")
        if not in_phase:
            reason[mask] = 2

        # validity: same rules as evi2_gpr
        clear = np.isfinite(ev) & mask[None]
        t_days = np.array([(d - night).days for d in dates], dtype=float)
        abs_days = np.array([(d - self.day0).days for d in dates], dtype=float)
        before, after = t_days <= 0, t_days > 0
        since_em = np.array([emerg is not None and d >= emerg for d in dates])
        win = g.get("obs_window_days")
        in_window = before & (t_days >= -win if win is not None else True)
        cb, ca = clear & before[:, None, None], clear & after[:, None, None]
        n_em = (cb & since_em[:, None, None]).sum(0)
        n_win = (clear & in_window[:, None, None]).sum(0)
        has_b, has_a = cb.any(0), ca.any(0)
        last_b = T - 1 - np.argmax(cb[::-1], 0)
        first_a = np.argmax(ca, 0)
        c0 = reason == 0
        reason[c0 & ((n_em < g["min_clear_obs_since_emergence"])
                     | (n_win < g.get("min_clear_obs_before", 0)) | ~has_b)] = 3
        reason[(reason == 0) & (-t_days[last_b] > g["max_gap_before_days"])] = 4
        reason[(reason == 0) & (~has_a | (t_days[first_a] > g["max_gap_after_days"]))] = 5
        # second obs after: next clear one within persist_days of the night and the reference's max gap
        tt = np.arange(T)[:, None, None]
        ca2 = (ca & (tt > first_a[None]) & (t_days <= fe["persist_days"])[:, None, None]
               & ((abs_days[:, None, None] - abs_days[last_b][None]) <= cfg["reference"]["max_gap_days"]))
        second_a = np.where(ca2.any(0), np.argmax(ca2, 0), -1) if fe["n_post"] >= 2 else np.full((H, W), -1)

        z_raw = np.full((H, W), np.nan)
        s_lvl = np.full((H, W), np.nan)
        pred1 = np.full((H, W), np.nan)
        std1 = np.full((H, W), np.nan)
        obs1 = np.full((H, W), np.nan)
        npost = np.zeros((H, W), "int8")
        ys, xs = np.nonzero(reason == 0)
        hp = self._hyper_params(field_id, cb, ev, abs_days, int(before.sum())) if len(ys) else None
        if len(ys) and hp is None:
            reason[ys, xs] = 6
            ys, xs = ys[:0], xs[:0]
        if len(ys):
            amp, ls, noise = hp
            med = base.field_median_series(f, cfg["reference"]["min_clear_share"])
            key = np.concatenate([cb[:, ys, xs].T.astype(int), first_a[ys, xs][:, None],
                                  second_a[ys, xs][:, None]], 1)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.ravel()
            for gi, k in enumerate(uniq):
                pos = np.nonzero(inv == gi)[0]
                gy, gx = ys[pos], xs[pos]
                idx, posts = np.nonzero(k[:-2])[0], [int(i) for i in k[-2:] if i >= 0]
                t, yv = abs_days[idx], ev[idx][:, gy, gx]
                mu = yv.mean(0)
                try:
                    low = base._chol(t, amp, ls, noise)
                except np.linalg.LinAlgError:
                    reason[gy, gx] = 6
                    continue
                kq = base._rbf(np.array([t[-1]]), t, amp, ls)
                s = mu + (kq @ base.cho_solve((low, True), yv - mu))[0]
                vs = max(amp - float((kq @ base.cho_solve((low, True), kq.T))[0, 0]), 0.0)
                il = idx[-1]
                slope = base.field_slope(dates, med, il)
                doy = dates[il].timetuple().tm_yday
                res, var = [], []
                for ia in posts:
                    X = np.column_stack([s, np.full_like(s, slope), np.full_like(s, abs_days[ia] - t[-1]),
                                         np.full_like(s, doy)])
                    d, vd = self.ref.predict(field_id, X)
                    res.append(ev[ia, gy, gx] - s - d)
                    var.append(vd)
                    if ia == posts[0]:
                        pred1[gy, gx], std1[gy, gx] = s + d, np.sqrt(vs + vd + noise)
                        obs1[gy, gx] = ev[ia, gy, gx]
                if len(posts) == 1:
                    zz = res[0] / np.sqrt(vs + var[0] + noise)
                else:      # GLS mean of two residuals sharing the error of s
                    a11, a22 = vs + var[0] + noise, vs + var[1] + noise
                    a12 = vs + fe["ref_rho"] * np.sqrt(var[0] * var[1])
                    det = a11 * a22 - a12 ** 2
                    w1, w2 = (a22 - a12) / det, (a11 - a12) / det
                    zz = (w1 * res[0] + w2 * res[1]) / (w1 + w2) * np.sqrt(w1 + w2)
                ok = np.isfinite(zz)
                reason[gy[~ok], gx[~ok]] = 6
                z_raw[gy[ok], gx[ok]], s_lvl[gy[ok], gx[ok]] = zz[ok], s[ok]
                npost[gy[ok], gx[ok]] = len(posts)
        valid = reason == 0
        # spatial support: mean of z_raw over valid pixels in the window
        k = fe["smooth_px"]
        zf = np.where(valid, z_raw, 0.0)
        cnt = uniform_filter(valid.astype(float), k, mode="constant")
        z_smooth = np.where(valid, uniform_filter(zf, k, mode="constant") / np.maximum(cnt, 1e-9), np.nan)
        stage = np.digitize(np.nan_to_num(s_lvl, nan=0.0), fe["stage_bins"]).astype("int8")
        return dict(valid=valid, reason=reason, z_raw=z_raw, z_smooth=z_smooth, stage=stage, npost=npost,
                    pred1=pred1, std1=std1, obs1=obs1, last_b=last_b, first_a=first_a, second_a=second_a,
                    clear=clear, t_days=t_days, stage_txt=stage_txt, emerg=emerg, peak=peak, night=night)

    def damage(self, field_id, night):
        fe, ncfg = self.fe, self.cfg["night"]
        r = self._evidence(field_id, night)
        night = r["night"]
        f = self.store.field(field_id)
        dates, mask = f["dates"], f["mask"]
        valid = r["valid"]
        z = (self.cal(field_id, r["z_smooth"], r["stage"], r["npost"]) if self.cal is not None
             else r["z_smooth"].copy())
        bad = valid & ~np.isfinite(z)
        r["reason"][bad] = 6
        valid = r["reason"] == 0
        z = np.where(valid, z, np.nan)
        score = np.where(valid, np.maximum(0.0, -z), np.nan).astype("float32")

        nmeta = self.store.night_meta(field_id, night)
        cp = self.cold_pool(field_id)
        p_frost, att = attribute(self.cfg, fe, nmeta, cp, mask, z, valid, fe["z_damaged"])

        # representative dates (as evi2_gpr)
        T = len(dates)
        def mode(a):
            v = a[valid]
            return int(np.bincount(v).argmax()) if v.size else None
        pi, qi = mode(r["last_b"]), mode(r["first_a"])
        clear_any = r["clear"].any((1, 2))
        if pi is None:
            cb = np.nonzero((r["t_days"] <= 0) & clear_any)[0]
            ca = np.nonzero((r["t_days"] > 0) & clear_any)[0]
            pi = int(cb[-1]) if len(cb) else None
            qi = int(ca[0]) if len(ca) else None
        nxt = None
        if qi is not None:
            later = np.nonzero(clear_any & (np.arange(T) > qi))[0]
            nxt = int(later[0]) if len(later) else None
        iso = lambda i: dates[i].isoformat() if i is not None else None
        days = lambda i: int(r["t_days"][i]) if i is not None else None
        emerg, peak = r["emerg"], r["peak"]
        meta = dict(field_id=field_id, night=night.isoformat(), method=NAME,
                    pre_date=iso(pi), post_date=iso(qi), next_date=iso(nxt),
                    gap_days_before=-days(pi) if pi is not None else None,
                    gap_days_after=days(qi), next_days_after=days(nxt),
                    growth_stage=r["stage_txt"], emergence=emerg and emerg.isoformat(),
                    peak=peak and peak.isoformat(), **nmeta, **att)
        f32 = lambda a: np.where(valid, a, np.nan).astype("float32")
        return {"score": score, "valid": valid, "meta": meta, "curve": None,
                "layers": {"z": f32(z), "predicted": f32(r["pred1"]), "pred_std": f32(r["std1"]),
                           "observed": f32(r["obs1"]), "z_raw": f32(r["z_raw"]), "z_smooth": f32(r["z_smooth"]),
                           "n_post": r["npost"], "p_frost": p_frost.astype("float32"), "cold_pool": cp,
                           "reason": r["reason"], "pre_idx": pi, "post_idx": qi, "next_idx": nxt}}
