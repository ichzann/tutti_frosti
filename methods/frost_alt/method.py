"""frost_alt: learned frost-free spectral change model + multiscale matched-filter scan with a field-level
empirical null.  Sentinel-2 only in the score; weather never gates it (only defines frost-free training /
calibration intervals and is reported in meta).

1. Expected spectrum without frost.  For every pixel, the 10-band surface reflectance at a post obs is
   predicted from the last usable obs before the night by gradient boosting (one HistGradientBoosting model per
   band and field fold), trained only on real frost-free intervals of non-lockbox fields (every night of the
   interval RTMA Tmin >= min_tmin_c, no trigger).  Features: pixel spectrum before, its offset from the field
   median, field-median spectrum, EVI2 of both, field / pixel EVI2 slope before, gap, day of year.
   Leave-fold-out: a field is predicted only by the model of the other folds.  Residual r = obs - expected.
2. Frost fingerprint (linear spectral mixing): a canopy loss L of a pixel with green fraction g moves its
   spectrum by L * g * (E_killed - E_green), E_killed = dead_share * E_dead + (1 - dead_share) * E_soil; green,
   soil and dead endmembers are medians of non-lockbox pixels (July EVI2 >= 0.85; May EVI2 <= 0.12; September
   obs after the field's peak, EVI2 <= 0.30).  g from the pixel's EVI2 before.
3. Per post obs (up to 2): matched filter of the fingerprint in the residual space whitened by the
   out-of-fold residual covariance (per stratum of field-median EVI2 before and post rank), with the haze
   subspace lambda^-1, lambda^-2 (Angstrom exponents 1-2) projected out (orthogonal subspace projection).
   Post obs combined with weights 0.5 ** (t / half_days).
4. Multiscale scan: the matched filter summed over 3x3 and 7x7 windows and over the whole field, each centred
   and scaled by its frost-free null median / MAD per stratum; pixel statistic M = max over scales.
5. Field-level empirical null: on frost-free calibration intervals of non-lockbox fields (other folds), the
   distribution of the field's 90th-percentile of M (a field is flagged when >= 10 % of its valid pixels pass),
   pooled over strata (null_pool: all) or per stratum (null_pool: stratum).
   score = Phi^-1(null CDF of that field statistic at the pixel's M); damaged if score >= 2, i.e. a frost-free
   field reaches 10 % flagged pixels with probability ~2.3 %.
Training and calibration intervals: obs before the night in [emergence, peak) of the season's smoothed field-median
EVI2 (hindsight selection, as the validation domain).  The validity gate at application uses only data up to the
obs before the night (field-median EVI2 >= min_pre_evi2 and not below the season maximum so far - senescence_drop).
Caches: cache/frost_alt_model_<fips>_<key>.joblib (models, residual covariances, endmembers),
cache/frost_alt_cal_<fips>_<key>.npz (raw scale statistics of the calibration intervals).
Dev runs (validation harness, spectral mode, 150 fields / 396 cases):
  frost_alt_dev_v1 (null_pool stratum): fpr_px 0.0356, fpr_field 0.0306, auc 0.9786, tpr@0.20 0.8039,
                    field_det@0.35 1.0, cov95 0.9591, haze fpr_field 0.0306
  frost_alt_dev_v2 (null_pool all):     fpr_px 0.0309, fpr_field 0.0281, auc 0.9811, tpr@0.20 0.8073,
                    field_det@0.35 1.0, cov95 0.9591, haze fpr_field 0.0255
Layers predicted / pred_std / observed: EVI2 of the expected and observed first post obs, pred_std from the
out-of-fold EVI2 residual quantiles (2.5-97.5 % half-width / 1.96) per stratum.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from data.common import cache_dir
from data.store import FieldStore
from methods.evi2_gpr.method import growth_phase

NAME = "frost_alt"
REASONS = {0: "valid", 1: "outside field / CDL", 2: "outside growth domain (data before the night)",
           3: "no usable obs before in time", 4: "pixel not clear before", 5: "no usable obs after in time",
           6: "pixel not clear after / model failed"}
SETTINGS = Path(__file__).with_name("settings.yaml")
LAMBDA = np.array([490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190.0])   # data.s2bands.BANDS


def is_lockbox(field_id, share):
    # same rule as validation.cases.is_holdout (re-implemented: methods must not read from validation/)
    return int(hashlib.md5(str(field_id).encode()).hexdigest(), 16) % 1000 < share * 1000


def fold_of(field_id, n):
    return int(hashlib.md5(f"frost_alt:{field_id}".encode()).hexdigest(), 16) % n


def evi2(r, axis=0):
    r = np.moveaxis(np.asarray(r, "float64"), axis, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.5 * (r[6] - r[2]) / (r[6] + 2.4 * r[2] + 1.0)


def features(a, A, fslope, pslope, gap, doy):
    """a: (n, 10) pixel spectrum before; A: (10,) field median before -> (n, 37)."""
    n = len(a)
    col = lambda v: np.broadcast_to(np.asarray(v, "float64"), (n,))[:, None]
    return np.hstack([a, a - A[None], np.broadcast_to(A, (n, 10)), col(evi2(a, 1)), col(evi2(A[:, None])[0]),
                      col(fslope), col(pslope), col(gap), col(doy)])


def stratum(e_pre, bins):
    return int(np.digitize(e_pre, bins))


# --------------------------------------------------------------------------------- field view --
def field_view(store, fid, usable_share):
    f = store.field(fid)
    B = store.bands(fid).astype("float64")                       # (T, 10, H, W)
    mask = f["mask"]
    clear = np.isfinite(B).all(1) & mask[None]
    share = clear.sum((1, 2)) / max(mask.sum(), 1)
    E = evi2(B, 1)                                                # (T, H, W)
    use = np.nonzero(share >= usable_share)[0]
    med = np.full(len(f["dates"]), np.nan)
    spec = np.full((len(f["dates"]), 10), np.nan)
    for i in use:
        med[i] = np.median(E[i][clear[i]])
        spec[i] = np.median(B[i][:, clear[i]], 1)
    return dict(dates=f["dates"], mask=mask, B=B, E=E, clear=clear, use=use, med=med, spec=spec, f=f)


def interval_geometry(v, night, s):
    """Indices (pre, prev, post1, post2) of usable dates around the night, or a reason code."""
    dates, use, med = v["dates"], v["use"], v["med"]
    dd = np.array([(dates[i] - night).days for i in use])
    before = use[dd <= 0]
    if not len(before) or (night - dates[before[-1]]).days > s["max_days_before"]:
        return 3, None
    pre = int(before[-1])
    seen = med[before]
    if not (med[pre] >= s["min_pre_evi2"] and med[pre] >= np.nanmax(seen) - s["senescence_drop"]):
        return 2, None
    prev = int(before[-2]) if len(before) >= 2 else None
    after = use[dd > 0]
    ok1 = [i for i in after if (dates[i] - night).days <= s["max_days_after"]
           and (dates[i] - dates[pre]).days <= s["max_pair_days"]]
    if not ok1:
        return 5, None
    p1 = int(ok1[0])
    ok2 = [i for i in after if i > p1 and (dates[i] - night).days <= s["max_days_post2"]
           and (dates[i] - dates[pre]).days <= s["max_pair_days"]]
    return 0, dict(pre=pre, prev=prev, posts=[p1] + ([int(ok2[0])] if ok2 else []))


def _slopes(v, pre, prev, ys, xs):
    if prev is None:
        return 0.0, np.full(len(ys), -9.0)
    dp = (v["dates"][pre] - v["dates"][prev]).days
    fs = (v["med"][pre] - v["med"][prev]) / dp
    ps = (v["E"][pre, ys, xs] - v["E"][prev, ys, xs]) / dp
    return float(fs), np.where(np.isfinite(ps), ps, -9.0)


# ------------------------------------------------------------------------------------ training --
def frost_free_set(store, s):
    n = store.nights
    cold = n.trigger | (n.tmin_c < s["min_tmin_c"])
    return set(zip(n.cell_id[cold], n.night[cold])), set(zip(n.cell_id, n.night))


def is_frost_free(store, fid, d0, d1, cold, known):
    cell = store.field_cell.get(fid)
    for k in range((d1 - d0).days):
        key = (cell, (d0 + dt.timedelta(days=k)).isoformat())
        if key in cold or key not in known:
            return False
    return True


def in_growth_phase(cfg, v, d):
    """Training / calibration selection only (hindsight, whole season): emergence <= d < peak of the smoothed
    field-median EVI2 (evi2_gpr.growth_phase), i.e. the growth phase the method is calibrated for."""
    f = v["f"]
    if "_phase" not in v:
        v["_phase"] = growth_phase(f["dates"], f["evi2"], f["mask"], cfg)
    emerg, peak, _ = v["_phase"]
    return emerg is not None and peak is not None and emerg <= d < peak


def build_training(store, s):
    """Pixel rows of frost-free growth-phase intervals (k, k + j) of non-lockbox fields + endmember samples."""
    rng = np.random.default_rng(0)
    cold, known = frost_free_set(store, s)
    rows, green, soil, dead = [], [], [], []
    for fid in store.field_ids():
        if is_lockbox(fid, s["lockbox_share"]):
            continue
        v = field_view(store, fid, s["usable_share"])
        use, med, dates, B, E, clear = v["use"], v["med"], v["dates"], v["B"], v["E"], v["clear"]
        # endmember samples
        pk = int(np.nanargmax(med)) if np.isfinite(med).any() else None
        for i in use:
            sel = None
            if dates[i].month == 7:
                sel = clear[i] & (E[i] >= 0.85); lst = green
            elif dates[i].month == 5:
                sel = clear[i] & (E[i] <= 0.12); lst = soil
            elif pk is not None and i > pk and dates[i].month >= 9 and med[pk] >= 0.7:
                sel = clear[i] & (E[i] <= 0.30); lst = dead
            if sel is not None and sel.any():
                px = B[i][:, sel].T
                lst.append(px[rng.choice(len(px), min(50, len(px)), replace=False)])
        for kk in range(len(use) - 1):
            pre = int(use[kk])
            if not (med[pre] >= s["min_pre_evi2"] and med[pre] >= np.nanmax(med[use[:kk + 1]]) - s["senescence_drop"]
                    and in_growth_phase(store.cfg, v, dates[pre])):
                continue
            prev = int(use[kk - 1]) if kk >= 1 else None
            for j in range(1, s["max_step"] + 1):
                if kk + j >= len(use):
                    break
                post = int(use[kk + j])
                gap = (dates[post] - dates[pre]).days
                if gap > s["max_pair_days"] or not is_frost_free(store, fid, dates[pre], dates[post], cold, known):
                    break
                ys, xs = np.nonzero(clear[pre] & clear[post])
                if len(ys) < 50:
                    continue
                pick = rng.choice(len(ys), min(s["train_px_per_pair"], len(ys)), replace=False)
                ys, xs = ys[pick], xs[pick]
                a = B[pre][:, ys, xs].T
                fs, ps = _slopes(v, pre, prev, ys, xs)
                X = features(a, v["spec"][pre], fs, ps, gap, dates[pre].timetuple().tm_yday)
                rows.append((fid, j, med[pre], X, B[post][:, ys, xs].T - a))
        store.field.cache_clear()
        store.bands.cache_clear()
    em = {k: np.median(np.concatenate(x), 0) for k, x in (("green", green), ("soil", soil), ("dead", dead))}
    return rows, em


def train(store, s):
    rows, em = build_training(store, s)
    fid = np.concatenate([[r[0]] * len(r[3]) for r in rows])
    jj = np.concatenate([np.full(len(r[3]), r[1]) for r in rows])
    epre = np.concatenate([np.full(len(r[3]), r[2]) for r in rows])
    X = np.concatenate([r[3] for r in rows])
    Y = np.concatenate([r[4] for r in rows])
    fold = np.array([fold_of(i, s["folds"]) for i in fid])
    rng = np.random.default_rng(1)
    models, oof = {}, np.full(Y.shape, np.nan)
    for q in range(s["folds"]):
        tr = np.nonzero(fold != q)[0]
        tr = rng.choice(tr, min(s["train_rows_per_fold"], len(tr)), replace=False)
        te = fold == q
        models[q] = []
        for b in range(10):
            m = HistGradientBoostingRegressor(random_state=0, **s["gbm"]).fit(X[tr], Y[tr, b])
            models[q].append(m)
            if te.any():                            # a fold can be empty in small stores
                oof[te, b] = m.predict(X[te])
    R = Y - oof
    st = np.digitize(epre, s["strata"])
    a = X[:, :10]
    e_res = evi2(a + Y, 1) - evi2(a + oof, 1)
    cov, estd = {}, {}
    for k in range(len(s["strata"]) + 1):
        for jp in (1, 2):
            sel = (st == k) & ((jj == 1) if jp == 1 else (jj >= 2))
            if sel.sum() < 30:                      # empty cell (small stores): pool all strata
                sel = (jj == 1) if jp == 1 else (jj >= 2)
            rr = R[sel]
            # robust covariance: clip residuals at 6 robust sd per band before the covariance
            sc = 1.4826 * np.median(np.abs(rr - np.median(rr, 0)), 0)
            rr = np.clip(rr, -6 * sc, 6 * sc)
            cov[(k, jp)] = np.cov(rr.T)
        sel = (st == k) & (jj == 1)
        sel = sel if sel.sum() >= 30 else jj == 1
        lo, hi = np.quantile(e_res[sel], [0.025, 0.975])
        estd[k] = float((hi - lo) / (2 * 1.96))
    info = dict(n_rows=len(Y), n_fields=len(set(fid)),
                n_rows_by_stratum=np.bincount(st, minlength=len(s["strata"]) + 1).tolist())
    return dict(models=models, cov=cov, estd=estd, em=em, info=info)


# ------------------------------------------------------------------------------------- method --
_CACHE = {}


def _key(s):
    return hashlib.md5(json.dumps(s, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _model_keys(s):
    return {k: s[k] for k in ("phase", "lockbox_share", "folds", "usable_share", "min_pre_evi2", "senescence_drop",
                              "max_pair_days", "min_tmin_c", "max_step", "train_px_per_pair",
                              "train_rows_per_fold", "gbm", "strata")}


def _cal_keys(s):
    # raw calibration statistics do not depend on the threshold or on how the null is pooled
    return {k: v for k, v in s.items() if k not in ("z_damaged", "null_pool")}


class Method:
    name = NAME

    def __init__(self, cfg, store):
        cfg = copy.deepcopy(cfg)
        if "frost_alt" not in cfg:
            cfg["frost_alt"] = yaml.safe_load(SETTINGS.read_text())
        self.cfg, self.store = cfg, store
        self.s = s = cfg["frost_alt"]
        self.score_threshold = float(s["z_damaged"])
        self.day0 = dt.date.fromisoformat(cfg["study"]["season_start"])
        self._gp = {}
        self.mdl = self._load_model(store)
        em = self.mdl["em"]
        ek = s["dead_share"] * em["dead"] + (1 - s["dead_share"]) * em["soil"]
        self.d_frost = ek - em["green"]                                    # per unit loss of g = 1
        self.e_green = float(evi2(em["green"][:, None])[0])
        self.e_soil = float(evi2(em["soil"][:, None])[0])
        self.haze = np.column_stack([(LAMBDA / 490.0) ** -a for a in s["haze_angstrom"]])
        self._W = {}
        for key, C in self.mdl["cov"].items():
            Wm = np.linalg.inv(np.linalg.cholesky(C))
            Q, _ = np.linalg.qr(Wm @ self.haze)
            self._W[key] = (np.eye(10) - Q @ Q.T) @ Wm                    # whiten, then remove haze subspace
        self.cal = self._load_calibration(store)

    # ------------------------------------------------------------------------------- caches --
    def _load_model(self, store):
        key = "model_" + _key(_model_keys(self.s))
        if key not in _CACHE:
            fp = cache_dir(self.cfg) / f"frost_alt_model_{store.fips}_{key[6:]}.joblib"
            if not fp.exists():
                if not isinstance(store, FieldStore):
                    raise RuntimeError("frost_alt model must first be built from a FieldStore")
                tmp = fp.with_name(f"{fp.name}.{os.getpid()}.tmp")
                joblib.dump(train(store, self.s), tmp)
                os.replace(tmp, fp)
            _CACHE[key] = joblib.load(fp)
        return _CACHE[key]

    def _load_calibration(self, store):
        key = "cal_" + _key(_cal_keys(self.s))
        if key not in _CACHE:
            fp = cache_dir(self.cfg) / f"frost_alt_cal_{store.fips}_{key[4:]}.npz"
            if not fp.exists():
                if not isinstance(store, FieldStore):
                    raise RuntimeError("frost_alt calibration must first be built from a FieldStore")
                tmp = fp.with_name(f"{fp.name}.{os.getpid()}.tmp.npz")
                np.savez_compressed(tmp, **self._calibration_raw(store))
                os.replace(tmp, fp)
            _CACHE[key] = Calibration(dict(np.load(fp)), self.s)
        return _CACHE[key]

    def _calibration_raw(self, store):
        """Raw scale statistics of every frost-free calibration interval (night = usable obs date)."""
        s = self.s
        cold, known = frost_free_set(store, s)
        rows, parts = [], {f"T{k}": [] for k in s["scales"] if k != "field"}
        for fid in store.field_ids():
            if is_lockbox(fid, s["lockbox_share"]):
                continue
            v = field_view(store, fid, s["usable_share"])
            for i in v["use"]:
                night = v["dates"][i]
                code, geo = interval_geometry(v, night, s)
                if code or not in_growth_phase(self.cfg, v, night):
                    continue
                last = v["dates"][geo["posts"][-1]]
                if not is_frost_free(store, fid, night, last, cold, known):
                    continue
                r = self._evidence(fid, night, v=v)
                val = r["valid"]
                if val.sum() < s["min_valid_px"]:
                    continue
                rows.append((fid, fold_of(fid, s["folds"]), r["stratum"], r["TF"], int(val.sum()), night.isoformat()))
                for k in s["scales"]:
                    if k != "field":
                        parts[f"T{k}"].append(r["T"][k][val].astype("float32"))
            store.field.cache_clear()
            store.bands.cache_clear()
        out = dict(field_id=np.array([r[0] for r in rows]), fold=np.array([r[1] for r in rows]),
                   stratum=np.array([r[2] for r in rows]), TF=np.array([r[3] for r in rows]),
                   n=np.array([r[4] for r in rows]), date=np.array([r[5] for r in rows]))
        for k, p in parts.items():
            out[k] = np.concatenate(p)
        return out

    # ----------------------------------------------------------------------------- evidence --
    def _evidence(self, field_id, night, v=None):
        s = self.s
        night = dt.date.fromisoformat(str(night))
        v = v or field_view(self.store, field_id, s["usable_share"])
        mask = v["mask"]
        H, W = mask.shape
        reason = np.where(mask, 0, 1).astype("uint8")
        code, geo = interval_geometry(v, night, s)
        out = dict(reason=reason, valid=np.zeros((H, W), bool), geo=geo, night=night, stratum=None, TF=np.nan,
                   T={}, pred1=None)
        if code:
            reason[mask] = code
            return out
        pre, posts = geo["pre"], geo["posts"]
        clear = v["clear"]
        reason[mask & ~clear[pre]] = 4
        reason[(reason == 0) & ~clear[posts[0]]] = 6
        ys, xs = np.nonzero(reason == 0)
        if not len(ys):
            return out
        dates, B = v["dates"], v["B"]
        a = B[pre][:, ys, xs].T
        fs, ps = _slopes(v, pre, geo["prev"], ys, xs)
        st = stratum(v["med"][pre], s["strata"])
        models = self.mdl["models"][fold_of(field_id, s["folds"])]
        g = np.clip((evi2(a, 1) - self.e_soil) / (self.e_green - self.e_soil), 0, 1)
        dF = g[:, None] * self.d_frost[None]                               # (n, 10)
        NUM = np.zeros(len(ys)); DEN = np.zeros(len(ys))
        pred1 = obs1 = None
        for jp, p in enumerate(posts, 1):
            ok = clear[p, ys, xs]
            gap = (dates[p] - dates[pre]).days
            X = features(a[ok], v["spec"][pre], fs, ps[ok], gap, dates[pre].timetuple().tm_yday)
            with threadpool_limits(1):        # small batches: one thread is faster and safe with harness workers
                pr = np.column_stack([m.predict(X) for m in models])
            r = B[p][:, ys[ok], xs[ok]].T - a[ok] - pr
            Wp = self._W[(st, jp)]
            wd, wr = dF[ok] @ Wp.T, r @ Wp.T
            w = 0.5 ** ((dates[p] - night).days / s["half_days"])
            NUM[ok] += w * (wd * wr).sum(1)
            DEN[ok] += w * w * (wd * wd).sum(1)
            if jp == 1:
                pred1 = evi2(a + pr, 1)
                obs1 = evi2(B[p][:, ys, xs].T, 1)
        bad = ~(np.isfinite(NUM) & (DEN > 0))
        reason[ys[bad], xs[bad]] = 6
        valid = reason == 0
        num = np.zeros((H, W)); den = np.zeros((H, W))
        num[ys, xs] = np.where(bad, 0, NUM); den[ys, xs] = np.where(bad, 0, DEN)
        T = {}
        for k in s["scales"]:
            if k == "field":
                continue
            un, ud = uniform_filter(num, k, mode="constant"), uniform_filter(den, k, mode="constant")
            T[k] = np.where(valid, un / np.sqrt(np.maximum(ud, 1e-30)) * k, np.nan)
        TF = float(num[valid].sum() / np.sqrt(den[valid].sum())) if valid.any() else np.nan
        lay = {}
        for name, arr in (("predicted", pred1), ("observed", obs1)):
            a2 = np.full((H, W), np.nan)
            a2[ys, xs] = arr
            lay[name] = a2
        lay["pred_std"] = np.where(valid, self.mdl["estd"][st], np.nan)
        lay["loss"] = np.where(valid, num / np.maximum(den, 1e-30) * 1.0, np.nan)   # unsmoothed loss estimate
        out.update(valid=valid, stratum=st, TF=TF, T=T, layers=lay)
        return out

    # ------------------------------------------------------------------------------- damage --
    def growth(self, field_id):
        if field_id not in self._gp:
            f = self.store.field(field_id)
            self._gp[field_id] = growth_phase(f["dates"], f["evi2"], f["mask"], self.cfg)
        return self._gp[field_id]

    def damage(self, field_id, night):
        s = self.s
        r = self._evidence(field_id, night)
        night = r["night"]
        f = self.store.field(field_id)
        dates, mask = f["dates"], f["mask"]
        H, W = mask.shape
        valid = r["valid"]
        z = np.full((H, W), np.nan)
        M = np.full((H, W), np.nan)
        if valid.any():
            M = self.cal.statistic(field_id, r["stratum"], r["T"], r["TF"])
            z = self.cal.score(field_id, r["stratum"], M)
            r["reason"][valid & ~np.isfinite(z)] = 6
            valid = r["reason"] == 0
        score = np.where(valid, z, np.nan).astype("float32")
        geo = r["geo"] or {}
        pi = geo.get("pre")
        posts = geo.get("posts", [])
        qi = posts[0] if posts else None
        nxt = posts[1] if len(posts) > 1 else None
        iso = lambda i: dates[i].isoformat() if i is not None else None
        days = lambda i: (dates[i] - night).days if i is not None else None
        emerg, peak, _ = self.growth(field_id)
        nmeta = self.store.night_meta(field_id, night)
        dmg = valid & (score >= self.score_threshold)
        meta = dict(field_id=field_id, night=night.isoformat(), method=NAME,
                    pre_date=iso(pi), post_date=iso(qi), next_date=iso(nxt),
                    gap_days_before=-days(pi) if pi is not None else None,
                    gap_days_after=days(qi), next_days_after=days(nxt),
                    growth_stage="growth" if valid.any() else (
                        REASONS[int(np.bincount(r["reason"][mask]).argmax())] if mask.any() else None),
                    emergence=emerg and emerg.isoformat(), peak=peak and peak.isoformat(), **nmeta,
                    stratum=r["stratum"], field_stat=r["TF"],
                    share_damaged=float(dmg.sum() / valid.sum()) if valid.any() else np.nan)
        lay = r.get("layers") or {}
        f32 = lambda a: np.where(valid, a, np.nan).astype("float32")
        layers = {"z": -score, "M": f32(M),     # report convention: z <= -2 = damaged (score = -z) "reason": r["reason"], "pre_idx": pi, "post_idx": qi,
                  "next_idx": nxt}
        if lay:
            layers.update(predicted=f32(lay["predicted"]), pred_std=f32(lay["pred_std"]),
                          observed=f32(lay["observed"]), loss=f32(lay["loss"]))
        return {"score": score, "valid": valid, "meta": meta, "curve": None, "layers": layers}


class Calibration:
    """Leave-fold-out frost-free null: per stratum, centring / scaling of every scale statistic (median, MAD)
    and the null distribution of the field's (1 - field_flag_share) quantile of M = max over scales."""

    def __init__(self, raw, s):
        self.s = s
        self.scales = [k for k in s["scales"] if k != "field"]
        self.use_field = "field" in s["scales"]
        n = raw["n"]
        off = np.concatenate([[0], np.cumsum(n)])
        folds, strata = raw["fold"], raw["stratum"]
        self.norm, self.null = {}, {}
        nst = len(s["strata"]) + 1
        for q in range(s["folds"]):
            for st in range(nst):
                ids = np.nonzero((folds != q) & (strata == st))[0]
                if len(ids) < s["cal_min_n"]:
                    # pool with the neighbouring strata until enough intervals
                    for w in range(1, nst):
                        ids = np.nonzero((folds != q) & (np.abs(strata - st) <= w))[0]
                        if len(ids) >= s["cal_min_n"]:
                            break
                nm = {}
                for k in self.scales:
                    x = np.concatenate([raw[f"T{k}"][off[i]:off[i + 1]] for i in ids])
                    med = float(np.median(x))
                    nm[k] = (med, float(1.4826 * np.median(np.abs(x - med))))
                tf = raw["TF"][ids]
                med = float(np.median(tf))
                nm["field"] = (med, float(1.4826 * np.median(np.abs(tf - med))))
                self.norm[(q, st)] = nm
                if s.get("null_pool", "stratum") == "stratum":
                    self.null[(q, st)] = np.sort([self._fieldq(self._max(
                        nm, {k: raw[f"T{k}"][off[i]:off[i + 1]] for k in self.scales}, raw["TF"][i])) for i in ids])
            if s.get("null_pool", "stratum") == "all":
                # one null over all strata; every interval standardised with its own stratum's norms
                a = np.sort([self._fieldq(self._max(
                    self.norm[(q, int(strata[i]))], {k: raw[f"T{k}"][off[i]:off[i + 1]] for k in self.scales},
                    raw["TF"][i])) for i in np.nonzero(folds != q)[0]])
                for st in range(nst):
                    self.null[(q, st)] = a

    def _max(self, nm, Ts, TF):
        M = None
        for k in self.scales:
            z = (Ts[k] - nm[k][0]) / nm[k][1]
            M = z if M is None else np.maximum(M, z)
        if self.use_field:
            M = np.maximum(M, (TF - nm["field"][0]) / nm["field"][1])
        return M

    def _fieldq(self, M):
        """Largest value that >= field_flag_share of the pixels reach."""
        x = np.sort(M)[::-1]
        k = int(np.ceil(self.s["field_flag_share"] * len(x))) - 1
        return float(x[max(k, 0)])

    def statistic(self, field_id, st, T, TF):
        q = fold_of(field_id, self.s["folds"])
        return self._max(self.norm[(q, st)], {k: T[k] for k in self.scales}, TF)

    def score(self, field_id, st, M):
        a = self.null[(fold_of(field_id, self.s["folds"]), st)]
        p = (np.searchsorted(a, M, side="left") + 0.5) / (len(a) + 1)
        return norm.ppf(p)
