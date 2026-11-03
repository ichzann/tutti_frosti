"""frost_recovery: plant anomaly first (EVI2 + leaf water), judged by its recovery trajectory against the
literature for the crop stage; weather only afterwards as attribution (never a gate).

1. Two plant signals per pixel: EVI2 (canopy greenness) and NDMI = (B8A - B11) / (B8A + B11) (leaf water).
   Their expected change from the last usable obs before the night to every usable obs in the recovery window
   (night, night + window_days] is learned on real frost-free intervals of non-lockbox fields (pairs and features
   of frost_alt.build_training; one HistGradientBoosting model per signal and field fold, leave-fold-out).
   Residual r = (observed - expected) of both signals.  The two residuals are correlated (both follow leaf
   area); they are whitened with their out-of-fold frost-free covariance per (EVI2 stratum, gap bin), so the
   shared part counts once and each signal adds only its independent evidence.
2. Frost direction: a canopy loss L of a pixel with green fraction g moves its spectrum by
   L * g * (E_killed - E_green) (linear mixing, endmembers as frost_alt) -> expected (dEVI2, dNDMI) per unit loss.
3. Recovery templates: loss(t) = A for t <= hold_days, then halving every h days (h = null: no recovery inside
   the window).  The allowed h depend on the crop stage, P-days (Sands et al. 1979) since emergence from gridMET
   Tmax / Tmin: vegetative {7, 14} d, tuber initiation / bulking {7, 14, none}, late bulk {14, none} (settings.yaml,
   literature in papers/).  For each template, matched filter over all post obs: S_h = sum_k w_k p_k / sqrt(sum_k
   w_k^2 e_k), w_k = template at obs k, p_k = whitened direction . whitened residual, e_k = |whitened direction|^2,
   numerator and denominator summed over a smooth_px x smooth_px window of valid pixels.
4. Noise rule (persistence): real damage is still present persist_days after the night (no regrowth before ~5 d,
   defoliation still visible at 10 d).  If >= 2 usable obs fall in (night, night + persist_days], S_h is
   also computed without the obs that contributes most, and the smaller value is kept: a one-date dip (haze,
   shadow, a bad acquisition) that the next obs does not confirm is not damage.
5. Pixel statistic S = max_h S_h over the stage's allowed templates.  Field-level empirical null as frost_alt:
   per (fold, stage) centring / scaling of S on frost-free calibration intervals of other folds, then the null
   distribution of the field's 90th percentile, pooled.  score = Phi^-1(null CDF); damaged if score >= 2.
   Calibration intervals use only the post obs up to the first cold night (RTMA Tmin < min_tmin_c or trigger).
6. Attribution (layers / meta only): p_frost, terrain_rho and attribution label of frost_evidence.attribute.
Caches: cache/frost_recovery_model_<fips>_<key>.joblib, cache/frost_recovery_cal_<fips>_<key>.npz.
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
import yaml
from scipy.ndimage import uniform_filter
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from data.common import cache_dir
from data.store import FieldStore
from methods.evi2_gpr.method import growth_phase
from methods.frost_alt import method as fa
from methods.frost_evidence.method import attribute, cold_pool_index

NAME = "frost_recovery"
REASONS = {0: "valid", 1: "outside field / CDL", 2: "outside growth domain (data before the night)",
           3: "no usable obs before in time", 4: "pixel not clear before", 5: "no usable obs after in time",
           6: "pixel not clear after / model failed", 7: "no temperature data for the crop stage"}
SETTINGS = Path(__file__).with_name("settings.yaml")
GAP_COL = 34                     # column of the pre -> post gap (days) in frost_alt.features
evi2 = fa.evi2


def ndmi(r, axis=0):
    r = np.moveaxis(np.asarray(r, "float64"), axis, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (r[7] - r[8]) / (r[7] + r[8])            # B8A, B11


def signals(a):
    """(n, 10) spectra -> (n, 2) [EVI2, NDMI]."""
    return np.column_stack([evi2(a, 1), ndmi(a, 1)])


def pday(tmax, tmin, p):
    """Daily physiological days (Sands et al. 1979): 1/24 [5 P(Tn) + 8 P((2Tn+Tx)/3) + 8 P((Tn+2Tx)/3) + 3 P(Tx)]."""
    tb, to, tm = p["t_base"], p["t_opt"], p["t_max"]

    def P(T):
        T = np.asarray(T, "float64")
        lo = 10 * (1 - ((T - to) / (to - tb)) ** 2)
        hi = 10 * (1 - ((T - to) / (tm - to)) ** 2)
        return np.where(T < tb, 0.0, np.where(T < to, lo, np.where(T < tm, hi, 0.0)))

    return (5 * P(tmin) + 8 * P((2 * tmin + tmax) / 3) + 8 * P((tmin + 2 * tmax) / 3) + 3 * P(tmax)) / 24


def template(t, hold, h):
    t = np.asarray(t, "float64")
    if h is None:
        return np.ones_like(t)
    return np.where(t <= hold, 1.0, 0.5 ** ((t - hold) / h))


def geometry(v, night, s, until=None):
    """Indices (pre, prev, posts) of usable dates around the night, or a reason code.  until: last allowed post
    date (calibration: frost-free part of the window)."""
    dates, use, med = v["dates"], v["use"], v["med"]
    dd = np.array([(dates[i] - night).days for i in use])
    before = use[dd <= 0]
    if not len(before) or (night - dates[before[-1]]).days > s["max_days_before"]:
        return 3, None
    pre = int(before[-1])
    if not (med[pre] >= s["min_pre_evi2"] and med[pre] >= np.nanmax(med[before]) - s["senescence_drop"]):
        return 2, None
    prev = int(before[-2]) if len(before) >= 2 else None
    posts = [int(i) for i in use[dd > 0] if (dates[i] - night).days <= s["window_days"]
             and (dates[i] - dates[pre]).days <= s["max_pair_days"] and (until is None or dates[i] <= until)]
    posts = posts[:s["max_posts"]]
    if not posts or (dates[posts[0]] - night).days > s["max_days_after"]:
        return 5, None
    return 0, dict(pre=pre, prev=prev, posts=posts)


# ------------------------------------------------------------------------------------ training --
def train(store, s):
    rows, em = fa.build_training(store, s)
    fid = np.concatenate([[r[0]] * len(r[3]) for r in rows])
    epre = np.concatenate([np.full(len(r[3]), r[2]) for r in rows])
    X = np.concatenate([r[3] for r in rows])
    Y10 = np.concatenate([r[4] for r in rows])
    a = X[:, :10]
    D = signals(a + Y10) - signals(a)                               # (n, 2) change of EVI2, NDMI
    ok = np.isfinite(D).all(1)
    fid, epre, X, D = fid[ok], epre[ok], X[ok], D[ok]
    fold = np.array([fa.fold_of(i, s["folds"]) for i in fid])
    rng = np.random.default_rng(1)
    models, oof = {}, np.full(D.shape, np.nan)
    for q in range(s["folds"]):
        tr = np.nonzero(fold != q)[0]
        tr = rng.choice(tr, min(s["train_rows_per_fold"], len(tr)), replace=False)
        te = fold == q
        models[q] = []
        for b in range(2):
            m = HistGradientBoostingRegressor(random_state=0, **s["gbm"]).fit(X[tr], D[tr, b])
            models[q].append(m)
            oof[te, b] = m.predict(X[te])
    R = D - oof
    st = np.digitize(epre, s["strata"])
    gb = np.digitize(X[:, GAP_COL], s["gap_bins"])
    cov, estd, ncov = {}, {}, {}

    def robust_cov(rr):
        sc = 1.4826 * np.median(np.abs(rr - np.median(rr, 0)), 0)
        return np.cov(np.clip(rr, -6 * sc, 6 * sc).T)

    for k in range(len(s["strata"]) + 1):
        for g in range(len(s["gap_bins"]) + 1):
            sel = (st == k) & (gb == g)
            if sel.sum() < 500:                     # thin cell: pool the gap bins of the stratum
                sel = st == k
            cov[(k, g)] = robust_cov(R[sel])
            ncov[f"{k},{g}"] = int(((st == k) & (gb == g)).sum())
        sel = (st == k) & (gb == 0)
        sel = sel if sel.sum() >= 500 else st == k
        lo, hi = np.quantile(R[sel, 0], [0.025, 0.975])
        estd[k] = float((hi - lo) / (2 * 1.96))
    corr = np.corrcoef(R.T)[0, 1]
    info = dict(n_rows=len(D), n_fields=len(set(fid)), n_cov=ncov, resid_corr_evi2_ndmi=float(corr),
                gap_quantiles=np.quantile(X[:, GAP_COL], [0.1, 0.5, 0.9]).tolist())
    return dict(models=models, cov=cov, estd=estd, em=em, info=info)


# ------------------------------------------------------------------------------------- method --
_CACHE = {}


def _key(d):
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _model_keys(s):
    return {k: s[k] for k in ("lockbox_share", "folds", "usable_share", "min_pre_evi2", "senescence_drop",
                              "max_pair_days", "min_tmin_c", "max_step", "train_px_per_pair",
                              "train_rows_per_fold", "gbm", "strata", "gap_bins")}


def _cal_keys(s):
    return {k: v for k, v in s.items() if k not in ("z_damaged",)}


class Method:
    name = NAME

    def __init__(self, cfg, store):
        cfg = copy.deepcopy(cfg)
        if NAME not in cfg:
            cfg[NAME] = yaml.safe_load(SETTINGS.read_text())
        self.cfg, self.store = cfg, store
        self.s = s = cfg[NAME]
        self.score_threshold = float(s["z_damaged"])
        self._gp, self._cp = {}, {}
        self.mdl = self._load_model(store)
        em = self.mdl["em"]
        ek = s["dead_share"] * em["dead"] + (1 - s["dead_share"]) * em["soil"]
        self.d_frost = ek - em["green"]                                  # spectrum change per unit loss, g = 1
        self.e_green = float(evi2(em["green"][:, None])[0])
        self.e_soil = float(evi2(em["soil"][:, None])[0])
        self.W = {k: np.linalg.inv(np.linalg.cholesky(C)) for k, C in self.mdl["cov"].items()}
        self.cal = self._load_calibration(store)

    def clear_cache(self):
        self._gp.clear()
        self._cp.clear()

    # ------------------------------------------------------------------------------- caches --
    def _load_model(self, store):
        key = "model_" + _key(_model_keys(self.s))
        if key not in _CACHE:
            fp = cache_dir(self.cfg) / f"{NAME}_model_{store.fips}_{key[6:]}.joblib"
            if not fp.exists():
                if not isinstance(store, FieldStore):
                    raise RuntimeError(f"{NAME} model must first be built from a FieldStore")
                tmp = fp.with_name(f"{fp.name}.{os.getpid()}.tmp")
                joblib.dump(train(store, self.s), tmp)
                os.replace(tmp, fp)
            _CACHE[key] = joblib.load(fp)
        return _CACHE[key]

    def _load_calibration(self, store):
        key = "cal_" + _key(_cal_keys(self.s))
        if key not in _CACHE:
            fp = cache_dir(self.cfg) / f"{NAME}_cal_{store.fips}_{key[4:]}.npz"
            if not fp.exists():
                if not isinstance(store, FieldStore):
                    raise RuntimeError(f"{NAME} calibration must first be built from a FieldStore")
                tmp = fp.with_name(f"{fp.name}.{os.getpid()}.tmp.npz")
                np.savez_compressed(tmp, **self._calibration_raw(store))
                os.replace(tmp, fp)
            _CACHE[key] = Calibration(dict(np.load(fp)), self.s)
        return _CACHE[key]

    def _calibration_raw(self, store):
        """Per frost-free calibration interval (night = a usable obs date): field's 90th percentile of S and a
        pixel subsample of S; post obs only up to the first cold night after the night."""
        s = self.s
        cold, known = fa.frost_free_set(store, s)
        rng = np.random.default_rng(2)
        rows, px = [], []
        for fid in store.field_ids():
            if fa.is_lockbox(fid, s["lockbox_share"]):
                continue
            v = fa.field_view(store, fid, s["usable_share"])
            cell = store.field_cell.get(fid)
            for i in v["use"]:
                night = v["dates"][i]
                if not fa.in_growth_phase(self.cfg, v, night):
                    continue
                until = night
                while (until - night).days < s["window_days"]:
                    key = (cell, until.isoformat())
                    if key in cold or key not in known:
                        break
                    until += dt.timedelta(days=1)
                r = self._evidence(fid, night, v=v, until=until)
                val = r["valid"]
                if val.sum() < s["min_valid_px"]:
                    continue
                x = r["S"][val]
                rows.append((fid, fa.fold_of(fid, s["folds"]), r["stage"], self._fieldq(x), night.isoformat(),
                             len(r["geo"]["posts"])))
                px.append(rng.choice(x, min(s["cal_px_per_interval"], len(x)), replace=False).astype("float32"))
            store.field.cache_clear()
            store.bands.cache_clear()
            self._gp.pop(fid, None)
        return dict(field_id=np.array([r[0] for r in rows]), fold=np.array([r[1] for r in rows]),
                    stage=np.array([r[2] for r in rows]), q=np.array([r[3] for r in rows]),
                    date=np.array([r[4] for r in rows]), n_post=np.array([r[5] for r in rows]),
                    n_px=np.array([len(p) for p in px]), px=np.concatenate(px))

    def _fieldq(self, x):
        x = np.sort(x)[::-1]
        return float(x[max(int(np.ceil(self.s["field_flag_share"] * len(x))) - 1, 0)])

    # ---------------------------------------------------------------------------- crop stage --
    def growth(self, field_id):
        if field_id not in self._gp:
            f = self.store.field(field_id)
            self._gp[field_id] = growth_phase(f["dates"], f["evi2"], f["mask"], self.cfg)
        return self._gp[field_id]

    def stage(self, field_id, night):
        """(P-days since emergence up to the night, stage index) or (nan, None) without temperature data."""
        s = self.s
        emerg = self.growth(field_id)[0]
        t = self.store.daily_temp(field_id)
        if t is None:
            return np.nan, None
        if emerg is None:
            pd_ = 0.0
        else:
            d0 = (emerg - dt.timedelta(days=s["emergence_offset_days"])).isoformat()
            tt = t[(t.index >= d0) & (t.index < night.isoformat())]
            pd_ = float(pday(tt.tmax_c.to_numpy(), tt.tmin_c.to_numpy(), s["pday"]).sum())
        k = next(i for i, st in enumerate(s["stages"]) if pd_ < st["until"])
        return pd_, k

    # ----------------------------------------------------------------------------- evidence --
    def _evidence(self, field_id, night, v=None, until=None):
        s = self.s
        night = dt.date.fromisoformat(str(night))
        v = v or fa.field_view(self.store, field_id, s["usable_share"])
        mask = v["mask"]
        H, W = mask.shape
        reason = np.where(mask, 0, 1).astype("uint8")
        out = dict(reason=reason, valid=np.zeros((H, W), bool), geo=None, night=night, stage=None, pdays=np.nan,
                   S=None, layers={})
        code, geo = geometry(v, night, s, until)
        out["geo"] = geo
        if code:
            reason[mask] = code
            return out
        pdays, sk = self.stage(field_id, night)
        out.update(pdays=pdays, stage=sk)
        if sk is None:
            reason[mask] = 7
            return out
        pre, posts = geo["pre"], geo["posts"]
        clear, dates, B = v["clear"], v["dates"], v["B"]
        reason[mask & ~clear[pre]] = 4
        reason[(reason == 0) & ~clear[posts[0]]] = 6
        ys, xs = np.nonzero(reason == 0)
        if not len(ys):
            return out
        a = B[pre][:, ys, xs].T
        I0 = signals(a)
        g = np.clip((I0[:, 0] - self.e_soil) / (self.e_green - self.e_soil), 0, 1)
        d = (signals(a + 0.2 * g[:, None] * self.d_frost[None]) - I0) / 0.2   # (n, 2) per unit loss
        fs, ps = fa._slopes(v, pre, geo["prev"], ys, xs)
        st = fa.stratum(v["med"][pre], s["strata"])
        models = self.mdl["models"][fa.fold_of(field_id, s["folds"])]
        n = len(ys)
        t_k, Pk, Ek = [], [], []
        pred1 = obs1 = None
        for p in posts:
            ok = clear[p, ys, xs]
            gap = (dates[p] - dates[pre]).days
            X = fa.features(a[ok], v["spec"][pre], fs, ps[ok], gap, dates[pre].timetuple().tm_yday)
            with threadpool_limits(1):
                pr = np.column_stack([m.predict(X) for m in models])
            r = signals(B[p][:, ys[ok], xs[ok]].T) - I0[ok] - pr
            Wm = self.W[(st, int(np.digitize(gap, s["gap_bins"])))]
            u, vv = r @ Wm.T, d[ok] @ Wm.T
            pk, ek = np.zeros(n), np.zeros(n)
            good = np.isfinite(u).all(1) & np.isfinite(vv).all(1)
            idx = np.nonzero(ok)[0][good]
            pk[idx] = (vv[good] * u[good]).sum(1)
            ek[idx] = (vv[good] * vv[good]).sum(1)
            t_k.append((dates[p] - night).days)
            Pk.append(pk)
            Ek.append(ek)
            if p == posts[0]:
                pred1 = np.full(n, np.nan)
                obs1 = np.full(n, np.nan)
                pred1[ok] = I0[ok, 0] + pr[:, 0]
                obs1[ok] = signals(B[p][:, ys[ok], xs[ok]].T)[:, 0]
        bad = Ek[0] <= 0
        reason[ys[bad], xs[bad]] = 6
        valid = reason == 0
        k = s["smooth_px"]

        def grid(x):
            a2 = np.zeros((H, W))
            a2[ys, xs] = np.where(bad, 0.0, x)
            return uniform_filter(a2, k, mode="constant") * k * k

        Pg = np.stack([grid(x) for x in Pk])                          # (K, H, W) window sums
        Eg = np.stack([grid(x) for x in Ek])
        t_k = np.array(t_k, float)
        inwin = np.nonzero(t_k <= s["persist_days"])[0]
        S = np.full((H, W), -np.inf)
        S_all_max = np.full((H, W), -np.inf)
        best = np.full((H, W), np.nan)
        loss = np.full((H, W), np.nan)
        loo_used = len(inwin) >= 2
        for hi, h in enumerate(s["stages"][sk]["half_days"]):
            w = template(t_k, s["hold_days"], h)[:, None, None]
            C, E = w * Pg, w * w * Eg
            num, den = C.sum(0), E.sum(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_all = num / np.sqrt(den)
                s_h = s_all
                if loo_used:
                    kk = inwin[np.argmax(C[inwin], axis=0)]                  # strongest obs in the persistence window
                    ck = np.take_along_axis(C, kk[None], 0)[0]
                    ekk = np.take_along_axis(E, kk[None], 0)[0]
                    s_h = np.minimum(s_all, (num - ck) / np.sqrt(np.maximum(den - ekk, 1e-30)))
                upd = valid & (s_h > S)
                S = np.where(upd, s_h, S)
                best = np.where(upd, -1.0 if h is None else float(h), best)
                loss = np.where(upd, num / den, loss)
                S_all_max = np.maximum(S_all_max, np.where(valid, s_all, -np.inf))
        valid &= np.isfinite(S)
        reason[mask & ~valid & (reason == 0)] = 6
        lay = {}
        for name, arr in (("predicted", pred1), ("observed", obs1)):
            a2 = np.full((H, W), np.nan)
            a2[ys, xs] = arr
            lay[name] = a2
        lay["pred_std"] = np.where(valid, self.mdl["estd"][st], np.nan)
        lay["loss"] = np.where(valid, loss, np.nan)
        lay["half_days_best"] = np.where(valid, best, np.nan)
        lay["S_all"] = np.where(valid, S_all_max, np.nan)
        out.update(valid=valid, S=np.where(valid, S, np.nan), layers=lay, stratum=st, t_post=t_k.tolist(),
                   n_persist=len(inwin), loo_used=loo_used)
        return out

    # ------------------------------------------------------------------------------- damage --
    def cold_pool(self, field_id):
        if field_id not in self._cp:
            row = self.store.fields[self.store.fields.field_id == field_id].iloc[0]
            self._cp[field_id] = cold_pool_index(self.cfg, row, self.cfg["frost_evidence"]["pool_radius_m"])
        return self._cp[field_id]

    def damage(self, field_id, night):
        s = self.s
        r = self._evidence(field_id, night)
        night = r["night"]
        f = self.store.field(field_id)
        dates, mask = f["dates"], f["mask"]
        H, W = mask.shape
        valid = r["valid"]
        score = np.full((H, W), np.nan)
        score_all = np.full((H, W), np.nan)
        if valid.any():
            score = self.cal.score(field_id, r["stage"], r["S"])
            score_all = self.cal.score(field_id, r["stage"], r["layers"]["S_all"])
            r["reason"][valid & ~np.isfinite(score)] = 6
            valid = r["reason"] == 0
        score = np.where(valid, score, np.nan).astype("float32")
        geo = r["geo"] or {}
        pi = geo.get("pre")
        posts = geo.get("posts", [])
        qi = posts[0] if posts else None
        nxt = posts[1] if len(posts) > 1 else None
        iso = lambda i: dates[i].isoformat() if i is not None else None
        days = lambda i: (dates[i] - night).days if i is not None else None
        emerg, peak, _ = self.growth(field_id)
        nmeta = self.store.night_meta(field_id, night)
        thr = self.score_threshold
        dmg = valid & (score >= thr)
        nv = max(int(valid.sum()), 1)
        lay = r["layers"]
        hb = lay.get("half_days_best")
        if dmg.sum():
            vals, cnt = np.unique(hb[dmg], return_counts=True)
            h_mode = float(vals[cnt.argmax()])
        else:
            h_mode = np.nan
        share = float(dmg.sum() / nv) if valid.any() else np.nan
        rejected = float((valid & (score_all >= thr) & (score < thr)).sum() / nv) if valid.any() else np.nan
        if not valid.any():
            trajectory = None
        elif share >= s["field_flag_share"]:
            trajectory = ("drop, no recovery in the window" if h_mode == -1 else
                          f"drop, recovering (half-life ~{h_mode:g} d)")
        elif rejected >= s["field_flag_share"]:
            trajectory = "single-date dip, not confirmed by the next obs (noise)"
        else:
            trajectory = "no anomaly"
        cp = self.cold_pool(field_id)
        z = -score
        fe = self.cfg["frost_evidence"]
        p_frost, att = attribute(self.cfg, fe, nmeta, cp, mask, z, valid, -thr)
        stage_name = s["stages"][r["stage"]]["name"] if r["stage"] is not None else None
        meta = dict(field_id=field_id, night=night.isoformat(), method=NAME,
                    pre_date=iso(pi), post_date=iso(qi), next_date=iso(nxt),
                    gap_days_before=-days(pi) if pi is not None else None,
                    gap_days_after=days(qi), next_days_after=days(nxt),
                    growth_stage="growth" if valid.any() else (
                        REASONS[int(np.bincount(r["reason"][mask]).argmax())] if mask.any() else None),
                    emergence=emerg and emerg.isoformat(), peak=peak and peak.isoformat(), **nmeta,
                    crop_stage=stage_name, pdays=r["pdays"], n_post=len(posts),
                    n_post_persist=r.get("n_persist"), persistence_check=r.get("loo_used"),
                    half_days_mode=h_mode, trajectory=trajectory, rejected_single_date_share=rejected, **att)
        meta["share_damaged"] = share
        f32 = lambda a: np.where(valid, a, np.nan).astype("float32")
        layers = {"z": z.astype("float32"), "reason": r["reason"], "pre_idx": pi, "post_idx": qi, "next_idx": nxt,
                  "p_frost": p_frost.astype("float32"), "cold_pool": cp}
        if lay:
            layers.update(predicted=f32(lay["predicted"]), pred_std=f32(lay["pred_std"]),
                          observed=f32(lay["observed"]), loss=f32(lay["loss"]),
                          half_days_best=f32(lay["half_days_best"]), S=f32(r["S"]))
        else:
            nanl = np.full((H, W), np.nan, "float32")
            layers.update(predicted=nanl, pred_std=nanl, observed=nanl)
        return {"score": score, "valid": valid, "meta": meta, "curve": None, "layers": layers}


class Calibration:
    """Leave-fold-out frost-free null: per (fold, stage) median / MAD of the pixel statistic S, then the pooled
    null distribution of the field's 90th percentile of the standardised S."""

    def __init__(self, raw, s):
        self.s = s
        off = np.concatenate([[0], np.cumsum(raw["n_px"])])
        folds, stages, q = raw["fold"], raw["stage"], raw["q"]
        nst = len(s["stages"])
        self.norm, self.null = {}, {}
        for f in range(s["folds"]):
            for k in range(nst):
                ids = np.nonzero((folds != f) & (stages == k))[0]
                if len(ids) < s["cal_min_n"]:
                    for w in range(1, nst):
                        ids = np.nonzero((folds != f) & (np.abs(stages - k) <= w))[0]
                        if len(ids) >= s["cal_min_n"]:
                            break
                x = np.concatenate([raw["px"][off[i]:off[i + 1]] for i in ids])
                med = float(np.median(x))
                self.norm[(f, k)] = (med, float(1.4826 * np.median(np.abs(x - med))))
            oth = np.nonzero(folds != f)[0]
            self.null[f] = np.sort([(q[i] - self.norm[(f, int(stages[i]))][0]) / self.norm[(f, int(stages[i]))][1]
                                    for i in oth])
        self.info = dict(n_intervals=len(q), by_stage=np.bincount(stages, minlength=nst).tolist())

    def score(self, field_id, stage, S):
        f = fa.fold_of(field_id, self.s["folds"])
        med, mad = self.norm[(f, stage)]
        a = self.null[f]
        p = (np.searchsorted(a, (S - med) / mad, side="left") + 0.5) / (len(a) + 1)
        return norm.ppf(p)
