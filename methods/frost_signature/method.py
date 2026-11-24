"""frost_signature: multi-index, recovery-shaped frost damage evidence (Sentinel-2 only); weather and
terrain only for attribution (as frost_evidence).

Per pixel and night:
  1. Expected value of each index at the clear obs after the night: GP level at the last clear obs before
     (field hyperparameters per index) + expected change over the gap from frost-free intervals of other
     fields (one GrowthReference per index).  Standardised residuals u = (obs - expected) / predictive sd.
       EVI2  canopy (red, NIR)           NDMI  leaf water (B8A, B11)
       NDRE  chlorophyll (B8A, B05)      PSRI  browning / senescence ((B04 - B02) / B06)
  2. Frost fingerprint in index space, per pixel: linear spectral mixing of the pixel's green fraction g into
     frost-killed material (method-owned endmembers from non-lockbox fields: green canopy, desiccated haulm,
     bare soil), derivative per unit canopy loss at the last clear obs before.  Haze fingerprint: the median
     band change of QA haze-dropped field-dates against their neighbouring usable dates (2024 smoke days).
  3. Per post obs: matched filter for the frost fingerprint in the whitened residual space (correlation
     between the indices' residuals on frost-free intervals) with the haze fingerprint projected out
     (orthogonal subspace projection, Harsanyi & Chang 1994).
  4. Recovery: canopy loss follows A * 0.5 ** (t / h) over the post obs (t days after the night);
     T = max over h in half_days of the combined statistic (likelihood ratio maximised over the speed).
  5. Spatial mean of T over smooth_px, empirical leave-fold-out calibration (frost_evidence.Calibrator);
     z = Phi^-1(rank); damaged if z <= z_damaged.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from data.common import cache_dir
from data.store import FieldStore
from methods.evi2_gpr import method as base
from methods.frost_evidence import method as fe_mod
from validation.cases import is_holdout

NAME = "frost_signature"
REASONS = dict(base.REASONS)
INDEX_NAMES = ["evi2", "ndmi", "ndre", "psri"]
SOIL_EVI2 = 0.10


def indices(r):
    """r: reflectance with the 10 bands (data.s2bands.BANDS) on axis 0 -> (4, ...) EVI2, NDMI, NDRE, PSRI."""
    b02, b04, b05, b06, b08, b8a, b11 = r[0], r[2], r[3], r[4], r[6], r[7], r[8]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([2.5 * (b08 - b04) / (b08 + 2.4 * b04 + 1.0),
                         (b8a - b11) / (b8a + b11),
                         (b8a - b05) / (b8a + b05),
                         (b04 - b02) / b06])


class _IndexView:
    """Store whose field()['evi2'] is replaced by index k (to reuse evi2_gpr's reference table)."""

    def __init__(self, store, k):
        self._s, self._k = store, k

    def __getattr__(self, a):
        return getattr(self._s, a)

    def field(self, fid):
        f = dict(self._s.field(fid))
        f["evi2"] = indices(np.moveaxis(self._s.bands(fid), 1, 0))[self._k]
        return f


def _raw_bands(cfg, fid):
    z = np.load(cache_dir(cfg, "s2bands") / f"{fid}.npz")
    r = z["refl"].astype("float32") / 10000.0
    r[r == 0] = np.nan
    return [dt.date.fromisoformat(str(d)) for d in z["dates"]], r


def spectra(cfg, store):
    """Method-owned endmembers and haze direction from non-lockbox fields (cached in cache/)."""
    key = fe_mod._cfg_key(cfg)
    fp = cache_dir(cfg) / f"frost_signature_spectra_{store.fips}_{key}.json"
    if fp.exists():
        return {k: np.array(v) for k, v in json.loads(fp.read_text()).items()}
    rc = cfg["reference"]
    ids = [i for i in store.field_ids() if not is_holdout(i, rc["lockbox_share"])]
    green, dead, soil = [], [], []
    for fid in ids:
        f, r = store.field(fid), store.bands(fid)
        ev, m = f["evi2"], f["mask"]
        med = np.array([np.nanmedian(e[m]) if np.isfinite(e[m]).any() else np.nan for e in ev])
        pk = int(np.nanargmax(med)) if np.isfinite(med).any() else None
        for t, d in enumerate(f["dates"]):
            ok = m & np.isfinite(ev[t])
            if d.month == 7:
                green.append(r[t][:, ok & (ev[t] >= 0.85)].T)
            if d.month == 5:
                soil.append(r[t][:, ok & (ev[t] <= 0.12)].T)
            if pk is not None and t > pk and d.month >= 9 and med[pk] >= 0.7:
                dead.append(r[t][:, ok & (ev[t] <= 0.30)].T)
    med_spec = lambda parts: np.nanmedian(np.concatenate([p for p in parts if len(p)]), 0)
    # haze: QA haze-dropped field-dates vs the mean of the nearest usable dates before and after (<= 15 d)
    qa = pd.read_parquet(cache_dir(cfg) / f"s2_qa_{store.fips}.parquet")
    qa = qa[qa.field_id.isin(ids)]
    diffs = []
    for fid, g in qa.groupby("field_id"):
        hz = g[g.drop_reason == "haze"]
        if not len(hz):
            continue
        dates, r = _raw_bands(cfg, fid)
        m = store.field(fid)["mask"]
        use = {d for d in pd.to_datetime(g[g.usable].date).dt.date}
        fmed = lambda t: np.nanmedian(r[t][:, m], 1)
        for d in pd.to_datetime(hz.date).dt.date:
            if d not in dates:
                continue
            t = dates.index(d)
            bef = [i for i in range(t) if dates[i] in use and (d - dates[i]).days <= 15]
            aft = [i for i in range(t + 1, len(dates)) if dates[i] in use and (dates[i] - d).days <= 15]
            if bef and aft:
                diffs.append(fmed(t) - 0.5 * (fmed(bef[-1]) + fmed(aft[0])))
    hz = np.nanmedian(np.array(diffs), 0)
    out = dict(green=med_spec(green), dead=med_spec(dead), soil=med_spec(soil), haze=hz / hz[0],
               n_haze_cases=np.array([len(diffs)]))
    fp.write_text(json.dumps({k: np.asarray(v).round(6).tolist() for k, v in out.items()}))
    return out


_REFS = {}


def index_references(cfg, store):
    key = fe_mod._cfg_key(cfg)
    if key not in _REFS:
        if not isinstance(store, FieldStore):
            raise RuntimeError("frost_signature references must first be built from a FieldStore")
        _REFS[key] = [base.GrowthReference(base.reference_table(cfg, _IndexView(store, k)), cfg["reference"])
                      for k in range(len(INDEX_NAMES))]
    return _REFS[key]


class Method(base.Method):
    name = NAME

    def __init__(self, cfg, store):
        cfg = copy.deepcopy(cfg)
        cfg["gpr"].update(hyper="field", mean="reference")
        super().__init__(cfg, store)
        self.fe = cfg["frost_signature"]          # same key names as frost_evidence where shared
        self.score_threshold = -self.fe["z_damaged"]
        self.refs = index_references(cfg, store)
        self.sp = spectra(cfg, store)
        ek = self.fe["dead_share"] * self.sp["dead"] + (1 - self.fe["dead_share"]) * self.sp["soil"]
        self.d_frost = ek - self.sp["green"]                         # band change per unit canopy loss of g = 1
        self.e_full = float(indices(self.sp["green"][:, None])[0, 0])
        self._cp = {}
        self.sigma = None
        self.sigma, self.cal = self._calibration(store)

    cold_pool = fe_mod.Method.cold_pool

    def _index_stack(self, field_id):
        return indices(np.moveaxis(self.store.bands(field_id), 1, 0).astype("float64"))

    # ------------------------------------------------------------------------------ evidence --
    def _evidence(self, field_id, night):
        cfg, g, fe = self.cfg, self.cfg["gpr"], self.fe
        night = dt.date.fromisoformat(str(night))
        f = self.store.field(field_id)
        dates, mask = f["dates"], f["mask"]
        X = self._index_stack(field_id)                              # (K, T, H, W)
        B = self.store.bands(field_id)                               # (T, 10, H, W)
        K, T, H, W = X.shape
        emerg, peak, _ = self.growth(field_id)
        reason = np.where(mask, 0, 1).astype("uint8")
        if not (emerg is not None and peak is not None and emerg <= night <= peak):
            reason[mask] = 2
        stage_txt = ("pre-emergence" if emerg is None or night < emerg else
                     "growth" if peak is None or night <= peak else "post-peak")
        clear = np.isfinite(X).all(0) & mask[None]
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
        reason[(reason == 0) & ((n_em < g["min_clear_obs_since_emergence"])
                                | (n_win < g.get("min_clear_obs_before", 0)) | ~has_b)] = 3
        reason[(reason == 0) & (-t_days[last_b] > g["max_gap_before_days"])] = 4
        reason[(reason == 0) & (~has_a | (t_days[first_a] > g["max_gap_after_days"]))] = 5
        # post obs used: first_a and later clear obs within persist_days and the reference's max gap
        ok_post = (ca & (t_days <= fe["persist_days"])[:, None, None]
                   & ((abs_days[:, None, None] - abs_days[last_b][None]) <= cfg["reference"]["max_gap_days"]))
        ok_post |= ca & (np.arange(T)[:, None, None] == first_a[None])
        rank = np.cumsum(ok_post, 0)
        posts = np.full((fe["n_post"], H, W), -1)
        for j in range(fe["n_post"]):
            sel = ok_post & (rank == j + 1)
            posts[j] = np.where(sel.any(0), np.argmax(sel, 0), -1)

        Tstat = np.full((H, W), np.nan)
        loss = np.full((H, W), np.nan)
        half = np.full((H, W), np.nan)
        s_evi2 = np.full((H, W), np.nan)
        u_first = np.full((K, H, W), np.nan)
        pred1 = np.full((H, W), np.nan); std1 = np.full((H, W), np.nan); obs1 = np.full((H, W), np.nan)
        npost = np.zeros((H, W), "int8")
        ys, xs = np.nonzero(reason == 0)
        nb = int(before.sum())
        hps = [self._hyper_params((field_id, k), cb, X[k], abs_days, nb) for k in range(K)] if len(ys) else []
        if len(ys) and any(h is None for h in hps):
            reason[ys, xs] = 6
            ys = ys[:0]
        if len(ys):
            meds = [base.field_median_series(dict(f, evi2=X[k]), cfg["reference"]["min_clear_share"])
                    for k in range(K)]
            key = np.concatenate([cb[:, ys, xs].T.astype(int), posts[:, ys, xs].T], 1)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.ravel()
            hl = np.array(fe["half_days"], float)
            for gi, kk in enumerate(uniq):
                pos = np.nonzero(inv == gi)[0]
                gy, gx = ys[pos], xs[pos]
                idx = np.nonzero(kk[:T])[0]
                pj = [int(i) for i in kk[T:] if i >= 0]
                t, il = abs_days[idx], idx[-1]
                tl = t[-1]
                doy = dates[il].timetuple().tm_yday
                u = np.zeros((K, len(pj), len(pos)))
                sd = np.zeros((K, len(pj), len(pos)))
                bad = False
                for k in range(K):
                    amp, ls, noise = hps[k]
                    yv = X[k][idx][:, gy, gx]
                    mu = yv.mean(0)
                    try:
                        low = base._chol(t, amp, ls, noise)
                    except np.linalg.LinAlgError:
                        bad = True
                        break
                    kq = base._rbf(np.array([tl]), t, amp, ls)
                    s = mu + (kq @ base.cho_solve((low, True), yv - mu))[0]
                    vs = max(amp - float((kq @ base.cho_solve((low, True), kq.T))[0, 0]), 0.0)
                    slope = base.field_slope(dates, meds[k], il)
                    for j, ia in enumerate(pj):
                        Xf = np.column_stack([s, np.full_like(s, slope), np.full_like(s, abs_days[ia] - tl),
                                              np.full_like(s, doy)])
                        d, vd = self.refs[k].predict(field_id, Xf)
                        sd[k, j] = np.sqrt(vs + vd + noise)
                        u[k, j] = (X[k][ia, gy, gx] - s - d) / sd[k, j]
                        if k == 0 and j == 0:
                            pred1[gy, gx], std1[gy, gx], obs1[gy, gx] = s + d, sd[k, j], X[k][ia, gy, gx]
                    if k == 0:
                        s_evi2[gy, gx] = s
                if bad:
                    reason[gy, gx] = 6
                    continue
                u_first[:, gy, gx] = u[:, 0]
                npost[gy, gx] = len(pj)
                if self.sigma is None:
                    continue
                # fingerprints at the last clear obs before (per unit canopy loss / per 0.01 blue haze)
                rho0 = B[il][:, gy, gx].astype("float64")
                i0 = indices(rho0)
                gfr = np.clip((i0[0] - SOIL_EVI2) / (self.e_full - SOIL_EVI2), 0, 1)
                sF = (indices(rho0 + 0.1 * gfr * self.d_frost[:, None]) - i0) / 0.1
                sH = (indices(rho0 + 0.01 * self.sp["haze"][:, None]) - i0) / 0.01
                st = np.digitize(s_evi2[gy, gx], fe["stage_bins"])
                num = np.zeros((len(pj), len(pos)))
                den = np.zeros((len(pj), len(pos)))
                for sti in np.unique(st):
                    q = st == sti
                    Wm = np.linalg.inv(np.linalg.cholesky(self.sigma[int(sti)]))
                    for j in range(len(pj)):
                        wa = Wm @ (sF[:, q] / sd[:, j][:, q])
                        wh = Wm @ (sH[:, q] / sd[:, j][:, q])
                        wu = Wm @ u[:, j][:, q]
                        hh = (wh * wh).sum(0)
                        pa = wa - wh * (wh * wa).sum(0) / hh
                        num[j, q] = (pa * wu).sum(0)
                        den[j, q] = (pa * wa).sum(0)
                tj = np.array([t_days[i] for i in pj])
                D = 0.5 ** (tj[None, :] / hl[:, None])                    # (n_half, n_post)
                Th = (D @ num) / np.sqrt(np.maximum(D ** 2 @ den, 1e-12))  # (n_half, m)
                Ah = (D @ num) / np.maximum(D ** 2 @ den, 1e-12)
                b = np.argmax(Th, 0)
                c = np.arange(len(pos))
                Tstat[gy, gx], loss[gy, gx], half[gy, gx] = Th[b, c], Ah[b, c], hl[b]
        valid = reason == 0
        if self.sigma is not None:
            bad = valid & ~np.isfinite(Tstat)
            reason[bad] = 6
            valid = reason == 0
        k = fe["smooth_px"]
        cnt = uniform_filter(valid.astype(float), k, mode="constant")
        T_s = np.where(valid, uniform_filter(np.where(valid, np.nan_to_num(Tstat), 0.0), k, mode="constant")
                       / np.maximum(cnt, 1e-9), np.nan)
        stage = np.digitize(np.nan_to_num(s_evi2, nan=0.0), fe["stage_bins"]).astype("int8")
        return dict(valid=valid, reason=reason, z_smooth=-T_s, T=Tstat, loss=loss, half=half, u_first=u_first,
                    stage=stage, npost=npost, pred1=pred1, std1=std1, obs1=obs1, last_b=last_b,
                    first_a=first_a, clear=clear, t_days=t_days, stage_txt=stage_txt,
                    emerg=emerg, peak=peak, night=night)

    # --------------------------------------------------------------------------- calibration --
    def _calibration(self, store):
        """Pass 1: residual correlation of the indices per stage; pass 2: calibration of T (cached)."""
        key = fe_mod._cfg_key(self.cfg)
        fp_s = cache_dir(self.cfg) / f"frost_signature_sigma_{store.fips}_{key}.json"
        fp_c = cache_dir(self.cfg) / f"frost_signature_cal_{store.fips}_{key}.parquet"
        if not (fp_s.exists() and fp_c.exists()) and not isinstance(store, FieldStore):
            raise RuntimeError("frost_signature calibration must first be built from a FieldStore")
        if not fp_s.exists():
            rng = np.random.default_rng(0)
            rows = []
            for fid, d0 in fe_mod.calibration_intervals(self, store):
                r = self._evidence(fid, d0)
                v = np.nonzero(r["valid"].ravel())[0]
                v = rng.choice(v, min(self.fe["cal_px_per_case"], len(v)), replace=False)
                rows.append(np.column_stack([r["stage"].ravel()[v], r["u_first"].reshape(len(INDEX_NAMES), -1)[:, v].T]))
            a = np.concatenate(rows)
            sig = {}
            for st in range(len(self.fe["stage_bins"]) + 1):
                uu = np.clip(a[a[:, 0] == st, 1:], -6, 6)
                sig[st] = np.corrcoef(uu.T).tolist()
            fp_s.write_text(json.dumps(sig))
        self.sigma = {int(k): np.array(v) for k, v in json.loads(fp_s.read_text()).items()}
        if not fp_c.exists():
            tmp = fp_c.with_name(f"{fp_c.name}.{os.getpid()}.tmp")
            fe_mod.calibration_table(self, store).to_parquet(tmp)
            os.replace(tmp, fp_c)
        cal = fe_mod.Calibrator(pd.read_parquet(fp_c), self.cfg["reference"]["folds"], self.fe["cal_min_n"]) \
            if self.fe["calibrate"] else None
        return self.sigma, cal

    # -------------------------------------------------------------------------------- damage --
    def damage(self, field_id, night):
        fe = self.fe
        r = self._evidence(field_id, night)
        night = r["night"]
        f = self.store.field(field_id)
        dates, mask = f["dates"], f["mask"]
        valid = r["valid"]
        z = (self.cal(field_id, r["z_smooth"], r["stage"], r["npost"]) if self.cal is not None
             else r["z_smooth"].copy())
        r["reason"][valid & ~np.isfinite(z)] = 6
        valid = r["reason"] == 0
        z = np.where(valid, z, np.nan)
        score = np.where(valid, np.maximum(0.0, -z), np.nan).astype("float32")
        nmeta = self.store.night_meta(field_id, night)
        cp = self.cold_pool(field_id)
        p_frost, att = fe_mod.attribute(self.cfg, fe, nmeta, cp, mask, z, valid, fe["z_damaged"])
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
        dmg = valid & (z <= fe["z_damaged"])
        meta = dict(field_id=field_id, night=night.isoformat(), method=NAME,
                    pre_date=iso(pi), post_date=iso(qi), next_date=iso(nxt),
                    gap_days_before=-days(pi) if pi is not None else None,
                    gap_days_after=days(qi), next_days_after=days(nxt),
                    growth_stage=r["stage_txt"], emergence=emerg and emerg.isoformat(),
                    peak=peak and peak.isoformat(), **nmeta, **att,
                    loss_median_damaged=float(np.median(r["loss"][dmg])) if dmg.any() else np.nan,
                    half_days_median_damaged=float(np.median(r["half"][dmg])) if dmg.any() else np.nan)
        f32 = lambda a: np.where(valid, a, np.nan).astype("float32")
        return {"score": score, "valid": valid, "meta": meta, "curve": None,
                "layers": {"z": f32(z), "predicted": f32(r["pred1"]), "pred_std": f32(r["std1"]),
                           "observed": f32(r["obs1"]), "T": f32(r["T"]), "loss": f32(r["loss"]),
                           "half_days": f32(r["half"]), "n_post": r["npost"],
                           **{f"u_{n}": f32(r["u_first"][i]) for i, n in enumerate(INDEX_NAMES)},
                           "p_frost": p_frost.astype("float32"), "cold_pool": cp, "reason": r["reason"],
                           "pre_idx": pi, "post_idx": qi, "next_idx": nxt}}
