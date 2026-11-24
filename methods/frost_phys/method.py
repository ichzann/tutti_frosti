"""frost_phys: physics-based spectral frost-damage detector in Sentinel-2 band space (no index, no GP).

Damage evidence (Sentinel-2 surface reflectance only; weather and terrain only for attribution):
  1. Band residuals.  For each pixel and each clear obs j after the night (up to n_post within persist_days):
       r_j = rho(post_j) - rho(pre) - Delta(e0, slope, gap_j, doy) - mu(stage, gap)                (10 bands)
     rho(pre) = the pixel's last clear obs before the night; Delta = expected frost-free 10-band change over the
     gap: mean of the k nearest frost-free field-median intervals of other fields' folds (features: pixel EVI2 at
     pre, field EVI2 slope before, gap, day of year); mu = mean residual of frost-free intervals (bias correction).
  2. Noise model.  Residual covariance Sigma(stage, gap bin) of the core of frost-free residuals (iteratively
     trimmed, consistency-rescaled; fixed-weight shrinkage to the diagonal); work in the whitened space.
     Compound-Gaussian local scale: the residual energy outside the frost + haze subspaces, pooled over the post
     obs and a scale_px window, relative to its frost-free median, inflates the noise (factor >= 1 only), so
     changes the frost model does not explain (cloud remnants, management) lose weight.
  3. Signal and nuisance subspaces (physics, method-owned, never from the harness):
       frost      killing a share L of the green canopy fraction g turns it into frost-killed material
                  E_m = m * desiccated haulm + (1 - m) * bare soil:  r = L g (E_m - E_green)  (linear mixing).
                  Endmembers = medians of non-lockbox field pixels: green = July EVI2 >= 0.85, haulm = September
                  0.15 <= EVI2 <= 0.30 after a peak >= 0.7, soil = May EVI2 <= 0.12.  m scanned on a grid.
       haze       residual path reflectance a (lambda / 490 nm)^-alpha, alpha in angstrom_range (Angstrom law): the
                  leading SVD components of that family are projected out (orthogonal subspace projection,
                  Harsanyi & Chang 1994), so any additive aerosol / thin-cloud residual has no effect.
  4. Detection.  Per post obs the standardised matched-filter output y_j = f'P r_j / |P f| (P: haze-free whitened
     space); over the post obs a GLS fit of a recovering loss A * 0.5^(t_j / h) * |P f|_j with the frost-free
     covariance R of (y_1, y_2, y_3):  T = max_{m, h} d'R^-1 y / sqrt(d'R^-1 d)  (one-sided: A > 0).
  5. Spatial mean of T over smooth_px x smooth_px valid pixels; empirical calibration z = Phi^-1(rank) among
     frost-free pixel-intervals of other fields (leave-fold-out; per stage, first-post gap bin, number of post obs).
  6. Decision: damaged if z <= threshold; the threshold controls the field-level false alarm rate: the largest z at
     which the field-cluster bootstrap 95 % upper bound of the share of frost-free calibration field-intervals with
     >= field_flag_share of their valid pixels flagged is <= field_alpha (and <= z_damaged).
Frost-free calibration intervals follow the documented null definition (README / validation: consecutive usable
acquisitions in the growth phase, every night between them without trigger and Tmin >= 6 C), non-lockbox fields only.
Validity: growth phase as evi2_gpr, extended causally (the season peak may move before the night when damage lowers
later obs; a night still counts if the smoothed series up to it is at its maximum on its last obs).
Predictive EVI2 layers (predicted / pred_std / observed): EVI2 of rho(pre) + Delta, sd = q95(|frost-free EVI2
residual|) / 1.96 per (stage, gap bin).
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter
from scipy.stats import chi2, norm
from sklearn.neighbors import NearestNeighbors

from data.common import cache_dir
from data.store import FieldStore
from methods.evi2_gpr import method as base
from methods.frost_evidence import method as fe_mod
from validation.cases import is_holdout

NAME = "frost_phys"
REASONS = dict(base.REASONS)
REASONS[6] = "statistic failed"
SETTINGS = Path(__file__).with_name("settings.yaml")
WAVE_NM = np.array([492., 560., 665., 704., 740., 783., 833., 865., 1614., 2202.])  # S2 B02..B12 centres
NB = len(WAVE_NM)


def evi2(r):
    """r: (10, ...) reflectance -> EVI2 (...)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.5 * (r[6] - r[2]) / (r[6] + 2.4 * r[2] + 1.0)


def _key(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _frost_free_set(store, min_tmin):
    n = store.nights
    cold = n.trigger | (n.tmin_c < min_tmin)
    return set(zip(n.cell_id[cold], n.night[cold]))


def _fold(field_id, n):
    return base._fold(field_id, n)


# ------------------------------------------------------------------------------ physics --
def endmembers(cfg, store, fp):
    """Method-owned endmember spectra (median over fields of per-field medians), non-lockbox fields."""
    if fp.exists():
        return {k: np.array(v) for k, v in json.loads(fp.read_text()).items()}
    pc = cfg["frost_phys"]
    parts = {"green": [], "haulm": [], "soil": []}
    for fid in store.field_ids():
        if is_holdout(fid, pc["lockbox_share"]):
            continue
        f, B = store.field(fid), store.bands(fid)
        m = f["mask"]
        ev = evi2(np.moveaxis(B, 1, 0))
        med = np.array([np.nanmedian(e[m]) if np.isfinite(e[m]).any() else np.nan for e in ev])
        pk = int(np.nanargmax(med)) if np.isfinite(med).any() else None
        for t, d in enumerate(f["dates"]):
            ok = m & np.isfinite(B[t]).all(0)
            sel = {"green": ok & (ev[t] >= 0.85) if d.month == 7 else None,
                   "soil": ok & (ev[t] <= 0.12) if d.month == 5 else None,
                   "haulm": (ok & (ev[t] >= 0.15) & (ev[t] <= 0.30)
                             if pk is not None and t > pk and d.month == 9 and med[pk] >= 0.7 else None)}
            for k, s in sel.items():
                if s is not None and s.sum() >= 50:
                    parts[k].append(np.median(B[t][:, s], 1))
    out = {k: np.median(np.array(v), 0) for k, v in parts.items()}
    out.update({f"n_{k}": [len(v)] for k, v in parts.items()})
    fp.write_text(json.dumps({k: np.asarray(v).round(6).tolist() for k, v in out.items()}))
    return {k: np.array(v) for k, v in json.loads(fp.read_text()).items()}


def haze_basis(pc):
    """Orthonormal basis (10, q) of the Angstrom family a (lambda / 490)^-alpha, alpha in angstrom_range."""
    al = np.linspace(*pc["angstrom_range"], 26)
    F = np.stack([(WAVE_NM / 490.0) ** -a for a in al], 1)
    U, s, _ = np.linalg.svd(F, full_matrices=False)
    return U[:, :pc["haze_rank"]], float((s[:pc["haze_rank"]] ** 2).sum() / (s ** 2).sum())


# --------------------------------------------------------------------- change reference --
def _field_series(store, fid, min_share):
    """Field-median band series (T, 10) and EVI2 (T,), NaN where < min_share of the field's pixels are clear."""
    f, B = store.field(fid), store.bands(fid)
    m = f["mask"]
    T = len(f["dates"])
    med = np.full((T, NB), np.nan)
    for t in range(T):
        x = B[t][:, m]
        ok = np.isfinite(x).all(0)
        if ok.mean() >= min_share:
            med[t] = np.median(x[:, ok], 1)
    return f["dates"], med, evi2(med.T)


def reference_table(cfg, store):
    """Frost-free field-median 10-band changes between usable acquisitions of non-lockbox fields."""
    pc = cfg["frost_phys"]
    cold = _frost_free_set(store, pc["ref_min_tmin_c"])
    rows = []
    for fid in store.field_ids():
        if is_holdout(fid, pc["lockbox_share"]):
            continue
        dates, med, ev = _field_series(store, fid, pc["ref_min_clear_share"])
        use = np.nonzero(np.isfinite(ev))[0]
        cell = store.field_cell.get(fid)
        for a in range(1, len(use) - 1):
            i, h = use[a], use[a - 1]
            slope = (ev[i] - ev[h]) / (dates[i] - dates[h]).days
            for j in range(1, pc["ref_max_step"] + 1):
                if a + j >= len(use):
                    break
                k = use[a + j]
                gap = (dates[k] - dates[i]).days
                if gap > pc["ref_max_gap_days"] or any(
                        (cell, (dates[i] + dt.timedelta(days=q)).isoformat()) in cold for q in range(gap)):
                    break
                rows.append(dict(field_id=fid, date=dates[i].isoformat(), e0=ev[i], slope=slope, gap=gap,
                                 doy=dates[i].timetuple().tm_yday,
                                 **{f"d{b}": med[k, b] - med[i, b] for b in range(NB)}))
    return pd.DataFrame(rows)


class SpectralReference:
    """Expected 10-band change: mean of the k nearest frost-free intervals in standardised
    (e0, slope, gap, doy), leave-fold-out (a field is predicted only from other folds)."""

    FEAT = ["e0", "slope", "gap", "doy"]

    def __init__(self, table, k, folds):
        self.folds = folds
        fold = table.field_id.map(lambda i: _fold(i, folds)).to_numpy()
        X = table[self.FEAT].to_numpy(float)
        Y = table[[f"d{b}" for b in range(NB)]].to_numpy(float)
        self.models = {}
        for q in range(folds):
            tr = fold != q
            mu, sd = X[tr].mean(0), X[tr].std(0)
            self.models[q] = (mu, sd, NearestNeighbors(n_neighbors=k).fit((X[tr] - mu) / sd), Y[tr])

    def predict(self, field_id, X):
        mu, sd, nn, Y = self.models[_fold(field_id, self.folds)]
        _, idx = nn.kneighbors((np.asarray(X, float) - mu) / sd)
        return Y[idx].mean(1)                                      # (n, 10)


def robust_cov(X, q=0.975, n_iter=5):
    """Mean and covariance of the core of X (iteratively trimmed at the chi2 q-quantile of the Mahalanobis
    distance), rescaled so that the median distance of all rows equals the chi2 median (consistency)."""
    p = X.shape[1]
    keep = np.ones(len(X), bool)
    for _ in range(n_iter):
        mu, C = X[keep].mean(0), np.cov(X[keep].T)
        d2 = np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(C), X - mu)
        keep = d2 <= chi2.ppf(q, p)
    return mu, C * np.median(d2) / chi2.ppf(0.5, p)


class Calibrator:
    """Leave-fold-out empirical CDF of the smoothed statistic; reference set = the most specific of
    (stage, first-post gap bin, n post obs) -> (stage, gap bin) -> (stage) with >= min_n samples."""

    def __init__(self, table, folds, min_n):
        self.folds = folds
        fold = table.field_id.map(lambda i: _fold(i, folds)).to_numpy()
        self.ref = {}
        for q in range(folds):
            t = table[fold != q]
            for keys in (["stage"], ["stage", "gapbin"], ["stage", "gapbin", "npost"]):
                for k, g in t.groupby(keys):
                    if len(g) >= min_n or keys == ["stage"]:
                        self.ref[(q,) + tuple(int(x) for x in np.atleast_1d(k))] = np.sort(g.z.to_numpy())

    def __call__(self, field_id, z, stage, gapbin, npost):
        q = _fold(field_id, self.folds)
        out = np.full(z.shape, np.nan)
        ok = np.isfinite(z)
        keys = np.stack([stage[ok], gapbin[ok], npost[ok]], 1).astype(int)
        zz = z[ok]
        res = np.full(len(zz), np.nan)
        for k in np.unique(keys, axis=0):
            sel = (keys == k).all(1)
            a = (self.ref.get((q, *k)) if (q, *k) in self.ref else self.ref.get((q, k[0], k[1]), self.ref.get((q, k[0]))))
            if a is None:
                continue
            res[sel] = norm.ppf((np.searchsorted(a, zz[sel], side="right") + 0.5) / (len(a) + 1))
        out[ok] = res
        return out


# ----------------------------------------------------------------------------- method --
_SETUP = {}


class Method(base.Method):
    name = NAME

    def __init__(self, cfg, store):
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("frost_phys", yaml.safe_load(SETTINGS.read_text()))
        cfg["gpr"].update(hyper="field", mean="constant")   # base class used for growth() only (no GP, no reference)
        super().__init__(cfg, store)               # growth(), day0; cfg gpr.mean stays as configured (unused)
        self.pc = cfg["frost_phys"]
        self._cp = {}
        self.fe = dict(self.pc)                     # attribution helper expects these keys
        key = _key({k: cfg[k] for k in ("frost_phys", "growth", "study")})
        if key not in _SETUP:
            if not isinstance(store, FieldStore):
                raise RuntimeError("frost_phys setup must first be built from a FieldStore")
            _SETUP[key] = self._setup(store, key)
        S = _SETUP[key]
        self.em, self.H, self.ref = S["em"], S["H"], S["ref"]
        self.sigma, self.evi2_sd, self.cal, self.z_thr = S["sigma"], S["evi2_sd"], S["cal"], S["z_thr"]
        self.mu, self.post_cov, self.qmed = S["mu"], S["post_cov"], S["qmed"]
        self.F = self._frost_dirs()
        self.score_threshold = -self.z_thr if self.z_thr is not None else -self.pc["z_damaged"]

    cold_pool = fe_mod.Method.cold_pool

    def _frost_dirs(self):
        g, h, s = self.em["green"], self.em["haulm"], self.em["soil"]
        return np.stack([m * h + (1 - m) * s - g for m in self.pc["dead_share_grid"]], 1)   # (10, M)

    # ------------------------------------------------------------------------ setup (cached) --
    def _setup(self, store, key):
        c = cache_dir(self.cfg)
        fx = lambda n, ext: c / f"frost_phys_{n}_{store.fips}_{key}.{ext}"
        pc = self.pc
        S = {"em": endmembers(self.cfg, store, fx("endmembers", "json")),
             "H": haze_basis(pc)[0]}
        fp = fx("reftable", "parquet")
        if not fp.exists():
            _atomic(reference_table(self.cfg, store), fp)
        S["ref"] = SpectralReference(pd.read_parquet(fp), pc["ref_k"], pc["folds"])
        # pass 1: residual mean and covariance (all post obs) and EVI2 residual scale per (stage, gap bin)
        self.em, self.H, self.ref, self.sigma, self.post_cov = S["em"], S["H"], S["ref"], None, None
        self.F = self._frost_dirs()
        fs = fx("sigma", "json")
        nst, ngb = len(pc["stage_bins"]) + 1, len(pc["gap_bins"]) + 1
        if not fs.exists():
            rng = np.random.default_rng(0)
            rows = []
            for fid, d0 in self.calibration_intervals(store):
                r = self._evidence(fid, d0)
                if "RES" not in r:
                    continue
                v = rng.choice(len(r["ys"]), min(pc["cal_px_per_case"], len(r["ys"])), replace=False)
                stg = r["stage"][r["ys"][v], r["xs"][v]]
                for j in range(pc["n_post"]):
                    rows.append(np.column_stack([stg, r["GB"][j, v], r["RES"][j][:, v].T, r["EVR"][j, v]]))
            a = np.concatenate(rows)
            a = a[np.isfinite(a).all(1)]
            out = {}
            for st in range(nst):
                for gb in range(ngb):
                    x = a[(a[:, 0] == st) & (a[:, 1] == gb)]
                    if len(x) < 2000:           # fall back to the stage
                        x = a[a[:, 0] == st]
                    mu, C = robust_cov(x[:, 2:2 + NB]) if pc["robust_cov"] else \
                        (x[:, 2:2 + NB].mean(0), np.cov(x[:, 2:2 + NB].T))
                    C = (1 - pc["cov_shrink"]) * C + pc["cov_shrink"] * np.diag(np.diag(C))
                    out[f"{st}_{gb}"] = dict(mean=mu.tolist(), cov=C.tolist(), n=len(x),
                                             evi2_sd=float(np.quantile(np.abs(x[:, -1]), 0.95) / 1.96))
            fs.write_text(json.dumps(out))
        sj = json.loads(fs.read_text())
        kk = lambda k: tuple(map(int, k.split("_")))
        S["sigma"] = {kk(k): np.array(v["cov"]) for k, v in sj.items()}
        S["mu"] = {kk(k): np.array(v["mean"]) for k, v in sj.items()}
        S["evi2_sd"] = {kk(k): v["evi2_sd"] for k, v in sj.items()}
        self.sigma, self.mu, self.evi2_sd = S["sigma"], S["mu"], S["evi2_sd"]
        # pass 1b: correlation of the standardised frost projections between post obs, per stage
        fr = fx("postcorr", "json")
        if not fr.exists():
            rng = np.random.default_rng(2)
            rows = []
            mid = self.F.shape[1] // 2
            self.qmed = None
            for fid, d0 in self.calibration_intervals(store):
                r = self._evidence(fid, d0)
                if "Y" not in r:
                    continue
                v = rng.choice(len(r["ys"]), min(pc["cal_px_per_case"], len(r["ys"])), replace=False)
                rows.append(np.column_stack([r["stage"][r["ys"][v], r["xs"][v]], r["ratio"][v],
                                             r["Y"][:, mid, v].T]))
            a = np.concatenate(rows)
            out = {}
            for st in range(nst):
                x = a[(a[:, 0] == st) & np.isfinite(a[:, 1])]
                qm = float(np.median(x[:, 1]))
                kap = np.maximum(1.0, x[:, 1] / qm) if pc["scale_adapt"] else np.ones(len(x))
                y = pd.DataFrame(x[:, 2:] / np.sqrt(kap)[:, None])
                out[str(st)] = dict(cov=y.cov(min_periods=1000).to_numpy().tolist(), qmed=qm)   # pairwise complete
            fr.write_text(json.dumps(out))
        pj = json.loads(fr.read_text())
        S["post_cov"] = {int(k): np.array(v["cov"]) for k, v in pj.items()}
        S["qmed"] = {int(k): v["qmed"] for k, v in pj.items()}
        self.qmed = S["qmed"]
        self.post_cov = S["post_cov"]
        # pass 2: calibration table of the smoothed statistic
        fc = fx("cal", "parquet")
        if not fc.exists():
            rng = np.random.default_rng(1)
            parts = []
            for fid, d0 in self.calibration_intervals(store):
                r = self._evidence(fid, d0)
                v = np.nonzero(r["valid"].ravel())[0]
                if not len(v):
                    continue
                n_valid = len(v)
                v = rng.choice(v, min(pc["cal_px_per_case"], len(v)), replace=False)
                parts.append(pd.DataFrame(dict(
                    field_id=fid, date=d0.isoformat(), n_valid=n_valid, z=-r["T_s"].ravel()[v],
                    stage=r["stage"].ravel()[v].astype("int8"), gapbin=r["gapbin"].ravel()[v].astype("int8"),
                    npost=r["npost"].ravel()[v].astype("int8"))))
            _atomic(pd.concat(parts, ignore_index=True), fc)
        tab = pd.read_parquet(fc)
        S["cal"] = Calibrator(tab, pc["folds"], pc["cal_min_n"])
        S["z_thr"] = self._field_threshold(tab, S["cal"]) if pc.get("field_alpha") else None
        return S

    def _field_threshold(self, tab, cal):
        """Largest z threshold (<= z_damaged) at which the share of frost-free calibration field-intervals flagged
        (>= field_flag_share of their sampled valid pixels at or below it) is <= field_alpha with confidence
        1 - field_delta: the (1 - delta) quantile of a field-cluster bootstrap of that share (risk control with an
        upper confidence bound, as Bates et al. 2021, J. ACM 68:43).  Calibrated z of each interval is leave-fold-out."""
        pc = self.pc
        rows = []
        for (fid, d), g in tab.groupby(["field_id", "date"]):
            if g.n_valid.iloc[0] < 100:
                continue
            z = cal(fid, g.z.to_numpy(), g.stage.to_numpy(), g.gapbin.to_numpy(), g.npost.to_numpy())
            rows.append((fid, np.quantile(z[np.isfinite(z)], pc["field_flag_share"])))
        q = pd.DataFrame(rows, columns=["fid", "q10"])
        codes, fids = pd.factorize(q.fid)
        rng = np.random.default_rng(0)
        W = rng.multinomial(len(fids), np.full(len(fids), 1 / len(fids)), size=pc["field_boot"])  # (B, n_fields)
        n_i = np.bincount(codes, minlength=len(fids))
        thr_grid = np.arange(pc["z_damaged"], -4.0, -0.005)
        thr = thr_grid[-1]
        for t in thr_grid:
            f_i = np.bincount(codes, weights=(q.q10.to_numpy() <= t).astype(float), minlength=len(fids))
            ucb = np.quantile((W @ f_i) / (W @ n_i), 1 - pc["field_delta"])
            if ucb <= pc["field_alpha"]:
                thr = t
                break
        self._thr_info = dict(n_intervals=len(q), n_fields=len(fids), thr=float(thr),
                              share_at_thr=float((q.q10 <= thr).mean()))
        return float(thr)

    def calibration_intervals(self, store):
        """(field_id, night) of frost-free intervals of non-lockbox fields, following the documented null
        definition (README, validation): consecutive usable acquisitions (a_k, a_k+1) with a_k in the field's growth
        phase [emergence, peak), every night of [a_k, a_k+1) in the field's RTMA cell without trigger and with
        Tmin >= ref_min_tmin_c; night drawn uniformly inside the interval (seeded by field and date)."""
        pc = self.pc
        cold = _frost_free_set(store, pc["ref_min_tmin_c"])
        for fid in store.field_ids():
            if is_holdout(fid, pc["lockbox_share"]):
                continue
            f = store.field(fid)
            emerg, peak, _ = self.growth(fid)
            if emerg is None or peak is None:
                continue
            med = base.field_median_series(f, 0.5)
            use = np.nonzero(np.isfinite(med))[0]
            cell = store.field_cell.get(fid)
            for i, j in zip(use[:-1], use[1:]):
                d0, d1 = f["dates"][i], f["dates"][j]
                if not (emerg <= d0 < peak):
                    continue
                nights = [d0 + dt.timedelta(days=k) for k in range((d1 - d0).days)]
                if any((cell, x.isoformat()) in cold for x in nights):
                    continue
                h = int(hashlib.md5(f"night:{fid}:{d0}".encode()).hexdigest()[:8], 16)
                yield fid, nights[h % len(nights)]

    def in_phase(self, field_id, night):
        """Growth phase as evi2_gpr (emergence <= night <= peak), or, when the season peak falls before the night,
        if the smoothed field-median EVI2 of the obs up to the night is still at its maximum on the last of them
        (causal: damage after the night can move the season peak before it, but cannot change this)."""
        emerg, peak, _ = self.growth(field_id)
        if emerg is None or night < emerg:
            return False
        if peak is not None and night <= peak:
            return True
        f = self.store.field(field_id)
        m = f["mask"]
        med = np.array([np.nanmedian(e[m]) if np.isfinite(e[m]).any() else np.nan for e in f["evi2"]])
        ok = np.isfinite(med) & np.array([d <= night for d in f["dates"]])
        if ok.sum() < 2:
            return False
        sm = base.running_median(med[ok], self.cfg["growth"]["smoothing_window_obs"])
        return bool(np.argmax(sm) == len(sm) - 1)

    # ------------------------------------------------------------------------------ evidence --
    def _evidence(self, field_id, night):
        pc = self.pc
        night = dt.date.fromisoformat(str(night))
        f = self.store.field(field_id)
        dates, mask = f["dates"], f["mask"]
        B = self.store.bands(field_id).astype("float64")              # (T, 10, H, W)
        T, _, H, W = B.shape
        emerg, peak, _ = self.growth(field_id)
        reason = np.where(mask, 0, 1).astype("uint8")
        if not self.in_phase(field_id, night):
            reason[mask] = 2
        stage_txt = ("pre-emergence" if emerg is None or night < emerg else
                     "growth" if peak is None or night <= peak else "post-peak")
        clear = np.isfinite(B).all(1) & mask[None]
        t_days = np.array([(d - night).days for d in dates], dtype=float)
        before, after = t_days <= 0, t_days > 0
        cb, ca = clear & before[:, None, None], clear & after[:, None, None]
        has_b, has_a = cb.any(0), ca.any(0)
        last_b = T - 1 - np.argmax(cb[::-1], 0)
        first_a = np.argmax(ca, 0)
        reason[(reason == 0) & ~has_b] = 3
        reason[(reason == 0) & (-t_days[last_b] > pc["max_gap_before_days"])] = 4
        reason[(reason == 0) & (~has_a | (t_days[first_a] > pc["max_gap_after_days"]))] = 5
        ok_post = ca & (t_days <= pc["persist_days"])[:, None, None]
        ok_post |= ca & (np.arange(T)[:, None, None] == first_a[None])
        rank = np.cumsum(ok_post, 0)
        posts = np.full((pc["n_post"], H, W), -1)
        for j in range(pc["n_post"]):
            sel = ok_post & (rank == j + 1)
            posts[j] = np.where(sel.any(0), np.argmax(sel, 0), -1)

        out = dict(T=np.full((H, W), np.nan), stage=np.zeros((H, W), "int8"), gapbin=np.zeros((H, W), "int8"),
                   res1=np.full((NB, H, W), np.nan), pred1=np.full((H, W), np.nan), obs1=np.full((H, W), np.nan),
                   std1=np.full((H, W), np.nan), npost=np.zeros((H, W), "int8"),
                   loss=np.full((H, W), np.nan), half=np.full((H, W), np.nan), mix=np.full((H, W), np.nan))
        ys, xs = np.nonzero(reason == 0)
        if len(ys):
            # features: pixel EVI2 at pre, field EVI2 slope before (field median, as evi2_gpr), doy
            med = base.field_median_series(f, 0.5)
            pre = last_b[ys, xs]
            rho0 = B[pre, :, ys, xs]                                    # (n, 10)
            slopes = np.array([base.field_slope(dates, med, i) for i in range(T)])
            doys = np.array([d.timetuple().tm_yday for d in dates])
            slope, doy = slopes[pre], doys[pre]
            if pc["pre_pair_max_gap"]:
                # level at the last obs before: average with the previous clear obs carried forward by the expected
                # frost-free change (halves the observation noise shared by all post residuals)
                cb2 = cb & (np.arange(T)[:, None, None] < last_b[None])
                has2 = cb2.any(0)[ys, xs]
                prev = (T - 1 - np.argmax(cb2[::-1], 0))[ys, xs]
                ad = np.array([(d - dates[0]).days for d in dates], float)
                g2 = ad[pre] - ad[prev]
                use = np.nonzero(has2 & (g2 <= pc["pre_pair_max_gap"]))[0]
                if len(use):
                    rp = B[prev[use], :, ys[use], xs[use]]
                    Xp = np.column_stack([evi2(rp.T), slopes[prev[use]], g2[use], doys[prev[use]]])
                    w0 = pc["pre_weight_last"]
                    rho0[use] = w0 * rho0[use] + (1 - w0) * (rp + self.ref.predict(field_id, Xp))
            e0 = evi2(rho0.T)
            stage = np.digitize(e0, pc["stage_bins"])
            out["stage"][ys, xs] = stage
            hl = np.array(pc["half_days"], float)
            M, J, n = self.F.shape[1], pc["n_post"], len(ys)
            Y = np.full((J, M, n), np.nan)          # standardised frost projection per post obs and mix
            FN = np.full((J, M, n), np.nan)         # its norm (signal per unit canopy loss)
            Q = np.full((J, n), np.nan)             # residual energy outside frost + nuisance subspaces
            TJ = np.full((J, n), np.nan)
            RES, GB, EVR = np.full((J, NB, n), np.nan), np.full((J, n), np.nan), np.full((J, n), np.nan)
            npost = np.zeros(n, int)
            for j in range(J):
                pj = posts[j][ys, xs]
                sel = np.nonzero(pj >= 0)[0]
                if not len(sel):
                    continue
                npost[sel] += 1
                gap = (np.array([(dates[a] - dates[b]).days for a, b in zip(pj[sel], pre[sel])])).astype(float)
                X = np.column_stack([e0[sel], slope[sel], gap, doy[sel]])
                dlt = self.ref.predict(field_id, X)
                r = B[pj[sel], :, ys[sel], xs[sel]] - rho0[sel] - dlt         # (n, 10)
                gb = np.digitize(gap, pc["gap_bins"])
                TJ[j, sel] = t_days[pj[sel]]
                if self.sigma is None:
                    RES[j][:, sel], GB[j, sel] = r.T, gb
                    EVR[j, sel] = evi2(B[pj[sel], :, ys[sel], xs[sel]].T) - evi2((rho0[sel] + dlt).T)
                if j == 0:
                    yy, xx = ys[sel], xs[sel]
                    out["res1"][:, yy, xx] = r.T
                    out["gapbin"][yy, xx] = gb
                    p1 = evi2((rho0[sel] + dlt).T)
                    out["pred1"][yy, xx], out["obs1"][yy, xx] = p1, evi2(B[pj[sel], :, yy, xx].T)
                    if self.sigma is not None:
                        out["std1"][yy, xx] = [self.evi2_sd[(int(a), int(b))] for a, b in zip(stage[sel], gb)]
                if self.sigma is None:
                    continue
                for st_gb in set(zip(stage[sel].tolist(), gb.tolist())):
                    q = (stage[sel] == st_gb[0]) & (gb == st_gb[1])
                    ii = sel[q]
                    Wm = np.linalg.inv(np.linalg.cholesky(self.sigma[st_gb]))
                    Hw = Wm @ self.H
                    P = np.eye(NB) - Hw @ np.linalg.pinv(Hw)
                    Fp = (P @ Wm @ self.F)[:, :, None]                          # (10, M, 1)
                    w = P @ (Wm @ (r[q] - self.mu[st_gb]).T)                    # (10, n_q), bias-corrected
                    if pc["project_brightness"]:                                # pixel-wise nuisance: pre spectrum
                        u = P @ (Wm @ rho0[ii].T)
                        u = u / np.linalg.norm(u, axis=0)
                        Fp = Fp - u[:, None, :] * (u[:, None, :] * Fp).sum(0)[None]
                        w = w - u * (u * w).sum(0)[None]
                    fn = np.sqrt((Fp ** 2).sum(0))                             # (M, n_q) or (M, 1)
                    y = (Fp * w[:, None, :]).sum(0) / fn                       # (M, n_q)
                    Y[j, :, ii] = y.T
                    FN[j, :, ii] = np.broadcast_to(fn, y.shape).T
                    Q[j, ii] = (w ** 2).sum(0) - np.max(np.maximum(y, 0) ** 2, 0)
            out["npost"][ys, xs] = npost
            if self.sigma is not None:
                # local noise scale (compound-Gaussian model): residual energy outside the frost + haze subspaces,
                # pooled over the post obs and a scale_px window, relative to its frost-free median; Y is divided
                # by sqrt(max(1, .)) so changes the frost model does not explain lose weight (never gain)
                dof = NB - self.H.shape[1] - 1
                qi, di = np.zeros((H, W)), np.zeros((H, W))
                qi[ys, xs] = np.nansum(Q, 0)
                di[ys, xs] = np.isfinite(Q).sum(0) * dof
                k = pc["scale_px"]
                ratio = (uniform_filter(qi, k, mode="constant")
                         / np.maximum(uniform_filter(di, k, mode="constant"), 1e-9))[ys, xs]
                out["ratio"] = ratio
                if getattr(self, "qmed", None) is not None and pc["scale_adapt"]:
                    qm = np.array([self.qmed[int(s_)] for s_ in stage])
                    kap = np.maximum(1.0, ratio / qm)
                    Y = Y / np.sqrt(kap)[None, None, :]
                    out["kappa"] = np.full((H, W), np.nan)
                    out["kappa"][ys, xs] = kap
            out.update(Y=Y, FN=FN, Q=Q, TJ=TJ, ys=ys, xs=xs, RES=RES, GB=GB, EVR=EVR)
            if self.sigma is not None and getattr(self, "post_cov", None) is not None:
                self._combine(out, stage, Y, FN, TJ, ys, xs)
        valid = reason == 0
        if self.sigma is not None:
            bad = valid & ~np.isfinite(out["T"])
            reason[bad] = 6
            valid = reason == 0
        k = pc["smooth_px"]
        cnt = uniform_filter(valid.astype(float), k, mode="constant")
        T_s = np.where(valid, uniform_filter(np.where(valid, np.nan_to_num(out["T"]), 0.0), k, mode="constant")
                       / np.maximum(cnt, 1e-9), np.nan)
        out.update(valid=valid, reason=reason, T_s=T_s, last_b=last_b, first_a=first_a, clear=clear,
                   t_days=t_days, stage_txt=stage_txt, emerg=emerg, peak=peak, night=night)
        return out

    def _combine(self, out, stage, Y, FN, TJ, ys, xs):
        """GLS combination over post obs: y_j ~ N(A d_j, R), d_j = 0.5^(t_j / h) * |P f|_j;
        T = max_{h, m} d'R^-1 y / sqrt(d'R^-1 d)  (one-sided: A > 0)."""
        pc = self.pc
        hl = np.array(pc["half_days"], float)
        M, n = Y.shape[1], Y.shape[2]
        best = np.full(n, -np.inf); bh = np.full(n, np.nan); bm = np.full(n, np.nan); amp = np.full(n, np.nan)
        npost = np.isfinite(Y[:, 0, :]).sum(0)
        for st in np.unique(stage):
            R = self.post_cov[int(st)]
            for k in range(1, Y.shape[0] + 1):
                sel = np.nonzero((stage == st) & (npost == k))[0]
                if not len(sel):
                    continue
                Ri = np.linalg.inv(R[:k, :k])
                y = Y[:k, :, sel]                                                # (k, M, n)
                for hi, h in enumerate(hl):
                    d = (0.5 ** (TJ[:k, sel] / h))[:, None, :] * FN[:k, :, sel]  # (k, M, n)
                    Rd = np.einsum("ij,jmn->imn", Ri, d)
                    num = (Rd * y).sum(0)
                    den = np.maximum((Rd * d).sum(0), 1e-12)
                    t = num / np.sqrt(den)                                       # (M, n)
                    mb = np.argmax(t, 0)
                    tb = t[mb, np.arange(len(sel))]
                    upd = tb > best[sel]
                    ii = sel[upd]
                    best[ii], bh[ii], bm[ii] = tb[upd], h, np.array(pc["dead_share_grid"])[mb[upd]]
                    amp[ii] = (num / den)[mb, np.arange(len(sel))][upd]
        out["T"][ys, xs] = np.where(np.isfinite(best), best, np.nan)
        out["half"][ys, xs], out["mix"][ys, xs], out["loss"][ys, xs] = bh, bm, amp

    # -------------------------------------------------------------------------------- damage --
    def damage(self, field_id, night):
        pc = self.pc
        r = self._evidence(field_id, night)
        night = r["night"]
        f = self.store.field(field_id)
        dates, mask = f["dates"], f["mask"]
        valid = r["valid"]
        z = self.cal(field_id, -r["T_s"], r["stage"], r["gapbin"], r["npost"])
        r["reason"][valid & ~np.isfinite(z)] = 6
        valid = r["reason"] == 0
        z = np.where(valid, z, np.nan)
        score = np.where(valid, np.maximum(0.0, -z), np.nan).astype("float32")
        z_thr = -self.score_threshold
        nmeta = self.store.night_meta(field_id, night)
        cp = self.cold_pool(field_id)
        p_frost, att = fe_mod.attribute(self.cfg, self.fe, nmeta, cp, mask, z, valid, z_thr)
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
        dmg = valid & (z <= z_thr)
        meta = dict(field_id=field_id, night=night.isoformat(), method=NAME,
                    pre_date=iso(pi), post_date=iso(qi), next_date=iso(nxt),
                    gap_days_before=-days(pi) if pi is not None else None,
                    gap_days_after=days(qi), next_days_after=days(nxt),
                    growth_stage=r["stage_txt"], emergence=emerg and emerg.isoformat(),
                    peak=peak and peak.isoformat(), **nmeta, **att, z_threshold=z_thr,
                    loss_median_damaged=float(np.median(r["loss"][dmg])) if dmg.any() else np.nan,
                    half_days_median_damaged=float(np.median(r["half"][dmg])) if dmg.any() else np.nan)
        f32 = lambda a: np.where(valid, a, np.nan).astype("float32")
        return {"score": score, "valid": valid, "meta": meta, "curve": None,
                "layers": {"z": f32(z), "predicted": f32(r["pred1"]), "pred_std": f32(r["std1"]),
                           "observed": f32(r["obs1"]), "T": f32(r["T"]), "T_smooth": f32(r["T_s"]),
                           "loss": f32(r["loss"]), "kappa": f32(r.get("kappa", np.full(valid.shape, np.nan))), "half_days": f32(r["half"]), "dead_share": f32(r["mix"]),
                           "n_post": r["npost"], "p_frost": p_frost.astype("float32"), "cold_pool": cp,
                           "reason": r["reason"], "pre_idx": pi, "post_idx": qi, "next_idx": nxt}}


def _atomic(df, fp):
    tmp = fp.with_name(f"{fp.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, fp)
