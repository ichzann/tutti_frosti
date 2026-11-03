"""frost_stat: frost damage as a two-level (field -> pixel) hypothesis test with an empirically calibrated null.

Evidence (Sentinel-2 only; weather and terrain only for attribution, as frost_evidence / frost_signature).
The per-pixel quantities are those of frost_signature (copied `_evidence`, same statistic T): standardised
residuals u of EVI2 / NDMI / NDRE / PSRI at up to 3 clear obs within 30 d after the night (expected value =
field-hyperparameter GP level + GrowthReference change), the frost fingerprint a (linear mixing of the green
fraction into killed material, method-owned endmembers) and the haze fingerprint h, both in u units;
T = recovery-shaped matched filter with haze projected out; pixel loss estimates per recovery half-time.
Change: the growth phase of a night is decided from the acquisitions up to the night only (`phase`).

Statistical layer:
  0. Null domain: consecutive usable acquisitions (a_k, a_k+1) of non-lockbox fields with a_k in the growth
     phase, no trigger night and every night of [a_k, a_k+1) with RTMA Tmin >= reference.min_tmin_c in the
     field's cell; night = reproducible random night of the interval (2024: 1091 intervals, 838 with >= 50 valid
     px and >= 1 post obs).  Per case: field summaries (below) + a pixel sample.  Cached.
  1. Field-level test.  (i) GLS statistic: the stacked field-median residual vector (posts x indices) is
     modelled under the null as N(X B_stage, S_stage): mean linear in covariates X (1, days from the last obs
     before, days to the first obs after, day of year, log valid pixels; per stage, OLS) and Ledoit-Wolf
     covariance of the null residuals (reference-growth error, level error and acquisition effects shared by
     a field's pixels); frost
     alternative mu + L * vec(D_h(t_j) a_j), recovery D_h = 0.5 ** (t / h), h in half_days, haze direction of
     every post as nuisance; statistic = max_h standardised GLS estimate of L.  (ii) partial-damage statistic:
     10 % quantile of the pixel statistic, robust-standardised per stage and pixel-count class.
     Conformal p-values against the out-of-sample statistics of null cases of other field folds; min-p of
     (i), (ii) calibrated by its own null distribution.  p_F <= alpha_field rejects the field-case.
  2. Pixel stage (only inside a rejected field-case): hierarchical pooling.  Pixel loss estimate
     l_i = L + delta_i + c + e_i (c common to the field); deviations d_i = l_i - median(l) remove c;
     delta_i ~ N(0, tau^2) with tau^2 = within-field variance - null within-field variance (method of moments),
     shrinkage w = tau^2 / (tau^2 + var_e); posterior L_i ~ N(L_hat_F + w d_i, sd_F^2 + w var_e) with L_hat_F,
     sd_F from the field GLS (sd scaled by the null MAD of the GLS statistic if > 1).  p_post = P(L_i <= 0).
  3. Pixel damaged iff p_F <= alpha_field and p_post <= alpha_pixel (selective / hierarchical testing, cf.
     Benjamini & Bogomolov 2014; conformal p-values, Bates et al. 2023).  Under a frost-free field-night the
     probability that any pixel is reported is <= alpha_field (up to exchangeability of null cases and the
     domain match).  Score = -Phi^-1(max(p_F * alpha_pixel / alpha_field, p_post)),
     threshold -Phi^-1(alpha_pixel).

Measured (2024 null domain, numbers only): between-date share of the variance of the field-median statistic
0.14 (contemporaneous-neighbour correction not used: on a widespread frost night the neighbours share the
damage); reference growth model field-median CV RMSE: kNN 0.0796, gradient boosting 0.0774, kNN + same
acquisition-pair mean 0.0788.
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
from scipy.stats import norm

from data.common import cache_dir
from data.store import FieldStore
from methods.evi2_gpr import method as base
from methods.frost_evidence import method as fe_mod
from methods.frost_signature import method as fs

NAME = "frost_stat"
REASONS = {**base.REASONS, 7: "too few valid pixels for the field-level test"}
EVIDENCE_VERSION = 4          # 2: causal growth phase, field summaries per post obs; 3: + pixel loss per h
SETTINGS = yaml.safe_load(Path(__file__).with_name("settings.yaml").read_text())


def lockbox(field_id, share):
    """Lockbox rule of the project (md5(field_id) % 1000 < share * 1000); lockbox fields are never used."""
    return int(hashlib.md5(str(field_id).encode()).hexdigest(), 16) % 1000 < share * 1000


def _key(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _write(fp, write):
    tmp = fp.with_name(f"{fp.name}.{os.getpid()}.tmp")
    write(tmp)
    os.replace(tmp, fp)


def spectra(cfg, store, key):
    """Copy of frost_signature.spectra (method-owned endmembers from non-lockbox fields, haze direction from
    QA haze-dropped field-dates), cached under this method's name."""
    fp = cache_dir(cfg) / f"{NAME}_spectra_{store.fips}_{key}.json"
    if fp.exists():
        return {k: np.array(v) for k, v in json.loads(fp.read_text()).items()}
    if not isinstance(store, FieldStore):
        raise RuntimeError("frost_stat spectra must first be built from a FieldStore")
    rc = cfg["reference"]
    ids = [i for i in store.field_ids() if not lockbox(i, rc["lockbox_share"])]
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
    qa = pd.read_parquet(cache_dir(cfg) / f"s2_qa_{store.fips}.parquet")
    qa = qa[qa.field_id.isin(ids)]
    diffs = []
    for fid, g in qa.groupby("field_id"):
        hz = g[g.drop_reason == "haze"]
        if not len(hz):
            continue
        dates, r = fs._raw_bands(cfg, fid)
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
    tmp = fp.with_name(f"{fp.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({k: np.asarray(v).round(6).tolist() for k, v in out.items()}))
    os.replace(tmp, fp)
    return {k: np.array(v) for k, v in json.loads(fp.read_text()).items()}


def null_intervals(m, store):
    """(field_id, night) of the null domain (module docstring, step 0)."""
    rc = m.cfg["reference"]
    n = store.nights
    cold = n.trigger.copy()
    if rc.get("min_tmin_c") is not None:
        cold |= n.tmin_c < rc["min_tmin_c"]
    coldset = set(zip(n.cell_id[cold], n.night[cold]))
    for fid in sorted(i for i in store.field_ids() if not lockbox(i, rc["lockbox_share"])):
        f = store.field(fid)
        emerg, peak, _ = m.growth(fid)
        if emerg is None or peak is None:
            continue
        med = base.field_median_series(f, rc["min_clear_share"])
        cell = store.field_cell.get(fid)
        use = np.nonzero(np.isfinite(med))[0]
        for i0, i1 in zip(use[:-1], use[1:]):
            d0, d1 = f["dates"][i0], f["dates"][i1]
            if not (emerg <= d0 < peak):
                continue
            gap = (d1 - d0).days
            if any((cell, (d0 + dt.timedelta(days=k)).isoformat()) in coldset for k in range(gap)):
                continue
            h = int(hashlib.md5(f"{fid}:{d0}".encode()).hexdigest(), 16)
            yield fid, d0 + dt.timedelta(days=h % gap)
        m.clear_cache()


def field_summary(r, n_post):
    """Field-case summary of the evidence: per post obs j (in time order) the field medians of the
    standardised residuals u (K indices), of the frost fingerprint a and of the haze fingerprint h (both in
    u units), the days after the night; and the 10 % quantile of the pixel statistic z_smooth."""
    v = r["valid"]
    out = dict(n_valid=int(v.sum()), q10=np.nan, P=0, doy=r["night"].timetuple().tm_yday)
    if not v.any():
        return out
    out["q10"] = float(np.quantile(r["z_smooth"][v], 0.10))
    out["gap_b"] = float(np.median(-r["t_days"][r["last_b"][v]]))
    lv = []
    for l in r["L_smooth"]:
        x = l[v][np.isfinite(l[v])]
        lo, hi = np.quantile(x, [0.01, 0.99]) if x.size else (np.nan, np.nan)
        lv.append(float(np.var(np.clip(x, lo, hi))) if x.size > 1 else np.nan)
    out["lvar"] = np.array(lv)
    U, A, Hh, t = [], [], [], []
    for j in range(n_post):
        uj = r["Upost"][j][:, v]
        ok = np.isfinite(uj).all(0) & np.isfinite(r["Apost"][j][:, v]).all(0)
        if ok.sum() < 0.5 * v.sum():
            break
        U.append(np.median(uj[:, ok], 1))
        A.append(np.median(r["Apost"][j][:, v][:, ok], 1))
        Hh.append(np.median(r["Hpost"][j][:, v][:, ok], 1))
        t.append(float(np.median(r["tpost"][j][v][ok])))
    if U:
        out.update(P=len(U), U=np.array(U), A=np.array(A), H=np.array(Hh), t=np.array(t))
    return out


def covariates(c):
    """Covariates of the null mean of the field-median residuals: days from the last obs before the night,
    days to the first obs after it, day of year of the night, log number of valid pixels."""
    return np.array([1.0, c["gap_b"], c["t"][0], c["doy"], np.log(c["n_valid"])])


def gls_stat(c, mu, S, half_days):
    """Field-level likelihood-ratio statistic for a frost-shaped change.  Null: stacked field-median residual
    vector u (P posts x K indices) ~ N(mu, S) (between-case mean and covariance of frost-free cases).
    Alternative: u = mu + L * vec(D_h(t_j) a_j), L > 0, recovery D_h(t) = 0.5 ** (t / h); the haze direction of
    every post is projected out (GLS with nuisance columns).  Returns max over h of the standardised
    estimate of L (N(0, 1) under a Gaussian null; calibrated empirically)."""
    P, K = c["P"], c["U"].shape[1]
    n = P * K
    S, mu = S[:n, :n], mu[:n]
    Si = np.linalg.inv(S)
    du = c["U"].ravel() - mu
    Hm = np.zeros((n, P))
    for j in range(P):
        Hm[j * K:(j + 1) * K, j] = c["H"][j]
    G = Hm.T @ Si @ Hm
    best = (-np.inf, 0, np.nan, np.nan)
    for i, h in enumerate(half_days):
        a = (c["A"] * (0.5 ** (c["t"] / h))[:, None]).ravel()
        f = Si @ a - Si @ Hm @ np.linalg.solve(G, Hm.T @ Si @ a)     # GLS filter orthogonal to haze
        fa = max(float(f @ a), 1e-12)                                 # = f' S f
        z = float(f @ du / np.sqrt(fa))
        if z > best[0]:
            best = (z, i, float(f @ du / fa), 1.0 / np.sqrt(fa))     # (z, h index, loss estimate, its sd)
    return best


class FieldNull:
    """Field-level null model and conformal p-values (module docstring, step 1).

    Fitted leave-fold-out on the null cases: per stage, mean / Ledoit-Wolf covariance of the stacked
    field-median residuals (GLS statistic) and median / MAD of the 10 % quantile (by pixel-count class).
    Every null case's statistics are computed with the parameters of the other folds (out of sample); a target
    field is compared with the null cases of the other folds. min-p of the two statistics, calibrated by its
    own null distribution (Westfall-Young style, less conservative than Bonferroni for correlated tests)."""

    def __init__(self, cases, folds, tc, half_days, K, n_post):
        from sklearn.covariance import LedoitWolf
        self.folds, self.tc, self.half = folds, tc, half_days
        c = [x for x in cases if x["n_valid"] >= tc["min_valid_px"] and x["P"] >= 1]
        for x in c:
            x["fold"] = base._fold(x["field_id"], folds)
            x["ncls"] = int(np.digitize(x["n_valid"], tc["n_valid_bins"]))
        self.par = {}
        n = n_post * K
        for q in range(folds):
            tr = [x for x in c if x["fold"] != q]
            for st in sorted({x["stage"] for x in tr}):
                g = [x for x in tr if x["stage"] == st]
                gf = [x for x in g if x["P"] == n_post]
                if len(gf) < tc["min_cases_stratum"]:
                    continue
                full = np.array([x["U"].ravel() for x in gf])
                Xc = np.array([covariates(x) for x in gf])
                mu = np.linalg.lstsq(Xc, full, rcond=None)[0]          # null mean = covariates @ mu
                S = LedoitWolf().fit(full - Xc @ mu).covariance_
                q10 = np.array([x["q10"] for x in g])
                loc = {None: (np.median(q10), 1.4826 * np.median(np.abs(q10 - np.median(q10))))}
                for nc in sorted({x["ncls"] for x in g}):
                    y = np.array([x["q10"] for x in g if x["ncls"] == nc])
                    if len(y) >= tc["min_cases_stratum"]:
                        loc[nc] = (np.median(y), 1.4826 * np.median(np.abs(y - np.median(y))))
                self.par[(q, st)] = (mu, S, loc)
        # out-of-sample statistics of every null case
        self.ref = []
        for x in c:
            s = self.stats(x, x["fold"], x["stage"])
            if s is not None:
                self.ref.append((x["fold"], x["stage"], s[0], s[1]))
        self.ref_fold = np.array([r[0] for r in self.ref])
        self.ref_s = np.array([[r[2], r[3]] for r in self.ref])
        # pooling parameters (pixel stage): null scale of the GLS statistic, within-field null variance of the
        # smoothed pixel loss estimate per recovery h (median over null cases of the other folds)
        self.pool = {}
        for (q, st) in self.par:
            sel = (self.ref_fold != q) & (np.array([r[1] for r in self.ref]) == st)
            g = self.ref_s[sel, 0]
            zsc = max(1.0, 1.4826 * np.median(np.abs(g - np.median(g))))
            lv = np.array([x["lvar"] for x in c if x["fold"] != q and x["stage"] == st])
            self.pool[(q, st)] = (zsc, np.nanmedian(lv, 0))
        self._pmin = {}
        for q in range(folds):
            R = self.ref_s[self.ref_fold != q]
            pm = np.min(np.column_stack([self._p(R[:, k], R[:, k], loo=True) for k in range(2)]), 1)
            self._pmin[q] = (R, np.sort(pm))

    def stats(self, x, q, st):
        """(GLS statistic, -standardised q10): both large = damage evidence."""
        if (q, st) not in self.par or x.get("P", 0) < 1:
            return None
        mu, S, loc = self.par[(q, st)]
        g = gls_stat(x, covariates(x) @ mu, S, self.half)[0]
        l0, s0 = loc.get(x["ncls"], loc[None])
        return g, -(x["q10"] - l0) / s0

    @staticmethod
    def _p(ref, x, loo=False):
        """Conformal p-value of x (upper tail) among ref; loo: x is an element of ref (leave itself out)."""
        a = np.sort(ref)
        ge = len(a) - np.searchsorted(a, x, side="left")
        if loo:
            return ge / len(a)                     # (1 + #{others >= x}) / (n - 1 + 1)
        return (1 + ge) / (len(a) + 1)

    def pvalue(self, field_id, x, stage):
        q = base._fold(field_id, self.folds)
        x = dict(x, ncls=int(np.digitize(x["n_valid"], self.tc["n_valid_bins"])))
        s = self.stats(x, q, stage)
        if s is None:
            return np.nan, {}
        R, pm = self._pmin[q]
        p = [float(self._p(R[:, k], s[k])) for k in range(2)]
        pmin = min(p)
        p_f = (1 + np.searchsorted(pm, pmin, side="right")) / (len(pm) + 1)
        mu, S, _ = self.par[(q, stage)]
        z, hi, lhat, se = gls_stat(x, covariates(x) @ mu, S, self.half)
        zsc, s2w = self.pool[(q, stage)]
        return float(p_f), dict(gls=s[0], q10=s[1], p_gls=p[0], p_q10=p[1], h_idx=hi, loss_field=lhat,
                                loss_field_sd=se * zsc, var_within=s2w[hi])


class Method(fs.Method):
    name = NAME

    def __init__(self, cfg, store):
        cfg = copy.deepcopy(cfg)
        cfg["gpr"].update(hyper="field", mean="reference")
        cfg["frost_signature"] = copy.deepcopy(SETTINGS["evidence"])
        cfg[NAME] = copy.deepcopy(SETTINGS)
        base.Method.__init__(self, cfg, store)
        self.fe = cfg["frost_signature"]
        self.tc = cfg[NAME]["test"]
        # caches depend only on the evidence settings (+ the null domain for the table)
        self.key = _key(dict({k: cfg[k] for k in ("gpr", "reference", "frost_signature", "night", "study")},
                             version=EVIDENCE_VERSION))
        a_f, a_p = self.tc["alpha_field"], self.tc["alpha_pixel"]
        self.score_threshold = float(-norm.ppf(a_p))
        self.refs = fs.index_references(cfg, store)
        self.sp = spectra(cfg, store, self.key)
        ek = self.fe["dead_share"] * self.sp["dead"] + (1 - self.fe["dead_share"]) * self.sp["soil"]
        self.d_frost = ek - self.sp["green"]
        self.e_full = float(fs.indices(self.sp["green"][:, None])[0, 0])
        self._cp = {}
        self._phase = {}
        self.sigma = None
        self.sigma = self._sigma(store)
        cases, pix = self._table(store)
        self.field_null = FieldNull(cases, cfg["reference"]["folds"], self.tc, self.fe["half_days"],
                                    len(fs.INDEX_NAMES), self.fe["n_post"])
        pix = pix.merge(pd.DataFrame([dict(case=c["case"], field_id=c["field_id"]) for c in cases]), on="case")
        self.px_null = fe_mod.Calibrator(pix, cfg["reference"]["folds"], self.fe["cal_min_n"])

    def phase(self, field_id, night):
        """Growth phase of the night from the acquisitions up to the night only (evi2_gpr.growth_phase on the
        truncated series): emergence = first smoothed field-median EVI2 > emergence_evi2; the night is in the
        growth phase if it is after emergence and the smoothed maximum so far is at the last acquisition
        before the night. Data after the night (where damage would show) never switch the test off."""
        f = self.store.field(field_id)
        ev = f["evi2"]
        nb = int(sum(d <= night for d in f["dates"]))
        key = (field_id, nb)
        if key not in self._phase:
            emerg, peak, (d_ok, _, _) = base.growth_phase(f["dates"][:nb], ev[:nb], f["mask"], self.cfg)
            self._phase[key] = (emerg, peak, emerg is not None and peak is not None and len(d_ok) > 0
                                and peak == d_ok[-1])
        return self._phase[key]

    # copy of frost_signature.Method._evidence (identical statistic T); additionally returns, per post obs,
    # the standardised residuals and the frost / haze fingerprints of every pixel (field-level test)
    def _evidence(self, field_id, night):
        cfg, g, fe = self.cfg, self.cfg["gpr"], self.fe
        night = dt.date.fromisoformat(str(night))
        f = self.store.field(field_id)
        dates, mask = f["dates"], f["mask"]
        X = self._index_stack(field_id)                              # (K, T, H, W)
        B = self.store.bands(field_id)                               # (T, 10, H, W)
        K, T, H, W = X.shape
        emerg, peak, in_phase = self.phase(field_id, night)          # causal: pre-night data only
        reason = np.where(mask, 0, 1).astype("uint8")
        if not in_phase:
            reason[mask] = 2
        stage_txt = ("pre-emergence" if emerg is None or night < emerg else
                     "growth" if in_phase else "post-peak")
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
        NP = fe["n_post"]
        Upost = np.full((NP, K, H, W), np.nan)     # standardised residuals per post obs
        Apost = np.full((NP, K, H, W), np.nan)     # frost fingerprint per unit canopy loss, in u units
        Hpost = np.full((NP, K, H, W), np.nan)     # haze fingerprint per 0.01 blue, in u units
        tpost = np.full((NP, H, W), np.nan)        # days after the night
        Lh = np.full((len(fe["half_days"]), H, W), np.nan)   # pixel canopy-loss estimate per recovery h
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
                i0 = fs.indices(rho0)
                gfr = np.clip((i0[0] - fs.SOIL_EVI2) / (self.e_full - fs.SOIL_EVI2), 0, 1)
                sF = (fs.indices(rho0 + 0.1 * gfr * self.d_frost[:, None]) - i0) / 0.1
                sH = (fs.indices(rho0 + 0.01 * self.sp["haze"][:, None]) - i0) / 0.01
                for j, ia in enumerate(pj):
                    Upost[j][:, gy, gx] = u[:, j]
                    Apost[j][:, gy, gx] = sF / sd[:, j]
                    Hpost[j][:, gy, gx] = sH / sd[:, j]
                    tpost[j][gy, gx] = t_days[ia]
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
                Lh[:, gy, gx] = Ah
        valid = reason == 0
        if self.sigma is not None:
            bad = valid & ~np.isfinite(Tstat)
            reason[bad] = 6
            valid = reason == 0
        k = fe["smooth_px"]
        cnt = uniform_filter(valid.astype(float), k, mode="constant")
        T_s = np.where(valid, uniform_filter(np.where(valid, np.nan_to_num(Tstat), 0.0), k, mode="constant")
                       / np.maximum(cnt, 1e-9), np.nan)
        L_s = np.stack([np.where(valid, uniform_filter(np.where(valid, np.nan_to_num(l), 0.0), k, mode="constant")
                                 / np.maximum(cnt, 1e-9), np.nan) for l in Lh])
        stage = np.digitize(np.nan_to_num(s_evi2, nan=0.0), fe["stage_bins"]).astype("int8")
        return dict(L_smooth=L_s, valid=valid, reason=reason, z_smooth=-T_s, T=Tstat, loss=loss, half=half, u_first=u_first,
                    stage=stage, npost=npost, pred1=pred1, std1=std1, obs1=obs1, last_b=last_b,
                    first_a=first_a, clear=clear, t_days=t_days, stage_txt=stage_txt,
                    emerg=emerg, peak=peak, night=night,
                    Upost=Upost, Apost=Apost, Hpost=Hpost, tpost=tpost)

    # ----------------------------------------------------------------------------- calibration --
    def _sigma(self, store):
        """Residual correlation of the 4 indices per stage (frost_signature pass 1, own cache)."""
        fp = cache_dir(self.cfg) / f"{NAME}_sigma_{store.fips}_{self.key}.json"
        if not fp.exists():
            if not isinstance(store, FieldStore):
                raise RuntimeError("frost_stat calibration must first be built from a FieldStore")
            rng = np.random.default_rng(0)
            rows = []
            for fid, d0 in fe_mod.calibration_intervals(self, store):
                r = self._evidence(fid, d0)
                v = np.nonzero(r["valid"].ravel())[0]
                v = rng.choice(v, min(self.fe["cal_px_per_case"], len(v)), replace=False)
                rows.append(np.column_stack([r["stage"].ravel()[v],
                                             r["u_first"].reshape(len(fs.INDEX_NAMES), -1)[:, v].T]))
            a = np.concatenate(rows)
            sig = {}
            for st in range(len(self.fe["stage_bins"]) + 1):
                uu = np.clip(a[a[:, 0] == st, 1:], -6, 6)
                sig[st] = np.corrcoef(uu.T).tolist()
            _write(fp, lambda p: p.write_text(json.dumps(sig)))
        return {int(k): np.array(v) for k, v in json.loads(fp.read_text()).items()}

    def _table(self, store):
        """Null domain: one row per field-case (field statistics) + a pixel sample (cached)."""
        fc = cache_dir(self.cfg) / f"{NAME}_nullcases_{store.fips}_{self.key}.pkl"
        fp = cache_dir(self.cfg) / f"{NAME}_nullpixels_{store.fips}_{self.key}.parquet"
        if not (fc.exists() and fp.exists()):
            if not isinstance(store, FieldStore):
                raise RuntimeError("frost_stat calibration must first be built from a FieldStore")
            rng = np.random.default_rng(0)
            cases, pix = [], []
            for fid, night in null_intervals(self, store):
                r = self._evidence(fid, night)
                v = r["valid"]
                if not v.any():
                    continue
                z = r["z_smooth"]
                ci = len(cases)
                cases.append(dict(case=ci, field_id=fid, night=night.isoformat(), stage=self._case_stage(r),
                                  **field_summary(r, self.fe["n_post"])))
                k = np.nonzero(v.ravel())[0]
                k = rng.choice(k, min(self.fe["cal_px_per_case"], len(k)), replace=False)
                pix.append(pd.DataFrame(dict(case=np.full(len(k), ci, "int32"), z=z.ravel()[k],
                                             stage=r["stage"].ravel()[k].astype("int8"),
                                             npost=r["npost"].ravel()[k].astype("int8"))))
            _write(fc, lambda p: pd.to_pickle(cases, p))
            _write(fp, pd.concat(pix, ignore_index=True).to_parquet)
        return pd.read_pickle(fc), pd.read_parquet(fp)

    @staticmethod
    def _case_stage(r):
        v = r["valid"]
        return int(np.bincount(r["stage"][v]).argmax()) if v.any() else -1

    # ---------------------------------------------------------------------------------- damage --
    def damage(self, field_id, night):
        fe, tc = self.fe, self.tc
        a_f, a_p = tc["alpha_field"], tc["alpha_pixel"]
        r = self._evidence(field_id, night)
        night = r["night"]
        f = self.store.field(field_id)
        dates, mask = f["dates"], f["mask"]
        valid = r["valid"]
        zs = r["z_smooth"]
        st = self._case_stage(r)
        stats = field_summary(r, fe["n_post"])
        p_f, ps = (np.nan, {})
        if stats["n_valid"] >= tc["min_valid_px"] and stats["P"] >= 1:
            p_f, ps = self.field_null.pvalue(field_id, stats, st)
        if not np.isfinite(p_f):
            r["reason"][valid] = 7
        zp = self.px_null(field_id, zs, r["stage"], r["npost"])
        r["reason"][(r["reason"] == 0) & ~np.isfinite(zp)] = 6
        valid = r["reason"] == 0
        p_px = norm.cdf(zp)                                      # pooled pixel-level null (layer only)
        p_post = np.full(zp.shape, np.nan)
        pool = {}
        if np.isfinite(p_f) and np.isfinite(ps.get("var_within", np.nan)):
            # pixel stage: hierarchical pooling of the pixel loss estimate with the field-level GLS estimate
            l = r["L_smooth"][ps["h_idx"]]
            x = l[valid & np.isfinite(l)]
            lo, hi = np.quantile(x, [0.01, 0.99])
            s2w = ps["var_within"]
            tau2 = max(0.0, float(np.var(np.clip(x, lo, hi))) - s2w)
            w = tau2 / (tau2 + s2w)
            post_m = ps["loss_field"] + w * (l - np.median(x))
            post_sd = np.sqrt(ps["loss_field_sd"] ** 2 + w * s2w)
            p_post = norm.cdf(-post_m / post_sd)
            pool = dict(pool_weight=w, tau2=tau2, var_within=s2w, loss_field=ps["loss_field"],
                        loss_field_sd=ps["loss_field_sd"], half_days_field=float(fe["half_days"][ps["h_idx"]]))
        p_comb = np.maximum(p_f * a_p / a_f, p_post)
        r["reason"][(r["reason"] == 0) & ~np.isfinite(p_comb)] = 7
        valid = r["reason"] == 0
        score = np.where(valid, -norm.ppf(np.clip(p_comb, 1e-12, 1.0 - 1e-12)), np.nan).astype("float32")
        dmg = valid & (score >= self.score_threshold)
        z_eff = np.where(valid, -score, np.nan)                  # damaged <=> z_eff <= -threshold

        nmeta = self.store.night_meta(field_id, night)
        cp = self.cold_pool(field_id)
        p_frost, att = fe_mod.attribute(self.cfg, fe, nmeta, cp, mask, z_eff, valid, -self.score_threshold)
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
                    peak=peak and peak.isoformat(), **nmeta, **att,
                    p_field=float(p_f), p_field_gls=float(ps.get("p_gls", np.nan)),
                    p_field_q10=float(ps.get("p_q10", np.nan)), field_gls=float(ps.get("gls", np.nan)),
                    field_damaged=bool(p_f <= a_f), **pool,
                    loss_median_damaged=float(np.median(r["loss"][dmg])) if dmg.any() else np.nan,
                    half_days_median_damaged=float(np.median(r["half"][dmg])) if dmg.any() else np.nan)
        f32 = lambda a: np.where(valid, a, np.nan).astype("float32")
        return {"score": score, "valid": valid, "meta": meta, "curve": None,
                "layers": {"z": f32(z_eff), "z_pixel": f32(zp), "p_pixel": f32(p_px), "p_post": f32(p_post),
                           "z_smooth": f32(zs), "predicted": f32(r["pred1"]), "pred_std": f32(r["std1"]),
                           "observed": f32(r["obs1"]), "T": f32(r["T"]), "loss": f32(r["loss"]),
                           "half_days": f32(r["half"]), "n_post": r["npost"],
                           **{f"u_{n}": f32(r["u_first"][i]) for i, n in enumerate(fs.INDEX_NAMES)},
                           "p_frost": p_frost.astype("float32"), "cold_pool": cp, "reason": r["reason"],
                           "pre_idx": pi, "post_idx": qi, "next_idx": nxt}}
