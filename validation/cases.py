"""Evaluation cases (field x night) and the synthetic frost injection, both method-independent."""
from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np

from data import dem


def is_holdout(field_id: str, share: float) -> bool:
    return int(hashlib.md5(str(field_id).encode()).hexdigest(), 16) % 1000 < share * 1000


def _running_median(x, w):
    h = w // 2
    return np.array([np.median(x[max(0, k - h):k + h + 1]) for k in range(len(x))])


def field_cases(store, field_id, vcfg):
    """Null-eligible intervals of one field's harness growth phase.

    Returns dicts with pre/post acquisition dates, candidate nights (no trigger night in the
    field's cell in [pre, post - 1]), the coldest of those nights' RTMA Tmin (tmin_min) and the
    stage (field-median EVI2 at pre)."""
    d = vcfg["domain"]
    f = store.field(field_id)
    mask = f["mask"]
    share = np.array([np.isfinite(e[mask]).mean() for e in f["evi2"]])
    use = np.nonzero(share >= d["min_field_clear_share"])[0]
    if len(use) < 3:
        return []
    med = np.array([np.nanmedian(f["evi2"][i][mask]) for i in use])
    sm = _running_median(med, d["smoothing_window_obs"])
    above = np.nonzero(sm > d["emergence_evi2"])[0]
    if not len(above):
        return []
    k0, kp = above[0], int(np.argmax(sm))
    lo, hi = d["stage_bins"]
    out = []
    for k in range(k0, kp):
        a, b = f["dates"][use[k]], f["dates"][use[k + 1]]
        nights = [a + dt.timedelta(days=i) for i in range((b - a).days)]
        trig = False
        cand, tmins = [], []
        for n in nights:
            try:
                nm = store.night_meta(field_id, n.isoformat())
            except KeyError:
                continue
            trig |= nm["trigger"]
            cand.append(n.isoformat())
            tmins.append(nm["tmin_c"])
        if trig or not cand:
            continue
        stage = "early" if med[k] < lo else "mid" if med[k] < hi else "closed"
        out.append(dict(field_id=field_id, pre_date=a.isoformat(), post_date=b.isoformat(),
                        gap_days=(b - a).days, pre_med_evi2=float(med[k]), stage=stage,
                        tmin_min=float(min(tmins)), nights=cand))
    return out


def injection_weight(cfg, store, field_id, pattern, vcfg):
    """Spatial weight (H, W) in [0, 1] of the injected damage, 0 outside the field mask."""
    mask = store.field(field_id)["mask"]
    w = mask.astype("float32")
    if pattern == "whole":
        return w
    row = store.fields[store.fields.field_id == field_id].iloc[0]
    z = dem.field_elevation(cfg, row)
    zf = z[mask & np.isfinite(z)]
    if zf.size < 10:
        return w                 # no DEM: fall back to the whole field
    q = np.quantile(zf, vcfg["injection"]["low_quantile"])
    return (mask & np.isfinite(z) & (z <= q)).astype("float32")


def recovery(dt_days, rc):
    """Share of the initial canopy loss still visible dt days after the night."""
    t = np.asarray(dt_days, float)
    lin = 1 - (1 - rc["floor"]) * (t - rc["hold_days"]) / (rc["half_days"] - rc["hold_days"])
    return np.where(t <= rc["hold_days"], 1.0, np.maximum(lin, rc["floor"]))


def inject(f, night, level, weight, vcfg):
    """Copy of the field dict with the frost signature added after the night.

    Returns (field dict, abs drop (H, W) and relative loss (H, W) at each pixel's first clear obs after)."""
    ic = vcfg["injection"]
    night = dt.date.fromisoformat(night)
    ev = f["evi2"].astype("float64").copy()
    dts = np.array([(d - night).days for d in f["dates"]])
    soil = ic["soil_evi2"]
    H, W = weight.shape
    drop_abs = np.zeros((H, W), "float32")
    drop_rel = np.zeros((H, W), "float32")
    seen = np.zeros((H, W), bool)
    for t in np.nonzero(dts > 0)[0]:
        loss = level * weight * float(recovery(dts[t], ic["recovery"]))
        e = ev[t]
        ok = np.isfinite(e) & (e > soil)
        new = np.where(ok, soil + (e - soil) * (1 - loss), e)
        first = np.isfinite(e) & ~seen
        drop_abs[first] = (e - new)[first]
        drop_rel[first] = np.where(ok, loss, 0)[first]
        seen |= np.isfinite(e)
        ev[t] = new
    g = dict(f)
    g["evi2"] = ev.astype(f["evi2"].dtype)
    return g, drop_abs, drop_rel


class InjectedStore:
    """The method's store with one field's EVI2 (and bands) replaced by an injected copy; everything
    else delegated."""

    def __init__(self, base, field_id, field_dict, refl=None):
        self._base, self._fid, self._f, self._r = base, field_id, field_dict, refl

    def __getattr__(self, k):
        return getattr(self._base, k)

    def field(self, field_id):
        return self._f if field_id == self._fid else self._base.field(field_id)

    def bands(self, field_id):
        if field_id == self._fid and self._r is not None:
            return self._r
        return self._base.bands(field_id)


# ------------------------------------------------------------ spectral injection (mode: spectral) --
# Sentinel-2 band centres (nm) in data.s2bands.BANDS order
WAVELENGTH = np.array([490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190], float)
I_RED, I_NIR = 2, 6


def evi2_from(refl):
    red, nir = refl[..., I_RED, :, :], refl[..., I_NIR, :, :]
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)


def endmembers(cfg, store, vcfg):
    """Harness-owned 10-band endmember spectra from dev fields (cached in cache/):
      green  closed canopy: clear pixels at or above the 95th EVI2 percentile of Jul dates
      dead   desiccated haulm: pixels whose EVI2 fell from >= 0.6 to <= 0.25 between consecutive clear
             obs after the field's peak (vine kill / senescence), spectrum at the later obs
      soil   bare soil: clear pixels with EVI2 <= 0.12 before Jun 1"""
    import json
    from data.common import cache_dir
    fp = cache_dir(cfg) / f"validation_endmembers_{store.fips}.json"
    if fp.exists():
        return {k: np.array(v) for k, v in json.loads(fp.read_text()).items()}
    sh = vcfg["split"]["holdout_share"]
    green, dead, soil = [], [], []
    rng = np.random.default_rng(0)
    for fid in store.field_ids():
        if is_holdout(fid, sh):
            continue
        f, r = store.field(fid), store.bands(fid)
        ev, m, dates = f["evi2"], f["mask"], f["dates"]
        med = np.array([np.nanmedian(e[m]) if np.isfinite(e[m]).any() else np.nan for e in ev])
        for t, d in enumerate(dates):
            ok = m & np.isfinite(ev[t])
            if d.month == 7:
                green.append(r[t][:, ok & (ev[t] >= 0.8)].T)
            if d < dt.date(d.year, 6, 1):
                soil.append(r[t][:, ok & (ev[t] <= 0.12)].T)
        if not np.isfinite(med).any():
            continue
        pk = int(np.nanargmax(med))
        prev = None
        for t in range(pk, len(dates)):
            if prev is not None:
                ok = m & np.isfinite(ev[t]) & np.isfinite(ev[prev]) & (ev[prev] >= 0.6) & (ev[t] <= 0.25)
                dead.append(r[t][:, ok].T)
            if np.isfinite(med[t]):
                prev = t

    def spec(parts, q=None):
        a = np.concatenate([p for p in parts if len(p)])
        a = a[rng.choice(len(a), min(len(a), 200000), replace=False)]
        if q is not None:                      # top EVI2 share of the candidate green pixels
            e = evi2_from(a[:, :, None, None])[:, 0, 0]
            a = a[e >= np.quantile(e, q)]
        return np.median(a, 0)
    em = dict(green=spec(green, 0.95), dead=spec(dead), soil=spec(soil))
    fp.write_text(json.dumps({k: v.round(5).tolist() for k, v in em.items()}))
    return em


def recovery_half(dt_days, hold, half):
    """Share of the initial loss still present dt days after the night: 1 up to hold, then halving every half."""
    t = np.asarray(dt_days, float)
    return np.where(t <= hold, 1.0, 0.5 ** ((t - hold) / half))


def inject_spectral(f, refl, night, level, weight, case, em, vcfg):
    """Frost as linear spectral mixing: on every clear obs after the night, a share loss = level * w * D(t)
    of the pixel's green canopy fraction g turns into frost-killed material
    E_k = dead_mix * E_dead + (1 - dead_mix) * E_soil:  refl' = refl + loss * g * (E_k - E_green).
    g = clip((EVI2 - soil_evi2) / (EVI2(E_green) - soil_evi2), 0, 1). EVI2 is recomputed from refl'.
    Returns (field dict, refl', abs EVI2 drop and relative loss at each pixel's first clear obs after)."""
    ic = vcfg["injection"]
    night = dt.date.fromisoformat(night)
    dts = np.array([(d - night).days for d in f["dates"]])
    soil = ic["soil_evi2"]
    eg = em["green"]
    ek = case["dead_mix"] * em["dead"] + (1 - case["dead_mix"]) * em["soil"]
    e_full = float(evi2_from(eg[None, :, None, None])[0, 0, 0])
    r2 = refl.copy()
    ev = f["evi2"].astype("float64").copy()
    H, W = weight.shape
    drop_abs = np.zeros((H, W), "float32")
    drop_rel = np.zeros((H, W), "float32")
    seen = np.zeros((H, W), bool)
    for t in np.nonzero(dts > 0)[0]:
        e = ev[t]
        ok = np.isfinite(e)
        g = np.clip((e - soil) / (e_full - soil), 0, 1)
        loss = level * weight * float(recovery_half(dts[t], ic["hold_days"], case["half_days"]))
        r2[t] = refl[t] + (loss * g)[None] * (ek - eg)[:, None, None]
        new = np.where(ok, evi2_from(r2[t][None])[0], np.nan)
        first = ok & ~seen
        drop_abs[first] = (e - new)[first]
        drop_rel[first] = np.where(g > 0, loss, 0)[first]
        seen |= ok
        ev[t] = new
    out = dict(f)
    out["evi2"] = ev.astype(f["evi2"].dtype)
    return out, r2, drop_abs, drop_rel


def inject_haze(f, refl, night, case, vcfg):
    """Nuisance, not damage: residual haze on the first clear obs after the night only, every pixel:
    refl' = refl + a * (lambda / 490 nm) ** -angstrom (path radiance falls with wavelength)."""
    hc = vcfg["haze"]
    night = dt.date.fromisoformat(night)
    dts = np.array([(d - night).days for d in f["dates"]])
    after = [t for t in np.nonzero(dts > 0)[0] if np.isfinite(f["evi2"][t]).any()]
    r2 = refl.copy()
    ev = f["evi2"].astype("float64").copy()
    if after:
        t = after[0]
        add = case["haze_blue"] * (WAVELENGTH / WAVELENGTH[0]) ** -hc["angstrom"]
        r2[t] = refl[t] + add[:, None, None]
        ev[t] = np.where(np.isfinite(ev[t]), evi2_from(r2[t][None])[0], np.nan)
    out = dict(f)
    out["evi2"] = ev.astype(f["evi2"].dtype)
    return out, r2
