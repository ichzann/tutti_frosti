"""season_drops: whole-season sudden-drop detector (no frost night, no neighbour fields).

Per field, season = emergence (evi2_gpr.growth_phase) .. the day before harvest (settings.yaml: first usable date
>= 40 d after the peak with field-median EVI2 <= 0.20 and no regrowth above 0.30).
Per pixel (EVI2 averaged over 3 x 3 clear pixels), each pair of consecutive clear dates (a, b) inside the season is
a candidate drop if EVI2_a >= 0.25, (EVI2_a - EVI2_b) / EVI2_a >= 0.15, b - a <= 30 d and the decline per day is
faster than normal ageing (ageing_rate: 90th percentile of field-median decline per day between peak and harvest
on the no-claim development fields, passed in by the caller).  It is confirmed if the next clear date c after b
(within 30 d) still shows >= half the drop: EVI2_c <= EVI2_a - 0.5 (EVI2_a - EVI2_b); unconfirmed drops (one-date
dips: haze, shadow, noise) are not counted.  Depth = (EVI2_a - EVI2_b) / EVI2_a.
Events = confirmed drops grouped by their (a, b) dates; an event is "cold" if an ERA5 trigger night D
(store.nights, tmin <= 2 C) falls in a - cold_lag_days <= D < b.
Field outputs (all events / cold events only): share = damaged / assessed pixels, depth = mean depth of the
damaged pixels (max over a pixel's events), share_depth = mean pixel depth over all assessed pixels (0 if none).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import uniform_filter

from methods.evi2_gpr.method import growth_phase
from methods.frost_alt import method as fa

NAME = "season_drops"
SETTINGS = Path(__file__).with_name("settings.yaml")


def settings():
    return yaml.safe_load(SETTINGS.read_text())


def season_window(cfg, v, s):
    """(emergence, peak, harvest) dates; harvest None if not seen (season runs to the last date)."""
    f = v["f"]
    emerg, peak, _ = growth_phase(f["dates"], f["evi2"], f["mask"], cfg)
    if emerg is None or peak is None:
        return emerg, peak, None
    dates, use, med = v["dates"], v["use"], v["med"]
    after = [i for i in use if (dates[i] - peak).days >= s["harvest_after_peak_days"]]
    for j, i in enumerate(after):
        later = [med[k] for k in after[j + 1:]]
        if med[i] <= s["harvest_evi2"] and (not later or max(later) <= s["regrow_evi2"]):
            return emerg, peak, dates[i]
    return emerg, peak, None


def ageing_rates(cfg, store, fids, s):
    """Field-median EVI2 decline per day for consecutive usable dates between peak and harvest."""
    out = []
    for fid in fids:
        v = fa.field_view(store, fid, s["usable_share"])
        emerg, peak, harv = season_window(cfg, v, s)
        if peak is None:
            continue
        d, med = v["dates"], v["med"]
        idx = [i for i in v["use"] if d[i] >= peak and (harv is None or d[i] < harv)]
        for a, b in zip(idx[:-1], idx[1:]):
            out.append((med[a] - med[b]) / max((d[b] - d[a]).days, 1))
        store.field.cache_clear()
        store.bands.cache_clear()
    return np.array(out)


def cold_nights(store, fid):
    n = store.nights
    cell = store.field_cell.get(fid)
    return sorted(dt.date.fromisoformat(x) for x in n.night[(n.cell_id == cell) & n.trigger])


def is_cold_interval(a, b, cold, s):
    return any(a - dt.timedelta(days=s["cold_lag_days"]) <= c < b for c in cold)


class Method:
    name = NAME

    def __init__(self, cfg, store, ageing_rate, s=None):
        self.cfg, self.store, self.s = cfg, store, s or settings()
        self.ageing_rate = float(ageing_rate)

    def season(self, fid):
        s, store = self.s, self.store
        v = fa.field_view(store, fid, s["usable_share"])
        dates, mask, E, clear = v["dates"], v["mask"], v["E"], v["clear"]
        emerg, peak, harv = season_window(self.cfg, v, s)
        meta = dict(field_id=fid, method=NAME, emergence=emerg, peak=peak, harvest=harv,
                    series=[(dates[i].isoformat(), float(v["med"][i])) for i in v["use"]])
        if emerg is None or peak is None:
            return dict(meta, reason="no season found")
        idx = [i for i in v["use"] if dates[i] >= emerg and (harv is None or dates[i] < harv)]
        k = s["smooth_px"]
        H, W = mask.shape
        V = np.full((len(idx), H, W), np.nan)
        for j, i in enumerate(idx):
            c = clear[i]
            num = uniform_filter(np.where(c, E[i], 0.0), k, mode="constant")
            den = uniform_filter(c.astype(float), k, mode="constant")
            with np.errstate(divide="ignore", invalid="ignore"):
                V[j] = np.where(c & (den > 0), num / den, np.nan)
        T = np.array([dates[i] for i in idx])
        ok = np.isfinite(V)
        valid = mask & (ok.sum(0) >= s["min_clear_dates"])
        if valid.sum() < s["min_valid_px"]:
            return dict(meta, reason="too few clear pixels / dates in the season", n_valid=int(valid.sum()))
        cold = cold_nights(store, fid)
        # next clear value / date after each index (reverse pass)
        K = len(idx)
        nx_v = np.full((K, H, W), np.nan)
        nx_t = np.full((K, H, W), -1, int)
        cur_v, cur_t = np.full((H, W), np.nan), np.full((H, W), -1, int)
        for j in range(K - 1, -1, -1):
            nx_v[j], nx_t[j] = cur_v, cur_t
            cur_v = np.where(ok[j], V[j], cur_v)
            cur_t = np.where(ok[j], j, cur_t)
        depth_all = np.zeros((H, W))
        depth_cold = np.zeros((H, W))
        events = {}
        la_v, la_t = np.full((H, W), np.nan), np.full((H, W), -1, int)
        for j in range(K):
            if j:
                has = ok[j] & (la_t >= 0) & valid
                gap = np.array([(T[j] - T[t]).days if t >= 0 else 0 for t in range(K)])
                g = np.where(la_t >= 0, gap[np.maximum(la_t, 0)], 0)
                drop = la_v - V[j]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = drop / la_v
                    rate = drop / np.maximum(g, 1)
                cand = (has & (la_v >= s["min_pre_evi2"]) & (rel >= s["min_rel_drop"]) & (g <= s["max_gap_days"])
                        & (rate > self.ageing_rate))
                if cand.any():
                    nt = nx_t[j]
                    cg = np.where(nt >= 0, np.array([(T[t] - T[j]).days if t >= 0 else 10 ** 6
                                                     for t in range(K)])[np.maximum(nt, 0)], 10 ** 6)
                    conf = cand & (nt >= 0) & (cg <= s["confirm_days"]) & \
                        (nx_v[j] <= la_v - s["confirm_keep"] * drop)
                    for a in np.unique(la_t[conf]):
                        m = conf & (la_t == a)
                        d = np.clip(rel, 0, 1)
                        is_cold = is_cold_interval(T[a], T[j], cold, s)
                        depth_all = np.where(m, np.maximum(depth_all, d), depth_all)
                        if is_cold:
                            depth_cold = np.where(m, np.maximum(depth_cold, d), depth_cold)
                        key = (T[a].isoformat(), T[j].isoformat())
                        e = events.setdefault(key, dict(a=key[0], b=key[1], cold=is_cold, n=0, dsum=0.0))
                        e["n"] += int(m.sum())
                        e["dsum"] += float(d[m].sum())
            la_v = np.where(ok[j], V[j], la_v)
            la_t = np.where(ok[j], j, la_t)
        nv = int(valid.sum())
        out = dict(meta, reason="", n_valid=nv, n_mask=int(mask.sum()), n_dates=K)
        for tag, dep in (("all", depth_all), ("cold", depth_cold)):
            dm = valid & (dep > 0)
            out[f"share_{tag}"] = float(dm.sum() / nv)
            out[f"depth_{tag}"] = float(dep[dm].mean()) if dm.any() else 0.0
            out[f"sd_{tag}"] = float(dep[valid].mean())
        out["events"] = sorted(([dict(a=e["a"], b=e["b"], cold=e["cold"], share=e["n"] / nv, depth=e["dsum"] / e["n"])
                                 for e in events.values()]), key=lambda e: e["a"])
        store.field.cache_clear()
        store.bands.cache_clear()
        return out
