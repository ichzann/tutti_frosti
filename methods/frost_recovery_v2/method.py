"""frost_recovery_v2: frost_recovery's mechanic over the whole season, no frost night.

Same expected-change model, matched filter, persistence rule and field-level null as frost_recovery, with:
- season = emergence .. day before harvest (season_drops.season_window), instead of emergence .. peak, for the
  training pairs and the frost-free calibration intervals (so natural ageing after the peak is in both);
- no senescence guard at application (senescence_drop = 10: a declining canopy is assessed, not refused);
- every usable date i in the season is a pseudo-night (pre = obs i, posts up to the day before harvest), instead
  of the ERA5 frost nights.
A pixel is damaged if its score >= z_damaged on any pseudo-night.  Depth of a damaged pixel = 1 - median EVI2 of
the post obs / EVI2 of the pre obs (clipped 0..1), max over its pseudo-nights.  Pseudo-night (pre, first post) is
"cold" if an ERA5 trigger night D has pre - 5 d <= D < first post (season_drops.cold_lag_days).  Field outputs as season_drops.
Must run in its own process: it swaps frost_alt.in_growth_phase (used by training and calibration).
"""
from __future__ import annotations

import copy
import datetime as dt

import numpy as np

from methods.frost_alt import method as fa
from methods.frost_recovery import method as fr
from methods.season_drops import method as sd

NAME = "frost_recovery_v2"
_SD = sd.settings()


def _whole_season(cfg, v, d):
    if "_season" not in v:
        v["_season"] = sd.season_window(cfg, v, _SD)
    emerg, peak, harv = v["_season"]
    return emerg is not None and peak is not None and emerg <= d and (harv is None or d < harv)


class Method(fr.Method):
    name = NAME

    def __init__(self, cfg, store):
        cfg = copy.deepcopy(cfg)
        if fr.NAME not in cfg:
            cfg[fr.NAME] = fr.yaml.safe_load(fr.SETTINGS.read_text())
        cfg[fr.NAME]["senescence_drop"] = 10.0
        cfg[fr.NAME]["season_scope"] = "emergence_to_harvest"
        fa.in_growth_phase = _whole_season
        super().__init__(cfg, store)

    def season(self, fid):
        s, store = self.s, self.store
        v = fa.field_view(store, fid, s["usable_share"])
        dates, E, clear, mask = v["dates"], v["E"], v["clear"], v["mask"]
        emerg, peak, harv = sd.season_window(self.cfg, v, _SD)
        meta = dict(field_id=fid, method=NAME, emergence=emerg, peak=peak, harvest=harv,
                    series=[(dates[i].isoformat(), float(v["med"][i])) for i in v["use"]])
        if emerg is None or peak is None:
            return dict(meta, reason="no season found")
        until = harv - dt.timedelta(days=1) if harv is not None else None
        cold = sd.cold_nights(store, fid)
        H, W = mask.shape
        assessed = np.zeros((H, W), bool)
        depth_all, depth_cold = np.zeros((H, W)), np.zeros((H, W))
        hit_all, hit_cold = np.zeros((H, W), bool), np.zeros((H, W), bool)
        events, n_nights = [], 0
        for i in v["use"]:
            night = dates[i]
            if night < emerg or (harv is not None and night >= harv):
                continue
            r = self._evidence(fid, night, v=v, until=until)
            val = r["valid"]
            if not val.any():
                continue
            n_nights += 1
            sc = self.cal.score(fid, r["stage"], r["S"])
            val = val & np.isfinite(sc)
            assessed |= val
            dmg = val & (sc >= self.score_threshold)
            if not dmg.any():
                continue
            pre, posts = r["geo"]["pre"], r["geo"]["posts"]
            P = np.where(clear[posts], E[posts], np.nan)
            with np.errstate(all="ignore"):
                d = np.clip(1 - np.nanmedian(P, 0) / E[pre], 0, 1)
            d = np.where(dmg & np.isfinite(d), d, 0.0)
            is_cold = sd.is_cold_interval(dates[pre], dates[posts[0]], cold, _SD)
            depth_all = np.maximum(depth_all, d)
            hit_all |= dmg
            if is_cold:
                depth_cold = np.maximum(depth_cold, d)
                hit_cold |= dmg
            events.append(dict(a=dates[pre].isoformat(), b=dates[posts[0]].isoformat(), cold=is_cold,
                               n=int(dmg.sum()), dsum=float(d[dmg].sum())))
        store.field.cache_clear()
        store.bands.cache_clear()
        self.clear_cache()
        nv = int(assessed.sum())
        if nv < s["min_valid_px"]:
            return dict(meta, reason="too few assessed pixels / pseudo-nights", n_valid=nv)
        out = dict(meta, reason="", n_valid=nv, n_mask=int(mask.sum()), n_dates=n_nights)
        for tag, dep, hit in (("all", depth_all, hit_all), ("cold", depth_cold, hit_cold)):
            dm = assessed & hit
            out[f"share_{tag}"] = float(dm.sum() / nv)
            out[f"depth_{tag}"] = float(dep[dm].mean()) if dm.any() else 0.0
            out[f"sd_{tag}"] = float(dep[assessed].mean())
        out["events"] = [dict(a=e["a"], b=e["b"], cold=e["cold"], share=e["n"] / nv, depth=e["dsum"] / e["n"])
                         for e in events]
        return out
