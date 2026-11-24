"""frost_recovery vs Brazilian PSR insurance outcomes (exploration round, no fitted calibration).

    python -m validation.brazil_psr.data <case>          # inputs (once per case)
    python -m validation.brazil_psr.run <case>           # method per window x event / placebo night -> results
    python -m validation.brazil_psr.run <case> --rescore <dir>

"Minimal adapt" as the Swiss benchmark: methods/frost_recovery unchanged except
  - one generic recovery-template set {14 d, none} (no potato P-day stages, no crop stage clock),
  - nights from ERA5-Land (trigger = night Tmin <= 2 C); frost-free training / calibration as in settings
    (every night Tmin >= min_tmin_c and no trigger),
  - no cold-pool attribution (NaN).
Training and the empirical null use frost-free intervals of the same case's windows.

Event nights per window: its cell's trigger nights, consecutive nights merged (coldest kept).  Placebo night per
event: event - 35 d if that night is not a trigger (its 30 d recovery window ends before the event).
Policy summary = max share_damaged over the window's event nights.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from data.common import cache_dir
from methods.frost_recovery import method as fr

from .data import BrazilStore, br_config, fields

RESULTS = Path(__file__).resolve().parents[1] / "results"
STAGES = [{"name": "generic", "until": 1e12, "half_days": [14, None]}]
PLACEBO_DAYS = 35


class BrMethod(fr.Method):
    def cold_pool(self, field_id):
        return np.full(self.store._mask[field_id].shape, np.nan, "float32")


def setup(case):
    cfg = br_config(case)
    s = yaml.safe_load(fr.SETTINGS.read_text())
    s["stages"] = copy.deepcopy(STAGES)
    cfg[fr.NAME] = s
    return cfg


def _runs(g):
    """Coldest night of each run of consecutive nights in g (night, tmin_c)."""
    g = g.sort_values("night")
    d = pd.to_datetime(g.night).dt.date.to_numpy()
    run = np.cumsum(np.r_[1, np.diff(d).astype("timedelta64[D]").astype(int) > 1])
    return [g.iloc[np.where(run == r)[0]].sort_values("tmin_c").night.iloc[0] for r in np.unique(run)]


def event_nights(n, regional_share=0.10):
    """cell_id -> event nights: the cell's own trigger runs plus the regional events (nights triggering in
    >= regional_share of cells, consecutive nights merged, the night with most trigger cells kept), so warm
    windows are also scored on the regional frost nights."""
    cnt = n[n.trigger].groupby("night").size()
    reg = cnt[cnt >= regional_share * n.cell_id.nunique()]
    reg_nights = _runs(pd.DataFrame({"night": reg.index, "tmin_c": -reg.to_numpy()})) if len(reg) else []
    out = {}
    for cid in n.cell_id.unique():
        own = _runs(n[n.trigger & (n.cell_id == cid)]) if (n.trigger & (n.cell_id == cid)).any() else []
        near = {e for e in own for r in reg_nights
                if abs((dt.date.fromisoformat(e) - dt.date.fromisoformat(r)).days) <= 3}
        out[cid] = sorted(set(own) | {r for r in reg_nights
                                      if not any(abs((dt.date.fromisoformat(e) - dt.date.fromisoformat(r)).days) <= 3
                                                 for e in near)})
    return out


_W = {}


def _init(case, cfg):
    store = BrazilStore(cfg, case)
    _W.update(store=store, m=BrMethod(cfg, store), ev=event_nights(store.nights),
              trig=set(zip(store.nights.cell_id[store.nights.trigger], store.nights.night[store.nights.trigger])))


def _one(fid):
    m, store = _W["m"], _W["store"]
    cell = store.field_cell[fid]
    jobs = [(e, "event", e) for e in _W["ev"].get(cell, [])]
    for e in list(_W["ev"].get(cell, [])):
        p = (dt.date.fromisoformat(e) - dt.timedelta(days=PLACEBO_DAYS)).isoformat()
        if (cell, p) not in _W["trig"] and (cell, p) in store._n.index:
            jobs.append((p, "placebo", e))
    rows = []
    for night, kind, ev in jobs:
        try:
            r = m.damage(fid, night)
        except Exception as ex:                          # noqa: BLE001 - record and continue
            rows.append(dict(field_id=fid, night=night, kind=kind, event=ev, error=repr(ex)))
            continue
        v, sc, meta = r["valid"], r["score"], r["meta"]
        rows.append(dict(field_id=fid, night=night, kind=kind, event=ev, n_valid=int(v.sum()),
                         n_mask=int(store._mask[fid].sum()),
                         q90_score=float(np.quantile(sc[v], 0.9)) if v.any() else np.nan,
                         share_damaged=meta["share_damaged"], trajectory=meta["trajectory"],
                         reason=meta["growth_stage"], pre_date=meta["pre_date"], post_date=meta["post_date"],
                         n_post=meta["n_post"], tmin_c=float(store._n.loc[(cell, night)].tmin_c)))
    store.field.cache_clear()
    store.bands.cache_clear()
    m.clear_cache()
    return rows


def run(case, workers=6):
    cfg = setup(case)
    _init(case, cfg)                                 # trains model + calibration once (cached)
    m = _W["m"]
    ids = _W["store"].field_ids()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(case, cfg)) as ex:
        rows = [r for rs in ex.map(_one, ids, chunksize=8) for r in rs]
    df = pd.DataFrame(rows)
    out = RESULTS / f"brazil_psr_{case}_{dt.datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True)
    df.to_parquet(out / "nights.parquet")
    info = dict(case=case, settings=cfg[fr.NAME], model_info=m.mdl["info"], calibration_info=m.cal.info,
                n_windows_with_mask=len(ids))
    (out / "run.json").write_text(json.dumps(info, indent=1, default=str))
    return out


# ---------------------------------------------------------------------------------- scoring --
def _auc(y, x):
    ok = np.isfinite(x)
    y, x = np.asarray(y)[ok], np.asarray(x)[ok]
    return float(roc_auc_score(y, x)) if 0 < y.sum() < len(y) else None


def _rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return (float(spearmanr(a[ok], b[ok])[0]), int(ok.sum())) if ok.sum() >= 10 else (None, int(ok.sum()))


def summarise(case, out):
    out = Path(out)
    cfg = setup(case)
    _, pol = fields(cfg, case)
    n = pd.read_parquet(out / "nights.parquet")
    ok = n[n.get("n_valid", pd.Series(0, index=n.index)).fillna(0) > 0]
    res = {"windows_scored": int(n.field_id.nunique()),
           "reasons_not_valid": n[n.n_valid.fillna(0) == 0].reason.value_counts().to_dict(),
           "event_nights": n[n.kind == "event"].night.value_counts().head(10).to_dict()}
    for kind in ("event", "placebo"):
        k = ok[ok.kind == kind]
        if k.empty:
            continue
        w = (k.sort_values("share_damaged", ascending=False).drop_duplicates("field_id")
             [["field_id", "night", "share_damaged", "q90_score", "trajectory", "tmin_c", "n_valid"]])
        tm = n[n.kind == kind].groupby("field_id").tmin_c.min().rename("tmin_min")
        p = pol.merge(w, on="field_id", how="inner").join(tm, on="field_id")
        p["flag"] = p.share_damaged >= cfg[fr.NAME]["field_flag_share"]
        r = {"policies_valid": len(p)}
        r["by_class"] = (p.groupby("claim_class").agg(n=("flag", "size"), flag_rate=("flag", "mean"),
                         share_mean=("share_damaged", "mean"), share_median=("share_damaged", "median"),
                         loss_ratio_mean=("loss_ratio", "mean"), tmin_mean=("tmin_min", "mean"))
                         .round(3).to_dict("index"))
        frost = p.claim_class.str.startswith("frost")
        r["auc_frostclaim_vs_none_share"] = _auc(frost[p.claim_class.isin(["frost_paid", "frost_unpaid", "none"])],
                                                 p.share_damaged[p.claim_class.isin(["frost_paid", "frost_unpaid", "none"])])
        sel = p.claim_class.isin(["frost_paid", "none"])
        r["auc_frostpaid_vs_none_share"] = _auc((p.claim_class == "frost_paid")[sel], p.share_damaged[sel])
        r["auc_frostpaid_vs_none_q90"] = _auc((p.claim_class == "frost_paid")[sel], p.q90_score[sel])
        r["auc_frostpaid_vs_none_era5_coldness"] = _auc((p.claim_class == "frost_paid")[sel], -p.tmin_min[sel])
        r["rho_share_lossratio_all"] = _rho(p.share_damaged.to_numpy(), p.loss_ratio.fillna(0).to_numpy())
        fr_ = p[frost]
        r["rho_share_lossratio_frostclaims"] = _rho(fr_.share_damaged.to_numpy(), fr_.loss_ratio.fillna(0).to_numpy())
        r["rho_tmin_lossratio_frostclaims"] = _rho(-fr_.tmin_min.to_numpy(), fr_.loss_ratio.fillna(0).to_numpy())
        p["share_bin"] = pd.cut(p.share_damaged, [-0.01, 0.0, 0.1, 0.25, 0.5, 1.0])
        r["share_bins"] = (p.groupby("share_bin", observed=True)
                           .agg(n=("flag", "size"), p_frost_claim=("claim_class", lambda c: c.str.startswith("frost").mean()),
                                p_frost_paid=("claim_class", lambda c: (c == "frost_paid").mean()),
                                loss_ratio_mean=("loss_ratio", "mean"),
                                loss_ratio_mean_frost=("loss_ratio", lambda s: s[p.loc[s.index, "claim_class"].str.startswith("frost")].mean()))
                           .round(3).reset_index().astype({"share_bin": str}).to_dict("records"))
        r["trajectory_by_class"] = pd.crosstab(p.claim_class, p.trajectory).to_dict("index")
        res[kind] = r
        p.drop(columns="share_bin").to_parquet(out / f"policies_{kind}.parquet")
    (out / "scores.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--rescore")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = a.rescore or run(a.case, a.workers)
    print(out)
    print(json.dumps(summarise(a.case, out), indent=1, default=str))
