"""Benchmark frost_recovery against Swiss Hail claims: Canton Zurich, maize, hail day 2021-07-12.

    python -m validation.swiss_hail.data                 # inputs (once)
    python -m validation.swiss_hail.run                  # method on every field -> 1 km comparison
    python -m validation.swiss_hail.run --rescore <dir>  # metrics only

"Minimal adapt" (user decision 2026-11-03): methods/frost_recovery unchanged except
  - event date = the hail day instead of an RTMA trigger night (SwissStore nights: radar-hail days),
  - one generic recovery-template set {14 d, none} instead of the potato P-day stages,
  - no terrain / frost attribution (cold_pool = NaN; attribution layers are not used here).
Training and the empirical null use radar-hail-free intervals of the same Zurich 2021 maize fields.

Truth: validation.swiss_hail.truth, crop "maize", 1 km cells.  A field belongs to the cell with the nearest
centroid.  Truth counts all insured maize fields in the cell (any size); the method sees fields >= 1 ha only.
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
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from methods.frost_recovery import method as fr

from . import truth
from .data import FIPS, SwissStore, ch_config, fields

EVENT = "2021-07-12"
CROP = "maize"
RESULTS = Path(__file__).resolve().parents[1] / "results"
STAGES = [{"name": "generic", "until": 1e12, "half_days": [14, None]}]


class HailMethod(fr.Method):
    def cold_pool(self, field_id):
        m = self.store._mask[field_id]
        return np.full(m.shape, np.nan, "float32")


def setup():
    cfg = ch_config()
    s = yaml.safe_load(fr.SETTINGS.read_text())
    s["stages"] = copy.deepcopy(STAGES)
    cfg[fr.NAME] = s
    return cfg


_W = {}


def _init(cfg):
    store = SwissStore(cfg)
    _W.update(store=store, m=HailMethod(cfg, store))


def _one(fid):
    m, store = _W["m"], _W["store"]
    r = m.damage(fid, EVENT)
    store.field.cache_clear()
    store.bands.cache_clear()
    m.clear_cache()
    v, sc = r["valid"], r["score"]
    meta = r["meta"]
    q90 = float(np.quantile(sc[v], 0.9)) if v.any() else np.nan
    return dict(field_id=fid, n_valid=int(v.sum()), n_mask=int(store._mask[fid].sum()), q90_score=q90,
                share_damaged=meta["share_damaged"], trajectory=meta["trajectory"], reason=meta["growth_stage"],
                pre_date=meta["pre_date"], post_date=meta["post_date"], n_post=meta["n_post"],
                half_days_mode=meta["half_days_mode"])


def run(workers=6):
    cfg = setup()
    _init(cfg)                                   # trains model + calibration once (cached), then workers load
    m = _W["m"]
    ids = _W["store"].field_ids()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(cfg,)) as ex:
        rows = list(ex.map(_one, ids, chunksize=8))
    df = pd.DataFrame(rows)
    df["flag"] = df.share_damaged >= cfg[fr.NAME]["field_flag_share"]
    out = RESULTS / f"swiss_hail_{dt.datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True)
    df.to_parquet(out / "fields.parquet")
    info = dict(event=EVENT, crop=CROP, fips=FIPS, settings=cfg[fr.NAME], model_info=m.mdl["info"],
                calibration_info=m.cal.info)
    (out / "run.json").write_text(json.dumps(info, indent=1, default=str))
    return out


# ---------------------------------------------------------------------------------- scoring --
def _cells(df):
    """Fields -> nearest maize truth cell; per-cell method summary joined with truth for every event."""
    cfg = ch_config()
    f = fields(cfg)[["field_id", "geometry"]]
    pts = f.set_index("field_id").representative_point().to_crs(4326)
    t = pd.read_parquet(truth.OUT / "truth_1km.parquet") if (truth.OUT / "truth_1km.parquet").exists() else truth.build(1)
    t = t[t.crop == CROP]
    grid = t[t.date == t.date.iloc[0]][["cell", "latitude", "longitude"]]
    nn = cKDTree(grid[["latitude", "longitude"]].to_numpy()).query(
        np.column_stack([pts.loc[df.field_id].y, pts.loc[df.field_id].x]))
    df = df.assign(cell=grid.cell.to_numpy()[nn[1]], dist_deg=nn[0])
    ev = df[df.n_valid > 0]
    c = ev.groupby("cell").agg(n_eval=("field_id", "size"), n_flag=("flag", "sum"),
                               mean_share=("share_damaged", "mean"), max_q90=("q90_score", "max"))
    c["pred_frac"] = c.n_flag / c.n_eval
    return df, c, t


def _scores(c, tt):
    j = c.join(tt.set_index("cell")[["n_fields", "n_damaged", "frac_damaged", "MESHS", "POH"]])
    y = (j.n_damaged > 0).to_numpy()
    res = dict(n_cells=len(j), n_cells_truth_damaged=int(y.sum()))
    if 0 < y.sum() < len(y):
        for k in ("pred_frac", "mean_share", "max_q90", "MESHS", "POH"):
            res[f"auc_{k}"] = float(roc_auc_score(y, j[k].fillna(-1e9)))
    pf = j.n_flag > 0
    tp, fp_, fn = int((pf & y).sum()), int((pf & ~y).sum()), int((~pf & y).sum())
    res.update(cells_flagged=int(pf.sum()), hits=tp, false_alarms=fp_, misses=fn,
               pod=tp / max(tp + fn, 1), far=fp_ / max(tp + fp_, 1), csi=tp / max(tp + fp_ + fn, 1))
    d = j[y]
    if len(d) >= 5:
        res["spearman_predfrac_truthfrac_damaged_cells"] = float(spearmanr(d.pred_frac, d.frac_damaged)[0])
        res["spearman_meanshare_truthfrac_damaged_cells"] = float(spearmanr(d.mean_share, d.frac_damaged)[0])
    res["spearman_predfrac_truthfrac_all_cells"] = float(spearmanr(j.pred_frac, j.frac_damaged.fillna(0))[0])
    return res, j


def rescore(out):
    out = Path(out)
    df = pd.read_parquet(out / "fields.parquet")
    df, c, t = _cells(df)
    sc = {"fields": dict(n=len(df), n_valid=int((df.n_valid > 0).sum()), n_flag=int(df.flag.sum()),
                         reasons=df[df.n_valid == 0].reason.value_counts().to_dict(),
                         trajectories=df.trajectory.value_counts().to_dict())}
    for d in sorted(t.date.unique()):
        r, j = _scores(c, t[t.date == d])
        sc[str(d)] = r
        if str(d) == EVENT:
            j.to_parquet(out / "cells.parquet")
            # amount: truth fraction bins in damaged cells vs method
            jj = j.copy()
            jj["truth_bin"] = pd.cut(jj.frac_damaged.fillna(0), [-0.01, 0, 0.2, 0.4, 0.7, 1.0])
            sc["amount_bins"] = (jj.groupby("truth_bin", observed=True)
                                 .agg(cells=("pred_frac", "size"), pred_frac_mean=("pred_frac", "mean"),
                                      mean_share_mean=("mean_share", "mean")).round(3).reset_index()
                                 .astype({"truth_bin": str}).to_dict("records"))
    # cells also hit on 2021-07-24 (post obs from 07-24 on carry that hail too)
    t24 = t[(t.date == dt.date(2021, 7, 24)) & (t.n_damaged > 0)].cell
    sc["cells_with_0724_damage"] = int(c.index.isin(t24).sum())
    (out / "scores.json").write_text(json.dumps(sc, indent=1, default=str))
    return sc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = a.rescore or run(a.workers)
    print(json.dumps(rescore(out), indent=1, default=str))
