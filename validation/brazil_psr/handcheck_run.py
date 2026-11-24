"""All damage methods + simple baselines on the hand-checked wheat outlines (handcheck_store cases).

    python -m validation.brazil_psr.handcheck_run [method ...]     # default: all METHODS, then baselines
    python -m validation.brazil_psr.handcheck_run table            # one row per field x method -> OUT/table.parquet
    python -m validation.brazil_psr.handcheck_run page             # results page with the area-match slider

Per case (year) and method: Method(cfg, store) as in validation.brazil_psr.run (frost_recovery with the generic
recovery templates; cold pool = NaN everywhere, no Brazil DEM).  Nights per field = run.event_nights (own ERA5
trigger runs + regional frost nights) and placebo nights (event - 35 d, not a trigger night).
Per night: share = meta share_damaged (else share of valid pixels with score >= score_threshold),
magnitude = mean score of the damaged pixels, q90 = 90th percentile score of valid pixels.
Field value per kind (event / placebo) = the night with the largest share.
Baselines (no method): ERA5 coldest event night (-tmin), field-median NDMI drop (last clear date before the
night minus the lowest clear median 5-40 d after), season peak field-median EVI2 (negated: low peak = damage).
"""
from __future__ import annotations

import datetime as dt
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import methods
from validation.brazil_psr import handcheck_store as hs
from validation.brazil_psr.data import BrazilStore, br_config
from validation.brazil_psr.run import PLACEBO_DAYS, event_nights, setup

OUT = hs.hc.OUT / ("methods" if hs.SET == "hc" else f"methods_{hs.SET}")
METHODS = ["frost_recovery", "frost_alt", "frost_stat", "frost_phys", "frost_signature", "frost_evidence", "evi2_gpr"]
CASES = [f"wheat{y}{hs.SET}" for y in hs.YEARS]
_W = {}


def _cfg(case, name):
    return setup(case) if name == "frost_recovery" else br_config(case)


def _init(case, name):
    cfg = _cfg(case, name)
    store = BrazilStore(cfg, case)
    mod = methods.load(name)
    m = (mod.Method if name != "frost_recovery" else __import__(
        "validation.brazil_psr.run", fromlist=["BrMethod"]).BrMethod)(cfg, store)
    m.cold_pool = lambda fid: np.full(store._mask[fid].shape, np.nan, "float32")
    n = store.nights
    _W.update(store=store, m=m, name=name, ev=event_nights(n),
              trig=set(zip(n.cell_id[n.trigger], n.night[n.trigger])))


def _nights(fid):
    store, cell = _W["store"], _W["store"].field_cell[fid]
    jobs = [(e, "event") for e in _W["ev"].get(cell, [])]
    for e in _W["ev"].get(cell, []):
        p = (dt.date.fromisoformat(e) - dt.timedelta(days=PLACEBO_DAYS)).isoformat()
        if (cell, p) not in _W["trig"] and (cell, p) in store._n.index:
            jobs.append((p, "placebo"))
    return jobs


def _one(fid):
    m, store = _W["m"], _W["store"]
    rows = []
    for night, kind in _nights(fid):
        base = dict(field_id=fid, method=_W["name"], night=night, kind=kind)
        try:
            r = m.damage(fid, dt.date.fromisoformat(night))
        except Exception as ex:                              # noqa: BLE001 - record and continue
            rows.append(dict(base, error=repr(ex)[:300]))
            continue
        v, sc, meta = r["valid"], r["score"], r.get("meta", {})
        dmg = v & (sc >= m.score_threshold)
        sh = meta.get("share_damaged")
        rows.append(dict(base, n_valid=int(v.sum()), n_mask=int(store._mask[fid].sum()),
                         share=float(sh) if sh is not None and np.isfinite(sh) else
                         (float(dmg.sum() / v.sum()) if v.any() else np.nan),
                         magnitude=float(sc[dmg].mean()) if dmg.any() else (0.0 if v.any() else np.nan),
                         q90=float(np.quantile(sc[v], 0.9)) if v.any() else np.nan,
                         reason=str(meta.get("growth_stage", meta.get("reason", "")))))
    for c in ("cache_clear",):
        getattr(store.field, c)()
        getattr(store.bands, c)()
    if hasattr(m, "clear_cache"):
        m.clear_cache()
    return rows


def run_method(name, workers=6):
    for case in CASES:
        fp = OUT / f"{name}_{case}.parquet"
        if fp.exists():
            continue
        t0 = dt.datetime.now()
        try:                                                 # a tiny year (3-4 fields) must not stop the others
            _init(case, name)                                # trains / calibrates once (cached on disk)
            ids = _W["store"].field_ids()
            with ProcessPoolExecutor(workers, initializer=_init, initargs=(case, name)) as ex:
                rows = [r for rs in ex.map(_one, ids, chunksize=2) for r in rs]
        except Exception as ex:                              # noqa: BLE001
            print("FAILED", name, case, repr(ex)[:200], flush=True)
            continue
        df = pd.DataFrame(rows).assign(case=case)
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_parquet(fp)
        ok = df.get("n_valid", pd.Series(dtype=float)).fillna(0) > 0
        print(name, case, len(ids), "fields", len(df), "nights", int(ok.sum()), "valid",
              f"{(dt.datetime.now() - t0).seconds // 60} min", flush=True)


def baselines():
    rows = []
    for case in CASES:
        cfg = br_config(case)
        store = BrazilStore(cfg, case)
        _W.update(store=store, ev=event_nights(store.nights),
                  trig=set(zip(store.nights.cell_id[store.nights.trigger], store.nights.night[store.nights.trigger])))
        for fid in store.field_ids():
            f, b = store.field(fid), store.bands(fid)
            m = store._mask[fid]
            d = np.array(f["dates"])
            with np.errstate(all="ignore"):
                nd = (b[:, 6] - b[:, 8]) / (b[:, 6] + b[:, 8])
            ok = np.array([np.isfinite(f["evi2"][t][m]).mean() >= 0.5 for t in range(len(d))])
            ev_med = np.array([np.nanmedian(f["evi2"][t][m]) if ok[t] else np.nan for t in range(len(d))])
            nd_med = np.array([np.nanmedian(nd[t][m]) if ok[t] else np.nan for t in range(len(d))])
            cell = store.field_cell[fid]
            for night, kind in _nights(fid):
                n0 = dt.date.fromisoformat(night)
                pre = np.nonzero(ok & (d <= n0))[0]
                post = ok & (d >= n0 + dt.timedelta(days=5)) & (d <= n0 + dt.timedelta(days=40))
                drop = (nd_med[pre[-1]] - np.nanmin(nd_med[post])) if len(pre) and post.any() else np.nan
                tmin = float(store._n.loc[(cell, night)].tmin_c)
                for meth, val in [("baseline_era5_cold", -tmin), ("baseline_ndmi_drop", drop),
                                  ("baseline_low_peak_evi2", -np.nanmax(ev_med) if ok.any() else np.nan)]:
                    rows.append(dict(field_id=fid, method=meth, night=night, kind=kind, case=case,
                                     n_valid=int(m.sum()), n_mask=int(m.sum()), share=np.nan,
                                     magnitude=np.nan, q90=val))
            store.field.cache_clear()
            store.bands.cache_clear()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "baselines.parquet")


def table():
    """field x method: the event (and placebo) night with the largest share (baselines: largest q90)."""
    df = pd.concat([pd.read_parquet(p) for p in OUT.glob("*.parquet")
                    if p.stem != "table" and not p.stem.startswith("season_")], ignore_index=True)
    df = df[df.n_valid.fillna(0) > 0]
    df["key"] = df.share.fillna(df.q90)
    best = (df.sort_values("key", ascending=False).drop_duplicates(["field_id", "method", "kind"])
            .drop(columns="key"))
    k = hs.kept()[["wid", "year", "group", "loss", "claim_class", "NR_AREA_TOTAL", "matched_ha",
                   "area_diff_ha", "area_diff_rel"]].rename(columns={"wid": "field_id"})
    t = best.merge(k, on="field_id")
    t.to_parquet(OUT / "table.parquet")
    print(t.groupby(["method", "kind"]).field_id.nunique().unstack())
    return t


def page():
    """OUT/../results/index.html: handcheck_results.html + one row per field x method x kind (from table())."""
    import json
    t = table()
    cols = dict(field_id="field_id", method="method", kind="kind", night="night", share="share", magnitude="magnitude",
                q90="q90", group="group", loss="loss", year="year", NR_AREA_TOTAL="ins", matched_ha="out",
                area_diff_ha="dha", area_diff_rel="drel")
    n_ev = t[t.kind == "event"].groupby("method").field_id.nunique()
    t = t[t.method.isin(n_ev.index[n_ev >= 10])]             # frost_stat: 1 scored field
    t = t[list(cols)].rename(columns=cols)
    t["loss"] = np.where(t.group == "none", np.nan, t.loss)
    rows = json.loads(t.to_json(orient="records", double_precision=4))
    tpl = (hs.hc.ROOT / "validation" / "brazil_psr" /
           ("handcheck_results_sel.html" if hs.SET == "sel" else "handcheck_results.html")).read_text()
    sj = OUT / "season.json"                                 # season_run.page(): whole-season detectors
    tpl = tpl.replace("__SEASON__", sj.read_text() if sj.exists() else "null")
    d = hs.hc.OUT / ("results" if hs.SET == "hc" else f"results_{hs.SET}")
    d.mkdir(exist_ok=True)
    (d / "index.html").write_text(tpl.replace("__DATA__", json.dumps(dict(rows=rows, default_method="frost_recovery"))))
    print(len(rows), "rows ->", d / "index.html")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["table"]:
        table()
    elif a == ["page"]:
        page()
    else:
        for nm in a or METHODS:
            if nm == "baselines":
                baselines()
            else:
                try:
                    run_method(nm)
                except Exception as ex:                      # noqa: BLE001 - one method must not stop the rest
                    print("FAILED", nm, repr(ex)[:300], flush=True)
        if not a:
            baselines()
            table()
