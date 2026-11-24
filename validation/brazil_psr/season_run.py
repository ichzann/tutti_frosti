"""Whole-season detectors (season_drops, frost_recovery_v2) on the automatic wheat selection (BR_SET=sel).

    BR_SET=sel python -m validation.brazil_psr.season_run rate     # ageing rate from no-claim development fields
    BR_SET=sel python -m validation.brazil_psr.season_run drops    # season_drops, all years
    BR_SET=sel python -m validation.brazil_psr.season_run v2       # frost_recovery_v2, per year
    BR_SET=sel python -m validation.brazil_psr.season_run page     # -> season.json, results page

Hold-out = frost_alt.is_lockbox(field_id, 0.3) (the methods' own lockbox: also left out of frost_recovery's
training and calibration); development = the rest.  The ageing rate is set from development no-claim fields only.
Output: OUT/season_<method>.parquet, one row per field (events and EVI2 series as JSON).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from methods.frost_alt import method as fa
from validation.brazil_psr import handcheck_run as hr
from validation.brazil_psr import handcheck_store as hs
from validation.brazil_psr.data import BrazilStore, br_config
from validation.brazil_psr.run import setup

OUT = hr.OUT
CASES = hr.CASES
HOLD_SHARE = 0.3
_W = {}


def is_hold(fid):
    return fa.is_lockbox(fid, HOLD_SHARE)


def _jsonable(r):
    r = dict(r)
    for k in ("emergence", "peak", "harvest"):
        r[k] = r[k].isoformat() if r.get(k) is not None else None
    r["series"] = json.dumps(r.get("series", []))
    r["events"] = json.dumps(r.get("events", []))
    return r


def ageing_rate():
    from methods.season_drops import method as sd
    s = sd.settings()
    k = hs.kept().set_index("wid")
    rates = []
    for case in CASES:
        cfg = br_config(case)
        store = BrazilStore(cfg, case)
        ids = [f for f in store.field_ids() if not is_hold(f) and f in k.index and k.loc[f, "group"] == "none"]
        rates.append(sd.ageing_rates(cfg, store, ids, s))
    r = np.concatenate(rates)
    q = float(np.quantile(r, s["ageing_quantile"]))
    info = dict(ageing_rate=q, n_pairs=len(r), quantile=s["ageing_quantile"],
                median=float(np.median(r)), q75=float(np.quantile(r, 0.75)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "season_ageing_rate.json").write_text(json.dumps(info, indent=1))
    print(info)
    return q


def _init_drops(case, rate):
    from methods.season_drops import method as sd
    cfg = br_config(case)
    store = BrazilStore(cfg, case)
    _W.update(m=sd.Method(cfg, store, rate))


def _init_v2(case):
    from methods.frost_recovery_v2 import method as v2
    cfg = setup(case)
    store = BrazilStore(cfg, case)
    _W.update(m=v2.Method(cfg, store))


def _one(fid):
    try:
        return _jsonable(_W["m"].season(fid))
    except Exception as ex:                                  # noqa: BLE001 - record and continue
        return dict(field_id=fid, reason=f"error: {repr(ex)[:200]}")


def run(name, workers=6):
    fp = OUT / f"season_{name}.parquet"
    rate = json.loads((OUT / "season_ageing_rate.json").read_text())["ageing_rate"] if name == "season_drops" else None
    parts = []
    for case in CASES:
        t0 = dt.datetime.now()
        init, args = (_init_drops, (case, rate)) if name == "season_drops" else (_init_v2, (case,))
        try:
            init(*args)                                      # trains / calibrates once (cached on disk)
            ids = BrazilStore(br_config(case), case).field_ids()
            with ProcessPoolExecutor(workers, initializer=init, initargs=args) as ex:
                rows = list(ex.map(_one, ids, chunksize=2))
        except Exception as ex:                              # noqa: BLE001 - a small year must not stop the rest
            print("FAILED", name, case, repr(ex)[:200], flush=True)
            ids = BrazilStore(br_config(case), case).field_ids()
            rows = [dict(field_id=f, reason=f"run failed for {case}") for f in ids]
        df = pd.DataFrame(rows).assign(case=case, method=name)
        parts.append(df)
        ok = (df.reason == "").sum() if "reason" in df else 0
        print(name, case, len(ids), "fields", ok, "assessed", f"{(dt.datetime.now() - t0).seconds // 60} min", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(fp)


def season_data():
    """Rows for the page: field x detector x events (all / cold), plus coverage and example timelines."""
    k = hs.kept()[["wid", "year", "group", "loss", "NR_AREA_TOTAL", "matched_ha", "area_diff_ha", "area_diff_rel"]]
    k = k.rename(columns={"wid": "field_id"}).drop_duplicates("field_id")
    rows, cover, series = [], [], {}
    for name in ("season_drops", "frost_recovery_v2"):
        fp = OUT / f"season_{name}.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp).merge(k, on="field_id")
        d["hold"] = d.field_id.map(is_hold)
        for hold, g in d.groupby("hold"):
            rc = g.reason.fillna("").replace("", "assessed").str.replace(r"error: .*", "error", regex=True)
            cover.append(dict(method=name, hold=bool(hold), n=len(g), counts=rc.value_counts().to_dict()))
        ok = d[d.reason.fillna("") == ""]
        for r in ok.itertuples():
            for ev in ("all", "cold"):
                rows.append(dict(field_id=r.field_id, method=name, ev=ev, share=getattr(r, f"share_{ev}"),
                                 depth=getattr(r, f"depth_{ev}"), sd=getattr(r, f"sd_{ev}"), hold=bool(r.hold)))
            series.setdefault(r.field_id, dict(series=json.loads(r.series), peak=r.peak, harvest=r.harvest,
                                               emergence=r.emergence))[name] = json.loads(r.events)
    t = pd.read_parquet(OUT / "table.parquet")
    t = t[(t.method == "frost_recovery") & (t.kind == "event")]
    for r in t.itertuples():
        for ev in ("all", "cold"):
            rows.append(dict(field_id=r.field_id, method="frost_recovery (frost nights)", ev=ev, share=r.share,
                             depth=None, sd=None, hold=bool(is_hold(r.field_id))))
    df = pd.DataFrame(rows).merge(k, on="field_id")
    df["loss"] = np.where(df.group == "none", np.nan, df.loss)
    df = df.rename(columns={"NR_AREA_TOTAL": "ins", "matched_ha": "out", "area_diff_ha": "dha", "area_diff_rel": "drel"})
    # cold nights per field (for the timelines)
    for case in CASES:
        try:
            store = BrazilStore(br_config(case), case)
        except Exception:                                    # noqa: BLE001
            continue
        from methods.season_drops.method import cold_nights
        for fid in store.field_ids():
            if fid in series:
                series[fid]["cold"] = [c.isoformat() for c in cold_nights(store, fid)]
    meta = k.set_index("field_id")
    tl = []
    for fid, s in series.items():
        if fid in meta.index and not is_hold(fid):
            m = meta.loc[fid]
            tl.append(dict(field_id=fid, group=m.group, loss=None if m.group == "none" else float(m.loss),
                           year=int(m.year), **s))
    rate = json.loads((OUT / "season_ageing_rate.json").read_text())
    return dict(rows=json.loads(df.to_json(orient="records", double_precision=4)), coverage=cover,
                timelines=tl, ageing=rate)


def page():
    (OUT / "season.json").write_text(json.dumps(season_data()))
    hr.page()


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["rate"]:
        ageing_rate()
    elif a == ["drops"]:
        run("season_drops")
    elif a == ["v2"]:
        run("frost_recovery_v2")
    elif a == ["page"]:
        page()
