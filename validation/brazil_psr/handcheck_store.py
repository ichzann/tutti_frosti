"""Brazil store for the hand-checked wheat outlines (field_handcheck + user verdicts), one case per year.

    python -m validation.brazil_psr.handcheck_store            # all years: fields -> S2 10 bands -> masks -> QA -> nights
    BR_SET=sel python -m validation.brazil_psr.handcheck_store # same for the automatic shape selection (field_full)

Case wheat<year>hc: fields = the policies the user kept (verdict "keep" in the hand-check artifact), window =
outline bounding box + MARGIN_M on the field_handcheck 10 m grid, mask = the outline pixels (no crop-mask rule).
S2, QA and ERA5-Land nights as validation.brazil_psr.data (same functions, cases registered in data.CASES).
BR_SET=sel: fields = field_full selection.parquet (automatic shape rules, no verdicts), cases wheat<year>sel.
BR_SET=full: the same for the ~400-field selection of the method competition, cases wheat<year>full (own caches,
so the 189-field sel caches stay as they were).
BR_SET=b4: batch 4 (second competition lockbox, selection_b4.parquet), cases wheat<year>b4.  QA clear-sky line
fitted on the full set's rows (frozen preprocessing), not on batch 4 itself.
"""
from __future__ import annotations

import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from data.common import cache_dir
from data.s2bands import BANDS
from validation.brazil_psr import data as br
from validation.brazil_psr import field_handcheck as hc

MARGIN_M = 300
SET = os.environ.get("BR_SET", "hc")                    # hc = hand-kept outlines, sel / full = automatic shape selection
AUTO = SET in ("sel", "full", "b4")
SEL = "selection_b4.parquet" if SET == "b4" else "selection.parquet"
if AUTO:
    YEARS = sorted(int(y) for y in pd.read_parquet(hc.OUT / SEL).year.unique())
else:
    YEARS = sorted(set(hc.YEARS) | set(hc.B2_YEARS))
for _y in YEARS:
    br.CASES[f"wheat{_y}{SET}"] = dict(crop="Trigo", year=str(_y), season=(f"{_y}-03-15", f"{_y}-11-15"),
                                    sow=(f"{_y}-03-15", f"{_y}-05-31"), peak=(f"{_y}-07-01", f"{_y}-09-30"))


def kept():
    """field_handcheck results joined with the user's verdicts (only kept fields), or with the selection (SET sel)."""
    r = pd.read_parquet(hc.OUT / "results.parquet")
    if AUTO:
        s = pd.read_parquet(hc.OUT / SEL)[["wid", "solidity", "centrality", "area_rel"]]
        r = r.merge(s, on="wid")
    else:
        r = r.merge(pd.read_parquet(hc.OUT / "verdicts.parquet"), on="wid")
        r = r[r.verdict == "keep"].copy()
    r["area_diff_ha"] = r.matched_ha - r.NR_AREA_TOTAL
    r["area_diff_rel"] = r.matched_ratio - 1
    return r


def outline(wid, year):
    hc.fs.set_year(year, hc.OUT)
    m = np.load(hc.fs._mdir() / f"{wid}.npz")
    return np.isin(m["lab"], m["matched"])


def write_fields(case, k):
    """fields_used_<case>.gpkg + policies.parquet (field_id = wid) in the layout data.fields reads."""
    cfg = br.br_config(case)
    rows, masks = [], {}
    for r in k.itertuples():
        o = outline(r.wid, int(r.year))
        ys, xs = np.nonzero(o)
        mg = MARGIN_M // 10
        r0, r1 = max(ys.min() - mg, 0), min(ys.max() + mg + 1, o.shape[0])
        c0, c1 = max(xs.min() - mg, 0), min(xs.max() + mg + 1, o.shape[1])
        x0, y0 = r.x0 + c0 * 10, r.y1 - r0 * 10
        rows.append(dict(field_id=r.wid, x=r.x, y=r.y, x0=x0, y0=y0, nx=c1 - c0, ny=r1 - r0, GEOID=case))
        masks[r.wid] = o[r0:r1, c0:c1]
    w = pd.DataFrame(rows)
    g = gpd.GeoDataFrame(w, geometry=[box(r.x0, r.y0 - r.ny * 10, r.x0 + r.nx * 10, r.y0) for r in w.itertuples()],
                         crs=br.CRS)
    c = cache_dir(cfg)
    g.to_file(c / f"fields_used_{case}.gpkg", driver="GPKG")
    k.assign(field_id=k.wid).to_parquet(c / "policies.parquet")
    return cfg, masks


def write_masks(cfg, case, masks):
    """raw -> s2/<fid>.npz (mask = outline), s2bands/<fid>.npz, s2 rows (QA input), as data.build_fields."""
    s2_dir, b_dir, raw = cache_dir(cfg, "s2"), cache_dir(cfg, "s2bands"), cache_dir(cfg, "raw")
    rows = []
    for fid, m in masks.items():
        z = np.load(raw / f"{fid}.npz")
        dates, clear, refl, aot = z["dates"], z["clear"], z["refl"], z["aot"]
        x = refl.astype("float32") / 10000.0
        ev = np.where(clear, br._evi2(x), np.nan).astype("float32")
        T = len(dates)
        rgb = np.zeros((T, 3) + m.shape, "uint8")
        out_refl = np.zeros_like(refl)
        for t in range(T):
            cc = clear[t] & m
            cs = float(cc.sum() / m.sum())
            blue = red = e = np.nan
            if cc.any():
                out_refl[t][:, cc] = refl[t][:, cc]
                rgb[t] = np.clip(x[t][[2, 1, 0]] / 0.3 * 255, 0, 255).astype("uint8")
                blue, red, e = (float(np.median(x[t, 0][cc])), float(np.median(x[t, 2][cc])),
                                float(np.median(ev[t][cc])))
            ev[t][~cc] = np.nan
            rows.append((fid, str(dates[t]), cs, blue, red, e, float(aot[t])))
        cov = np.broadcast_to(m, (T,) + m.shape).copy()
        np.savez_compressed(s2_dir / f"{fid}.npz", dates=dates, evi2=ev, rgb=rgb, covered=cov, poly=m)
        np.savez_compressed(b_dir / f"{fid}.npz", dates=dates, bands=np.array(BANDS), refl=out_refl)
    df = pd.DataFrame(rows, columns=["field_id", "date", "clear_share", "blue", "red", "evi2", "aot"])
    df.to_parquet(cache_dir(cfg) / f"s2_rows_{case}.parquet")


def build(log=print):
    """Per year: fields, S2, masks, nights.  QA clear-sky line fitted once on all years' rows (one year alone
    has too few field-dates per red bin, e.g. 18 fields in 2020), then QA per year."""
    from data import s2qa
    k = kept()
    log(f"kept fields {len(k)}: " + str(pd.crosstab(k.year, k.group).to_dict("index")))
    cfgs = {}
    for y in YEARS:
        case = f"wheat{y}{SET}"
        ky = k[k.year == y]
        cfg, masks = write_fields(case, ky)
        br.sentinel2_raw(cfg, case, log=log)
        if not (cache_dir(cfg) / f"s2_rows_{case}.parquet").exists():
            write_masks(cfg, case, masks)
        n, fc = br.nights(cfg, case)
        log(f"{case}: {len(ky)} fields, {fc.cell_id.nunique()} ERA5 cells, trigger cell-nights {int(n.trigger.sum())}")
        cfgs[case] = cfg
    pooled = pd.concat([pd.read_parquet(cache_dir(c) / f"s2_rows_{case}.parquet") for case, c in cfgs.items()])
    if SET == "b4":                                       # same clear line as the full set the methods were built on
        pooled = pd.concat([pd.read_parquet(fp) for fp in
                            sorted(cache_dir(cfg).parent.glob("wheat20[0-9][0-9]full/s2_rows_*full.parquet"))])
        log(f"QA clear line from full-set rows: {len(pooled)}")
    br.clear_line = lambda df, mc, me: s2qa.clear_line(pooled, mc, me)
    for case, cfg in cfgs.items():
        br.qa(cfg, case, log=log)


if __name__ == "__main__":
    build(log=lambda *a, **kw: print(*a, flush=True))
