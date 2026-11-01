"""runs/<id>/report.html: a static page that only reads (embeds) run outputs.

    python -m report.build runs/<id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml

from data.ndmi import NDMI_RANGE

from .pixels import STOPS, Z_MIN, build_pixel_layers

TEMPLATE = Path(__file__).with_name("template.html")
FN_COLS = ["valid_share", "mean_z", "share_damaged", "growth_stage", "rendered", "pre_date",
           "post_date", "next_date", "gap_days_before", "gap_days_after", "next_days_after",
           "dose", "regime", "trigger", "near_trigger", "n_valid", "n_px", "mean_score", "tmin_c", "cell_id"]


def _clean(v, nd=3):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), nd)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _geojson(fp, simplify_m=None):
    """GeoJSON for the page: optionally simplified (in UTM metres), coordinates rounded to ~0.1 m."""
    g = gpd.read_file(fp)
    if simplify_m:
        g["geometry"] = g.to_crs(32612).simplify(simplify_m).to_crs(4326)
    g["geometry"] = shapely.set_precision(g.geometry, 1e-6)
    return json.loads(g.to_json(drop_id=True))


def _trim_curves(detail, fn, season_start):
    """GPR curves are evaluated over the whole season; the page draws them only up to post_date + 2 d."""
    s0 = pd.Timestamp(season_start)
    end = {(r.field_id, r.night): (pd.Timestamp(r.post_date) - s0).days + 2
           for r in fn.itertuples() if r.post_date}
    for fid, d in detail.items():
        for n, c in d["curves"].items():
            k = sum(day <= end.get((fid, n), 1e9) for day in c["day"])
            d["curves"][n] = {key: v[:k] for key, v in c.items()}
    return detail


def build(run_dir) -> Path:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    meta.update(season_start=cfg["study"]["season_start"], emergence_evi2=cfg["growth"]["emergence_evi2"],
                z_damaged=cfg["damage"]["z_damaged"], ndmi_range=NDMI_RANGE)
    nights = pd.read_parquet(run_dir / "nights.parquet")
    cells = json.loads((run_dir / "rtma_cells.geojson").read_text())
    pot = {f["properties"]["cell_id"] for f in cells["features"] if f["properties"]["has_potato"]}
    nd = {}
    for night, g in nights.groupby("night"):
        gp = g[g.cell_id.isin(pot)]
        nd[night] = dict(
            trig=g.cell_id[g.trigger].tolist(), near=g.cell_id[g.near_trigger].tolist(),
            ptrig=int(gp.trigger.sum()), pnear=int(gp.near_trigger.sum()),
            maxdose=_clean(gp.dose_degree_hours[gp.trigger].max()) if gp.trigger.any() else 0,
            tmin=_clean(gp.tmin_c.min()),
            cell={r.cell_id: [_clean(r.tmin_c, 1), _clean(r.dose_degree_hours, 1), _clean(r.wind_pre_tmin_ms, 1),
                              _clean(r.cloud_pre_tmin_pct, 0), _clean(r.dewpoint_sunset_c, 1), r.regime,
                              _clean(r.trigger_thr_c, 1)]
                  for r in gp.itertuples()})   # weather tooltips only for cells with potato fields
    fn = pd.read_parquet(run_dir / "field_nights.parquet")
    fnd = {}
    for r in fn.itertuples():
        fnd.setdefault(r.night, {})[r.field_id] = [_clean(getattr(r, c)) for c in FN_COLS]
    acq = pd.read_parquet(run_dir / "s2_acquisitions.parquet")
    data = dict(meta=meta, nights=nd, fn=fnd, fn_cols=FN_COLS,
                fields=_geojson(run_dir / "fields.geojson", simplify_m=3), cells=_geojson(run_dir / "rtma_cells.geojson"),
                detail=_trim_curves(json.loads((run_dir / "fields_detail.json").read_text()), fn, meta["season_start"]),
                acq=[[r.date, _clean(r.clear_frac)] for r in acq.itertuples()],
                pixel_bounds=build_pixel_layers(run_dir),
                zmap=dict(z_min=Z_MIN, stops=STOPS.astype(int).tolist()))
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out = run_dir / "report.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    print(build(sys.argv[1]))
