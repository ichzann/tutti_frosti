"""runs/<id>/report.html: a static page that only reads (embeds) run outputs.

    python -m report.build runs/<id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TEMPLATE = Path(__file__).with_name("template.html")
FN_COLS = ["valid_share", "mean_z", "share_damaged", "growth_stage", "rendered", "pre_date",
           "post_date", "next_date", "gap_days_before", "gap_days_after", "next_days_after",
           "dose", "trigger", "near_trigger", "n_valid", "n_px", "mean_score", "tmin_c", "cell_id"]


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 3)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def build(run_dir) -> Path:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    meta.update(season_start=cfg["study"]["season_start"], emergence_evi2=cfg["growth"]["emergence_evi2"])
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
            cell={r.cell_id: [_clean(r.tmin_c), _clean(r.dose_degree_hours), _clean(r.wind_mean_ms),
                              _clean(r.gust_max_ms), _clean(r.cloud_mean_pct), _clean(r.dewpoint_sunset_c)]
                  for r in g.itertuples()})
    fn = pd.read_parquet(run_dir / "field_nights.parquet")
    fnd = {}
    for r in fn.itertuples():
        fnd.setdefault(r.night, {})[r.field_id] = [_clean(getattr(r, c)) for c in FN_COLS]
    acq = pd.read_parquet(run_dir / "s2_acquisitions.parquet")
    data = dict(meta=meta, nights=nd, fn=fnd, fn_cols=FN_COLS,
                fields=json.loads((run_dir / "fields.geojson").read_text()), cells=cells,
                detail=json.loads((run_dir / "fields_detail.json").read_text()),
                acq=[[r.date, _clean(r.clear_frac)] for r in acq.itertuples()])
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out = run_dir / "report.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    print(build(sys.argv[1]))
