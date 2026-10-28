"""Step 0 sanity check: potato area, field terrain and trigger-night counts per candidate county."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

from . import csb, dem
from .common import cache_dir, load_config


def field_terrain(fields: gpd.GeoDataFrame, dem_fp) -> pd.DataFrame:
    rows = []
    with rasterio.open(dem_fp) as src:
        for geom in fields.geometry:
            a, _ = mask(src, [geom], crop=True, filled=True, nodata=np.nan)
            v = a[0][np.isfinite(a[0])]
            if v.size == 0:
                rows.append((np.nan,) * 4)
                continue
            rows.append((v.min(), v.max(), np.median(v), np.percentile(v, 95) - np.percentile(v, 5)))
    return pd.DataFrame(rows, columns=["elev_min", "elev_max", "elev_median", "relief_p5_p95"])


def report(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    cells = gpd.read_file(cache_dir(cfg) / "rtma_cells.gpkg")
    nights = pd.read_parquet(cache_dir(cfg) / "nights.parquet")
    out = []
    for name, fips in cfg["study"]["candidate_counties"].items():
        f = csb.potato_fields(cfg, fips)
        t = field_terrain(f, dem.county_dem(cfg, fips))
        area_ha = f.area / 1e4
        pc = cells.to_crs(f.crs)
        pc = pc[pc.intersects(f.union_all())]
        cell_nights = nights[nights.cell_id.isin(pc.cell_id)]
        g = cell_nights.groupby("night")
        out.append({
            "county": name, "fips": fips,
            "potato_fields": len(f), "potato_area_ha": round(area_ha.sum()),
            "fields_ge_4ha_raw": int((area_ha >= cfg["data"]["min_field_area_ha"]).sum()),
            "field_elev_min_m": round(t.elev_min.min()), "field_elev_max_m": round(t.elev_max.max()),
            "field_median_elev_p5_p95_m": f"{t.elev_median.quantile(.05):.0f}-{t.elev_median.quantile(.95):.0f}",
            "within_field_relief_median_m": round(t.relief_p5_p95.median(), 1),
            "within_field_relief_max_m": round(t.relief_p5_p95.max(), 1),
            "rtma_cells_with_potato": len(pc),
            "trigger_nights_any_cell": int(g.trigger.any().sum()),
            "trigger_cell_nights": int(cell_nights.trigger.sum()),
            "near_trigger_nights_any_cell": int(g.near_trigger.any().sum()),
            "near_trigger_cell_nights": int(cell_nights.near_trigger.sum()),
            "nights_missing_hours": int((g.n_hours_missing.max() > 0).sum()),
        })
    return pd.DataFrame(out).set_index("county").T


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(report())
