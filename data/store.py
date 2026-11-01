"""Read-only access to preprocessed inputs, handed to methods (methods never download)."""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd

from .common import cache_dir
from .fields import field_mask


class FieldStore:
    def __init__(self, cfg, fips: str | None = None):
        self.cfg = cfg
        self.fips = fips or cfg["study"]["county_fips"]
        c = cache_dir(cfg)
        f = gpd.read_file(c / f"fields_used_{self.fips}.gpkg")
        # final field pixels = buffered polygon AND CDL potato AND confidence rule
        self._mask = {r.field_id: field_mask(cfg, r) for r in f.itertuples()}
        self.fields = f[[self._mask[i].any() for i in f.field_id]].reset_index(drop=True)
        self.nights = pd.read_parquet(c / "nights.parquet")
        cells = gpd.read_file(c / "rtma_cells.gpkg").to_crs(self.fields.crs)
        # each field is assigned the RTMA cell containing its centroid
        pts = gpd.GeoDataFrame(self.fields[["field_id"]], geometry=self.fields.representative_point(),
                               crs=self.fields.crs)
        j = gpd.sjoin(pts, cells[["cell_id", "geometry"]], predicate="within", how="left")
        self.field_cell = dict(zip(j.field_id, j.cell_id))
        self._n = self.nights.set_index(["cell_id", "night"])

    def field_ids(self):
        return list(self.fields.field_id)

    @lru_cache(maxsize=64)
    def field(self, field_id):
        z = np.load(cache_dir(self.cfg, "s2") / f"{field_id}.npz")
        d = {k: z[k] for k in z.files}
        d["dates"] = [dt.date.fromisoformat(s) for s in d["dates"]]
        d["mask"] = self._mask[field_id]
        return d

    def night_meta(self, field_id, night):
        cell = self.field_cell[field_id]
        r = self._n.loc[(cell, dt.date.fromisoformat(str(night)).isoformat())]
        return dict(cell_id=cell, dose=float(r.dose_degree_hours), tmin_c=float(r.tmin_c),
                    trigger=bool(r.trigger), near_trigger=bool(r.near_trigger), regime=r.regime)
