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
        # field-level acquisition QA (data.s2qa): dates not usable for a field are hidden whole
        qa_fp = c / f"s2_qa_{self.fips}.parquet"
        use_qa = cfg.get("s2_qa", {}).get("enabled", True)
        qa = pd.read_parquet(qa_fp) if use_qa and qa_fp.exists() else None
        self._drop = ({} if qa is None else
                      qa[~qa.usable].groupby("field_id").apply(
                          lambda g: dict(zip(g.date, g.drop_reason))).to_dict())

    def field_ids(self):
        return list(self.fields.field_id)

    @lru_cache(maxsize=64)
    def field(self, field_id):
        z = np.load(cache_dir(self.cfg, "s2") / f"{field_id}.npz")
        d = {k: z[k] for k in z.files}
        d["dates"] = [dt.date.fromisoformat(s) for s in d["dates"]]
        d["mask"] = self._mask[field_id]
        drop = self._drop.get(field_id, {})
        d["drop_reason"] = [drop.get(x.isoformat(), "") for x in d["dates"]]
        for t, r in enumerate(d["drop_reason"]):
            if r:
                d["evi2"][t] = np.nan   # the method then sees no clear pixel on this date
        return d

    @lru_cache(maxsize=16)
    def bands(self, field_id):
        """Surface reflectance (T, 10, H, W) float32 in data.s2bands.BANDS order, aligned with
        field()['dates']; NaN wherever field()['evi2'] is NaN (not clear, or date dropped by QA)."""
        z = np.load(cache_dir(self.cfg, "s2bands") / f"{field_id}.npz")
        r = z["refl"].astype("float32") / 10000.0
        r[np.broadcast_to(~np.isfinite(self.field(field_id)["evi2"])[:, None], r.shape)] = np.nan
        return r

    def daily_temp(self, field_id):
        """gridMET daily tmax_c / tmin_c (DataFrame indexed by date) of the cell nearest the field's
        representative point; None if cache/gridmet_<fips>.parquet (python -m data.gridmet) is missing."""
        if not hasattr(self, "_gm"):
            fp = cache_dir(self.cfg) / f"gridmet_{self.fips}.parquet"
            self._gm = pd.read_parquet(fp) if fp.exists() else None
            if self._gm is not None:
                cells = self._gm[["lat", "lon"]].drop_duplicates().to_numpy()
                pts = self.fields.representative_point().to_crs(4326)
                d = ((pts.y.to_numpy()[:, None] - cells[None, :, 0]) ** 2 +
                     ((pts.x.to_numpy()[:, None] - cells[None, :, 1]) * np.cos(np.radians(43.0))) ** 2)
                near = cells[d.argmin(1)]
                self._gm_cell = {fid: (la, lo) for fid, (la, lo) in zip(self.fields.field_id, near)}
                self._gm_by = {k: g.set_index("date")[["tmax_c", "tmin_c"]].sort_index()
                               for k, g in self._gm.groupby(["lat", "lon"])}
        if self._gm is None:
            return None
        return self._gm_by[self._gm_cell[field_id]]

    def night_meta(self, field_id, night):
        cell = self.field_cell[field_id]
        r = self._n.loc[(cell, dt.date.fromisoformat(str(night)).isoformat())]
        return dict(cell_id=cell, dose=float(r.dose_degree_hours), tmin_c=float(r.tmin_c),
                    trigger=bool(r.trigger), near_trigger=bool(r.near_trigger), regime=r.regime)
