"""What a method gets: one field of the competition package (FRAME.md section 2), loaded lazily.

    from competition.field import load_fields, dev_labels
    fields = load_fields("dev")              # list[Field], arrays read from disk on first access
    f = fields[0]; f.refl.shape              # (T, 10, H, W) float32 reflectance 0..1, NaN where not clear
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from competition.harness import DATA
from data.s2bands import BANDS


class Field:
    """field_id, year, and (lazily) dates, refl, clear, usable, outline.  Call drop() to free the arrays."""

    bands = list(BANDS)                       # B02 B03 B04 B05 B06 B07 B08 B8A B11 B12

    def __init__(self, field_id: str, year: int):
        self.field_id, self.year = field_id, int(year)

    @cached_property
    def _z(self):
        z = np.load(DATA / "fields" / f"{self.field_id}.npz")
        return {k: z[k] for k in z.files}

    @cached_property
    def dates(self) -> np.ndarray:
        """(T,) datetime64[D], ascending."""
        return self._z["dates"].astype("datetime64[D]")

    @cached_property
    def clear(self) -> np.ndarray:
        """(T, H, W) bool: pixel clear in the scene classification."""
        return self._z["clear"]

    @cached_property
    def usable(self) -> np.ndarray:
        """(T,) bool: scene passes the field QA (enough clear outline pixels, no haze, no high aerosol)."""
        return self._z["usable"]

    @cached_property
    def outline(self) -> np.ndarray:
        """(H, W) bool: field pixels.  The window around it is outline bbox + 300 m."""
        return self._z["outline"]

    @cached_property
    def refl(self) -> np.ndarray:
        """(T, 10, H, W) float32 reflectance, NaN where the pixel is not clear."""
        x = self._z["refl"].astype("float32") / 10000.0
        x[np.broadcast_to(~self.clear[:, None], x.shape)] = np.nan
        return x

    def band(self, name: str) -> np.ndarray:
        """(T, H, W) one band by name, e.g. f.band("B08")."""
        return self.refl[:, self.bands.index(name)]

    def drop(self):
        for k in ("_z", "dates", "clear", "usable", "outline", "refl"):
            self.__dict__.pop(k, None)

    def __repr__(self):
        return f"Field({self.field_id}, {self.year})"


def field_table() -> pd.DataFrame:
    return pd.read_parquet(DATA / "fields.parquet")


def load_fields(split: str = "dev") -> list[Field]:
    t = field_table()
    t = t[t.split == split]
    return [Field(r.field_id, r.year) for r in t.itertuples()]


def dev_labels() -> pd.DataFrame:
    """field_id, year, group (none / mild / medium / heavy), claim, loss (NaN for no-claim = censored < 0.35)."""
    return pd.read_parquet(DATA / "labels_dev.parquet")
