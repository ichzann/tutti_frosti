"""Field-median index series on usable dates (baselines: field number spread over the outline)."""
from __future__ import annotations

import numpy as np


def index_series(f, kind):
    nir, red, swir = f.band("B08"), f.band("B04"), f.band("B11")
    x = {"ndmi": (nir - swir) / (nir + swir), "evi2": 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)}[kind]
    med = np.array([np.nanmedian(x[t][f.outline]) if f.usable[t] and np.isfinite(x[t][f.outline]).any()
                    else np.nan for t in range(len(f.dates))])
    return med[np.isfinite(med)]


def as_map(f, v):
    return np.where(f.outline, np.float32(v), np.float32(np.nan))
