"""Baseline: largest drop of the field-median NDMI between consecutive usable dates in the season."""
from competition.baselines._common import as_map, index_series

import numpy as np


def predict(f):
    s = index_series(f, "ndmi")
    return as_map(f, float(np.max(s[:-1] - s[1:])) if len(s) > 1 else np.nan)
