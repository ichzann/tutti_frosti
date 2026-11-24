"""Baseline: 1 - season maximum of the field-median EVI2 (low peak = damage)."""
from competition.baselines._common import as_map, index_series

import numpy as np


def predict(f):
    s = index_series(f, "evi2")
    return as_map(f, 1.0 - float(np.max(s)) if len(s) else np.nan)
