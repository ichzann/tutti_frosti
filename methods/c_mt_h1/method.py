"""c_mt_h1: c_r6a prior (Module B, methods/c_mt/prior.py) on frost-relative time DOY' = DOY - delta_y.
delta_y from the Module A clock (dev-split imagery only, user decision 2026-11-09).  See README.md."""
from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from methods.c_mt import clock as K
from methods.c_mt import prior as P

importlib.reload(P)
CFG = yaml.safe_load(Path(__file__).with_name("settings.yaml").read_text())


def fit(fields, labels):
    delta = K.delta_table(CFG["time_axis"], CFG.get("perm_seed"))
    P.set_delta(delta, CFG["time_axis"] + ("" if CFG.get("perm_seed") is None else f"_perm{CFG['perm_seed']}"))
    P.fit(fields, labels)


def predict(f):
    return P.predict(f)
