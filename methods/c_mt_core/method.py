"""c_mt_core: stage-locked damage evidence alone (Module C) with a linear calibration (doc 3.5).
L_p = clip(b0 + b1 E_p + b2 X~_k, 0, 1); b by least squares on c_r1a targets, negative b1 / b2 set to 0 and refit.
Clock: dev-split imagery only (user decision 2026-11-09).  See README.md."""
from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from methods.c_mt import fuse as D

importlib.reload(D)
CFG = yaml.safe_load(Path(__file__).with_name("settings.yaml").read_text())


def fit(fields, labels):
    D.fit(fields, labels, CFG, "c_mt_core")


def predict(f):
    return D.predict(f)
