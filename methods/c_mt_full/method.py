"""c_mt_full: Module B prior (time axis by rule R1) fused with Module C evidence (doc 3.5).
L_p = sigmoid(logit(clip(A_p, .02, .98)) + beta1 E_p + beta2 X~_k); beta by OOF-prior censored C grid.
Clock: dev-split imagery only (user decision 2026-11-09).  See README.md."""
from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from methods.c_mt import fuse as D

importlib.reload(D)
CFG = yaml.safe_load(Path(__file__).with_name("settings.yaml").read_text())


def fit(fields, labels):
    D.fit(fields, labels, CFG, "c_mt_full")


def predict(f):
    return D.predict(f)
