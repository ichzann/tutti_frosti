"""Falsification checks F1-F5 and F8 (doc 4.3): single-feature C of the field-mean evidence E on all dev fields
(no fit; label-free features; scorer pair logic competition.score._pairs / _c).  Also X (exposure) alone.

    python -m methods.c_mt.fchecks
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from competition import score as SC
from competition.field import dev_labels
from methods.c_mt import common as C
from methods.c_mt import evidence as E


def field_E(D, kappa):
    with np.errstate(all="ignore"):
        return np.nanmean(np.log1p(D / kappa)) if np.isfinite(D).any() else np.nan


def table():
    lab = dev_labels().reset_index(drop=True)
    raw = {fid: dict(np.load(E.cache_file(fid))) for fid in lab.field_id}
    rows = []
    for cf in E.CONFIGS:
        Dm = np.array([np.nanmean(raw[f][cf]) if np.isfinite(raw[f][cf]).any() else np.nan for f in lab.field_id])
        kappa = np.nanmedian(Dm[Dm > 0])
        e = np.array([field_E(raw[f][cf], kappa) for f in lab.field_id])
        rows.append((cf, kappa, e))
    X = np.array([float(raw[f]["X"]) for f in lab.field_id])
    out = {}
    for name, kappa, e in rows + [("exposure_X", np.nan, X)]:
        d = lab.copy()
        d["score"] = np.where(np.isfinite(e), e, np.nanmean(e))
        es = (d.score - d.score.mean()) / d.score.std()          # standardised (doc E / X~ scale), all dev fields
        A, K, PN = SC._pairs(d)
        same = (d.year.to_numpy()[:, None] == d.year.to_numpy()[None, :]).astype(float)
        _, z, p = SC._jt(d)
        out[name] = dict(C=SC._c(A, K), C_within_year=SC._c(A * same, K * same), AUC=SC._c(PN, K * PN),
                         kappa=float(kappa), n_nan=int((~np.isfinite(e)).sum()),
                         median_by_group={g: float(es[d.group == g].median()) for g in
                                          ["none", "mild", "medium", "heavy"]}, jt_z=z)
    return out


if __name__ == "__main__":
    t = table()
    lines = ["config | C | C within-year | AUC | kappa | n NaN | median E none/mild/medium/heavy | JT z"]
    for k, v in t.items():
        mg = " / ".join(f"{x:.3f}" for x in v["median_by_group"].values())
        lines.append(f"{k} | {v['C']:.4f} | {v['C_within_year']:.4f} | {v['AUC']:.4f} | {v['kappa']:.4g} | "
                     f"{v['n_nan']} | {mg} | {v['jt_z']:.2f}")
    print("\n".join(lines))
    (C.HERE / "report" / "fchecks.txt").write_text("\n".join(lines) + "\n")
    (C.HERE / "report" / "fchecks.json").write_text(json.dumps(t, indent=1))
