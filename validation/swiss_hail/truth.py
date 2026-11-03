"""Swiss Hail crop claims (Portmann et al. 2024, NHESS 24:2541) -> per event x 1 km cell truth table.

Open data (CC-BY-4.0, Zenodo):
  damage   10.5281/zenodo.11064767  number of damaged fields per grid cell and hail day (claims assessed in the
                                    field by Swiss Hail staff; zero-loss claims removed; ~24 % of claims placed at
                                    random inside their municipality's cropland)
  exposure 10.5281/zenodo.11064756  number of fields and crop area per grid cell (2021 land use)
  hazard   10.5281/zenodo.11064781  radar MESHS (mm) and POH (%), daily max 06-06 UTC, ~1 km grid

Grids share the cell centroids of the exposure file; damage rows index into it (CLIMADA Impact, CSR matrix
event x centroid).  Insurance coverage ~69 % of field crops, 43 % of grapevine: damaged counts are a lower bound.

    python -m validation.swiss_hail.truth          # download (once) -> cache/swiss_hail/truth_<res>km.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from data.common import ROOT

RAW = ROOT / "cache" / "swiss_hail" / "raw"
OUT = ROOT / "cache" / "swiss_hail"
RECORDS = {"damage": "11064767", "exposure": "11064756", "hazard": "11064781"}
CROPS = ["field_crops", "wheat", "maize", "barley", "rapeseed", "grapevine"]
HAZARD = {"MESHS": "hail_MESHS_CH_12_events_2017_2021.h5", "POH": "hail_POH_CH_12_events_2017_2021.h5"}


def _get(url, dest: Path):
    if dest.exists():
        return
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)


def download():
    for kind in ("damage", "exposure"):
        d = RAW / kind
        if not any(d.glob("*.h5")):
            z = RAW / RECORDS[kind] / "data-files.zip"
            _get(f"https://zenodo.org/api/records/{RECORDS[kind]}/files/data-files.zip/content", z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
    for fn in HAZARD.values():
        _get(f"https://zenodo.org/api/records/{RECORDS['hazard']}/files/{fn}/content", RAW / RECORDS["hazard"] / fn)


def _ordinal(d):
    return dt.date.fromordinal(int(d))


def read_exposure(crop, res):
    with h5py.File(RAW / "exposure" / f"exposure_CH_{crop}_{res}km.h5") as f:
        g = f["exposures"]
        items = [x.decode() for x in g["block0_items"][()]]
        v = pd.DataFrame(g["block0_values"][()], columns=items)
    return v[["latitude", "longitude", "n_fields", "area_ha"]]


def read_damage(crop, res):
    """-> (events DataFrame, CSR (data, indices, indptr), centroid lat/lon (n, 2))."""
    with h5py.File(RAW / "damage" / f"hail_damage_CH_{crop}_{res}km.h5") as f:
        ev = pd.DataFrame({"event_id": f["event_id"][()],
                           "date": [_ordinal(d) for d in f["date"][()]],
                           "event_name": [s.decode() for s in f["event_name"][()]],
                           "n_damaged_total": f["at_event"][()]})
        csr = tuple(f["imp_mat"][k][()] for k in ("data", "indices", "indptr"))
        coord = f["coord_exp"][()]
    return ev, csr, coord


def read_hazard(var):
    with h5py.File(RAW / RECORDS["hazard"] / HAZARD[var]) as f:
        lat, lon = f["centroids/lat"][()], f["centroids/lon"][()]
        csr = tuple(f["intensity"][k][()] for k in ("data", "indices", "indptr"))
        dates = [_ordinal(d) for d in f["date"][()]]
    return lat, lon, csr, dates


def build(res=1):
    """Long table: one row per (crop, event, exposed cell) with n_fields, n_damaged, frac_damaged, MESHS, POH."""
    download()
    haz = {v: read_hazard(v) for v in HAZARD}
    out = []
    for crop in CROPS:
        exp = read_exposure(crop, res)
        ev, (data, ind, ptr), coord = read_damage(crop, res)
        # same cells, different order: put the exposure rows into damage-centroid order
        dist, order = cKDTree(exp[["latitude", "longitude"]].to_numpy()).query(coord)
        assert len(coord) == len(exp) and dist.max() < 1e-9 and len(set(order)) == len(order)
        exp = exp.iloc[order].reset_index(drop=True)
        hz = {}
        for v, (lat, lon, (hd, hi, hp), hdates) in haz.items():
            nn = cKDTree(np.column_stack([lat, lon])).query(coord)[1]   # nearest radar pixel (~1 km grid)
            hz[v] = (nn, hd, hi, hp, hdates)
        for e, row in ev.iterrows():
            nd = np.zeros(len(exp))
            sl = slice(ptr[e], ptr[e + 1])
            nd[ind[sl]] = data[sl]
            t = exp.copy()
            t.insert(0, "cell", np.arange(len(exp)))
            t.insert(0, "date", row["date"])
            t.insert(0, "event_id", int(row["event_id"]))
            t.insert(0, "crop", crop)
            t["n_damaged"] = nd
            t["frac_damaged"] = np.clip(nd / t["n_fields"].where(t["n_fields"] > 0), 0, 1)
            for v, (nn, hd, hi, hp, hdates) in hz.items():
                k = hdates.index(row["date"])            # crops have 10-12 of the 12 radar days
                grid = np.zeros(len(haz[v][0]), "float32")
                s2 = slice(hp[k], hp[k + 1])
                grid[hi[s2]] = hd[s2]
                t[v] = grid[nn]
            out.append(t)
    df = pd.concat(out, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"truth_{res}km.parquet")
    return df


def summary(df):
    g = df.groupby(["crop", "date"])
    s = pd.DataFrame({"cells_exposed": g.size(),
                      "cells_damaged": g["n_damaged"].apply(lambda x: int((x > 0).sum())),
                      "fields_damaged": g["n_damaged"].sum().astype(int),
                      "cells_meshs>0": g["MESHS"].apply(lambda x: int((x > 0).sum())),
                      "median_frac_in_damaged_cells": g.apply(lambda x: x.loc[x.n_damaged > 0, "frac_damaged"].median(),
                                                              include_groups=False).round(3)})
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=1, choices=[1, 2, 4, 8])
    a = ap.parse_args()
    df = build(a.res)
    pd.set_option("display.width", 200, "display.max_rows", 100)
    print(summary(df).to_string())
