"""Leak and overfitting checks for the second lockbox (LOCKBOX2.md), harness only.

    COMP_DIR=competition_b4 python -m competition.lockbox2_checks overlap   # batch-4 vs package outlines / points
    COMP_DIR=competition_b4 python -m competition.lockbox2_checks report    # after all lockbox runs

overlap: per batch-4 field, outline pixels (10 m, UTM) shared with any competition field (dev or old lockbox, any
year) as a share of the batch-4 outline, and the distance to the nearest dev policy point.
report: per method dev LOYO M1, lockbox-1 M1, lockbox-2 M1 [95 %], label-permutation p (loss / claim / group
shuffled within year, PERM times), M1 without flagged fields (point < 1 km from a dev point or outline overlap
> 0), paired M1 differences vs c_r6a.  Writes competition_b4/report.json.
"""
from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from competition import score as sc
from competition.harness import C, HARN
from data.common import ROOT

OLD = ROOT / "competition"
BR = ROOT / "cache" / "brazil_psr"
PERM = 2000
NEAR_M = 1000
METHODS = ["c_r7a", "c_mt_full", "c_r6a", "c_r7b", "c_r4b", "low_peak_evi2", "ndmi_drop", "era5_coldest",
           "frost_recovery_v2", "frost_recovery"]


def _pixels(case_dir, wid):
    case = case_dir.name
    g = gpd.read_file(case_dir / f"fields_used_{case}.gpkg").set_index("field_id")
    r = g.loc[wid]
    o = np.load(case_dir / "s2" / f"{wid}.npz")["poly"]
    i, j = np.nonzero(o)
    return set(zip((r.x0 + j * 10).astype(int).tolist(), (r.y0 - i * 10).astype(int).tolist()))


def overlap():
    old = pd.read_parquet(OLD / "_harness" / "split.parquet")
    new = pd.read_parquet(HARN / "split.parquet")
    pts = {}
    for d in BR.glob("wheat20[0-9][0-9]full"):
        p = pd.read_parquet(d / "policies.parquet")
        pts.update({w: (x, y) for w, x, y in zip(p.wid, p.x, p.y)})
    for d in BR.glob("wheat20[0-9][0-9]b4"):
        p = pd.read_parquet(d / "policies.parquet")
        pts.update({w: (x, y) for w, x, y in zip(p.wid, p.x, p.y)})
    pix_old = {}
    for r in old.itertuples():
        pix_old[r.wid] = _pixels(BR / r.case, r.wid)
    owner = {}
    for w, s in pix_old.items():
        for px in s:
            owner.setdefault(px, set()).add(w)
    split_of = old.set_index("wid").split
    dev_xy = np.array([pts[w] for w in old.wid[old.split == "dev"]])
    rows = []
    for r in new.itertuples():
        s = _pixels(BR / r.case, r.wid)
        hit = {}
        for px in s:
            for w in owner.get(px, ()):
                hit[w] = hit.get(w, 0) + 1
        best = max(hit, key=hit.get) if hit else None
        x, y = pts[r.wid]
        rows.append(dict(field_id=r.field_id, wid=r.wid, year=r.year, n_px=len(s),
                         overlap_share=hit[best] / len(s) if best else 0.0, overlap_wid=best,
                         overlap_split=split_of[best] if best else "",
                         dist_dev_m=float(np.hypot(dev_xy[:, 0] - x, dev_xy[:, 1] - y).min())))
    t = pd.DataFrame(rows)
    t["flag"] = (t.overlap_share > 0) | (t.dist_dev_m < NEAR_M)
    t.to_parquet(HARN / "overlap.parquet")
    print(t.sort_values("overlap_share", ascending=False).head(10).to_string())
    print(f"overlap > 0: {(t.overlap_share > 0).sum()} (dev {((t.overlap_share > 0) & (t.overlap_split == 'dev')).sum()}),"
          f" > 50 %: {(t.overlap_share > .5).sum()}, point < {NEAR_M} m from dev: {(t.dist_dev_m < NEAR_M).sum()},"
          f" flagged {t.flag.sum()}")


def _perm_p(d, rng):
    d = d.reset_index(drop=True)
    A, K, _ = sc._pairs(d)
    obs = sc._c(A, K)
    lab = d[["claim", "loss", "group"]]
    hits = 0
    for _ in range(PERM):
        idx = np.arange(len(d))
        for _, g in d.groupby("year").groups.items():
            g = np.asarray(g)
            idx[g] = rng.permutation(g)
        p = d.copy()
        p[["claim", "loss", "group"]] = lab.iloc[idx].to_numpy()
        p["claim"] = p.claim.astype(bool)
        p["loss"] = p.loss.astype(float)
        a, k, _ = sc._pairs(p)
        hits += sc._c(a, k) >= obs
    return (hits + 1) / (PERM + 1)


def _m1(name, split, base=OLD):
    fp = base / "scores" / f"{name}__{split}.json"
    return json.loads(fp.read_text())["M1_C"] if fp.exists() else None


def report():
    flag = pd.read_parquet(HARN / "overlap.parquet").set_index("field_id").flag
    rng = np.random.default_rng(11)
    out = {}
    for m in METHODS:
        fp = C / "scores" / f"{m}__lockbox.parquet"
        if not fp.exists():
            print(m, "no lockbox-2 run")
            continue
        d = pd.read_parquet(fp)
        met = json.loads((C / "scores" / f"{m}__lockbox.json").read_text())
        keep = ~d.field_id.map(flag).fillna(False).to_numpy(bool)
        a, k, _ = sc._pairs(d[keep].reset_index(drop=True))
        out[m] = dict(dev=_m1(m, "dev"), lockbox1=_m1(m, "lockbox"), lockbox2=met["M1_C"], lockbox2_ci=met["M1_ci"],
                      M2=met["M2_C_within_year"], M4=met["M4_AUC"], jt_z=met["M3_jt_z"], M3=met["M3_median"],
                      per_year=met["per_year_C"], n=met["n"], n_errors=met["n_errors"], n_nan=met["n_nan"],
                      perm_p=_perm_p(d, rng), lockbox2_unflagged=sc._c(a, k), n_unflagged=int(keep.sum()))
        r = out[m]
        print(f"{m:18s} dev {r['dev'] or float('nan'):.3f} lb1 {r['lockbox1'] or float('nan'):.3f} "
              f"lb2 {r['lockbox2']:.3f} [{r['lockbox2_ci'][0]:.3f}, {r['lockbox2_ci'][1]:.3f}] M2 {r['M2']:.3f} "
              f"AUC {r['M4']:.3f} perm p {r['perm_p']:.4f} | unflagged {r['lockbox2_unflagged']:.3f} "
              f"(n {r['n_unflagged']}) err {r['n_errors']} nan {r['n_nan']}")
    cmp = {}
    for m in out:
        if m != "c_r6a" and "c_r6a" in out:
            cmp[m] = sc.compare(m, "c_r6a", "lockbox")
    for b in ("low_peak_evi2", "ndmi_drop", "era5_coldest"):
        if b in out and "c_r6a" in out:
            cmp[f"c_r6a-{b}"] = sc.compare("c_r6a", b, "lockbox")
    (C / "report.json").write_text(json.dumps(dict(methods=out, compare=cmp), indent=1, default=float))


if __name__ == "__main__":
    {"overlap": overlap, "report": report}[sys.argv[1]]()
