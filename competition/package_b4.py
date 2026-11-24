"""Second lockbox package (LOCKBOX2.md): the competition's 275 dev fields + the 99 batch-4 fields as lockbox.

    BR_SET=b4 python -m validation.brazil_psr.handcheck_store      # first: batch-4 stores (cases wheat<year>b4)
    COMP_DIR=competition_b4 python -m competition.package_b4      # build the package
    COMP_DIR=competition_b4 python -m competition.score lockbox <method>

Dev fields keep their competition ids, labels and npz files (hard links), so fit() sees exactly the competition
dev set.  Batch-4 fields get ids C2001.. in random order (no collision with method caches keyed by field id).
Old lockbox fields are not in the package.  _harness/split.parquet holds only the batch-4 rows (labels, ERA5).
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd

from competition import harness as h
from data.common import ROOT

OLD = ROOT / "competition"
SEED = 20261111


def build():
    if h.C == OLD:
        sys.exit("set COMP_DIR=competition_b4 (never write into the competition package)")
    sp_fp = h.HARN / "split.parquet"
    if sp_fp.exists():
        sys.exit(f"{sp_fp} exists; batch-4 ids are drawn once")
    src = h._sources("b4").sort_values(["year", "wid"]).reset_index(drop=True)
    sel = pd.read_parquet(ROOT / "cache/brazil_psr/handcheck_wheat/selection_b4.parquet")
    missing = sorted(set(sel.wid) - set(src.wid))
    print(f"batch-4 fields with store data: {len(src)} of {len(sel)}; missing {missing}")
    src["field_id"] = [f"C{2001 + i:04d}" for i in np.random.default_rng(SEED).permutation(len(src))]
    src["split"] = "lockbox"
    unk = (src.group != "none") & ~np.isfinite(src.loss)
    src.loc[unk, "group"] = "paid_unknown"
    src["era5_tmin"] = src.wid.map(h.coldest_night(src))
    (h.DATA / "fields").mkdir(parents=True, exist_ok=True)
    h.HARN.mkdir(parents=True, exist_ok=True)

    old = pd.read_parquet(OLD / "data" / "fields.parquet")
    dev = old[old.split == "dev"]
    for fid in dev.field_id:
        dst = h.DATA / "fields" / f"{fid}.npz"
        if not dst.exists():
            os.link(OLD / "data" / "fields" / f"{fid}.npz", dst)
    shutil.copy(OLD / "data" / "labels_dev.parquet", h.DATA / "labels_dev.parquet")

    meta = []
    for r in src.itertuples():
        z = np.load(f"{r.dir}/raw/{r.wid}.npz")
        o = np.load(f"{r.dir}/s2/{r.wid}.npz")["poly"]
        q = h._QA[r.case]
        u = q[q.field_id == r.wid].set_index("date").usable
        dates = z["dates"]
        usable = np.array([bool(u.get(str(d), False)) for d in dates])
        np.savez_compressed(h.DATA / "fields" / f"{r.field_id}.npz", dates=dates, refl=z["refl"], clear=z["clear"],
                            usable=usable, outline=o, year=np.int32(r.year))
        meta.append(dict(field_id=r.field_id, year=r.year, split="lockbox", H=o.shape[0], W=o.shape[1],
                         T=len(dates), n_outline=int(o.sum())))
    src.drop(columns=["dir"]).to_parquet(sp_fp)
    pd.concat([dev, pd.DataFrame(meta)]).sort_values("field_id").to_parquet(h.DATA / "fields.parquet")
    print(f"package {h.C}: dev {len(dev)}, lockbox {len(src)}")
    print(pd.crosstab(src.year, src.group).reindex(columns=h.GROUPS + ["paid_unknown"], fill_value=0).to_string())


if __name__ == "__main__":
    build()
