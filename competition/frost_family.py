"""frost_recovery family -> competition score files (LOCKBOX2.md), harness only.

    python -m competition.frost_family dev                              # full stores -> competition/scores (dev 275)
    python -m competition.frost_family lockbox1                         # full stores -> old lockbox (first run)
    COMP_DIR=competition_b4 python -m competition.frost_family lockbox  # b4 stores -> second lockbox

Both methods train themselves per year store without labels.  Field numbers:
  frost_recovery_v2: share_all (damaged share over the whole season; season_run v2)
  frost_recovery:    largest damaged share over the field's ERA5 event nights (handcheck_run, kind event)
A field without a value is NaN; the scorer counts it in n_nan and scores it 0 (FRAME rule).
Lockbox runs are entered in the package's lockbox_runs.json ledger and refused a second time.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from competition import score as sc
from competition.harness import HARN
from data.common import ROOT

HC = ROOT / "cache" / "brazil_psr" / "handcheck_wheat"
NAMES = ("frost_recovery_v2", "frost_recovery")


def _values(name, SET):
    d = HC / f"methods_{SET}"
    if name == "frost_recovery_v2":
        s = pd.read_parquet(d / "season_frost_recovery_v2.parquet")
        s = s[s.get("reason", pd.Series("", index=s.index)).fillna("") == ""]
        return s.set_index("field_id").share_all.astype(float)
    df = pd.concat([pd.read_parquet(p) for p in d.glob("frost_recovery_wheat*.parquet")], ignore_index=True)
    df = df[(df.kind == "event") & (df.n_valid.fillna(0) > 0)]
    return df.groupby("field_id").share.max().astype(float)


def _rows(v, ids):
    """ids: DataFrame wid, field_id."""
    s = ids.wid.map(v)
    return pd.DataFrame(dict(field_id=ids.field_id, score=s.to_numpy(), dmg_share=np.nan,
                             n_valid=np.where(s.notna(), 1, 0), n_outline=1, error=""))


def run(which):
    old = pd.read_parquet(ROOT / "competition" / "_harness" / "split.parquet")
    if which == "dev":
        SET, ids, split = "full", old[old.split == "dev"], "dev"
    elif which == "lockbox1":
        SET, ids, split = "full", old[old.split == "lockbox"], "lockbox"
    else:
        if HARN == ROOT / "competition" / "_harness":
            sys.exit("set COMP_DIR=competition_b4 for the second lockbox")
        SET, ids, split = "b4", pd.read_parquet(HARN / "split.parquet"), "lockbox"
    led = HARN / "lockbox_runs.json"
    for name in NAMES:
        if split == "lockbox":
            done = json.loads(led.read_text()) if led.exists() else {}
            if name in done:
                print(f"{name} already scored on this lockbox ({done[name]}); skipped")
                continue
        v = _values(name, SET)
        print(f"{name} {which}: values for {int(ids.wid.isin(v.index).sum())} of {len(ids)} fields")
        sc._save(name, split, _rows(v, ids))
        if split == "lockbox":
            done[name] = time.strftime("%Y-%m-%d %H:%M")
            led.write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    run(sys.argv[1])
