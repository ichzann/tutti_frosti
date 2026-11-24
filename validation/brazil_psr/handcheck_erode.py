"""Edge test: frost_recovery damaged share on hand-check outlines shrunk inwards by 0-50 m.

    python -m validation.brazil_psr.handcheck_erode          # -> OUT/methods_erode/frost_recovery_<case>.parquet

Same scoring as handcheck_run (cached calibration, same event / placebo nights); per night the share of valid
pixels with score >= threshold is computed inside the outline eroded by k pixels (10 m each).
Columns share_e<k>, n_valid_e<k> for k in ERODE_PX.
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

from validation.brazil_psr import handcheck_run as hr

NAME = "frost_recovery"
ERODE_PX = [0, 1, 2, 3, 5]
OUT = hr.hs.hc.OUT / "methods_erode"


def _one(fid):
    m, store = hr._W["m"], hr._W["store"]
    poly = store._mask[fid]
    inner = {k: binary_erosion(poly, iterations=k) if k else poly for k in ERODE_PX}
    rows = []
    for night, kind in hr._nights(fid):
        base = dict(field_id=fid, method=NAME, night=night, kind=kind, n_mask=int(poly.sum()))
        try:
            r = m.damage(fid, dt.date.fromisoformat(night))
        except Exception as ex:                              # noqa: BLE001
            rows.append(dict(base, error=repr(ex)[:300]))
            continue
        v, sc, meta = r["valid"], r["score"], r.get("meta", {})
        dmg = v & (sc >= m.score_threshold)
        sh = meta.get("share_damaged")
        base["share_meta"] = float(sh) if sh is not None and np.isfinite(sh) else np.nan
        for k, mk in inner.items():
            vv = v & mk
            base[f"n_valid_e{k}"] = int(vv.sum())
            base[f"share_e{k}"] = float((dmg & mk).sum() / vv.sum()) if vv.any() else np.nan
        rows.append(base)
    store.field.cache_clear()
    store.bands.cache_clear()
    if hasattr(m, "clear_cache"):
        m.clear_cache()
    return rows


def run(workers=6):
    OUT.mkdir(parents=True, exist_ok=True)
    for case in hr.CASES:
        fp = OUT / f"{NAME}_{case}.parquet"
        if fp.exists():
            continue
        t0 = dt.datetime.now()
        try:
            hr._init(case, NAME)
            ids = hr._W["store"].field_ids()
            with ProcessPoolExecutor(workers, initializer=hr._init, initargs=(case, NAME)) as ex:
                rows = [r for rs in ex.map(_one, ids, chunksize=2) for r in rs]
        except Exception as ex:                              # noqa: BLE001
            print("FAILED", case, repr(ex)[:200], flush=True)
            continue
        pd.DataFrame(rows).assign(case=case).to_parquet(fp)
        print(case, len(ids), "fields", len(rows), "nights", f"{(dt.datetime.now() - t0).seconds // 60} min", flush=True)


if __name__ == "__main__":
    run()
