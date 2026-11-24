"""Exploration harness for the field-matching rules (wheat 2019, the 50 field_match windows).

    python -m validation.brazil_psr.field_explore

Fields, season curves, phenology and snapshot colours are computed once per window (v3 segmentation,
cached in OUT/prep).  Each rule variant is then run from the true insurance point and from PLACEBO
random winter-crop points (>= 400 m from the true point, same insured area).  An area match from the
true point only counts as evidence of location as far as it beats the placebo rate.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import ndimage

from validation.brazil_psr import field_segments as fs

fs.V = "v3"
OUT = fs.OUT / "explore"
N_PX = fs.N_PX
PLACEBO_PER_WINDOW = 3
TOL = 0.35


def prep(w):
    fp = OUT / "prep" / f"{w}.pkl"
    if fp.exists():
        return pd.read_pickle(fp)
    d, c, x = fs._load(w)
    lab, edge = fs.segments(w, d, c, x)
    cur = fs.seg_curves(lab, d, c, x)
    ph = {k: fs.phenology(v) for k, v in cur.items()}
    win = [k for k, p in ph.items() if p["winter"]]
    sn = fs._snaps(w)
    snap = {k: [np.nanmedian(sn[i][:, lab == k], 1) if np.isfinite(sn[i][0][lab == k]).mean() >= 0.5 else None
                for i in range(len(fs.SNAPS))] for k in win}
    area = {k: float((lab == k).sum()) / 100 for k in cur}
    adj = {}
    for k in win:
        g = ndimage.binary_dilation(lab == k, iterations=fs.ADJ_PX)
        adj[k] = set(np.unique(lab[g])) - {0, k}
    gap = {}                                               # edge-to-edge distance (m) between winter fields
    for k in win:
        dt_k = ndimage.distance_transform_edt(lab != k) * 10
        gap[k] = {q: float(dt_k[lab == q].min()) for q in win if q != k}
    p = dict(lab=lab, cur=cur, ph=ph, win=win, snap=snap, area=area, adj=adj, gap=gap,
             pw=fs.pixel_winter(d, c, x))
    fp.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(p, fp)
    return p


def _dist_to(lab, k, cy, cx):
    yy, xx = np.nonzero(lab == k)
    return float(np.hypot(yy - cy, xx - cx).min() * 10)


def similar(p, k, s, prm):
    """(ok, score) for joining field k to start s."""
    a, b = p["ph"][k], p["ph"][s]
    du, dd = abs(a["greenup"] - b["greenup"]), abs(a["greendown"] - b["greendown"])
    me = float(np.abs(p["cur"][k]["evi2"] - p["cur"][s]["evi2"]).mean())
    mn = float(np.abs(p["cur"][k]["ndmi"] - p["cur"][s]["ndmi"]).mean())
    ok = du <= prm["days"] and dd <= prm["days"] and me <= prm["curve"] and mn <= prm["curve"]
    sc = du / prm["days"] + dd / prm["days"] + me / prm["curve"] + mn / prm["curve"]
    if prm["snap"] is not None:
        ds = [float(np.abs(u - v).mean()) for u, v in zip(p["snap"][k], p["snap"][s]) if u is not None and v is not None]
        if ds:
            ok = ok and max(ds) <= prm["snap"]
            sc += max(ds) / prm["snap"]
    return ok, sc


def run_point(p, cy, cx, ins, prm):
    lab, win = p["lab"], p["win"]
    near = [(k, _dist_to(lab, k, cy, cx)) for k in win]
    near = [(k, d) for k, d in near if d <= prm["start_m"]]
    if not near:
        return dict(outcome="no_start", ratio=np.nan, start_ha=0.0)
    starts = [min(near, key=lambda t: t[1])[0]] if prm["start"] == "nearest" else [k for k, _ in near]
    best = None
    for s in starts:
        group, order = {s}, [s]
        while True:
            if prm["reach_m"] > 0:                    # non-adjacent: any winter field within reach of the group
                front = [k for k in win if k not in group and min(p["gap"][g][k] for g in group) <= prm["reach_m"]]
            else:
                front = sorted(set().union(*(p["adj"][g] for g in group)) & set(win) - group)
            cand = [(sc, k) for k in front for ok, sc in [similar(p, k, s, prm)] if ok]
            if not cand:
                break
            _, k = min(cand)
            group.add(k)
            order.append(k)
        cum = np.cumsum([p["area"][k] for k in order])
        j = int(np.argmin(np.abs(cum - ins)))
        r = dict(ratio=float(cum[j] / ins), start_ha=p["area"][s], n=j + 1, group_ha=float(cum[-1]),
                 fields=list(order[:j + 1]), joined=list(order))
        if best is None or abs(r["ratio"] - 1) < abs(best["ratio"] - 1):
            best = r
    best["outcome"] = "match" if abs(best["ratio"] - 1) <= TOL else "miss"
    return best


def placebo_points():
    fp = OUT / "placebo_points.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    rng = np.random.default_rng(11)
    rows = []
    for r in fs.sample().itertuples():
        p = prep(r.wid)
        cy, cx = int(round((r.y1 - r.y) / 10 - 0.5)), int(round((r.x - r.x0) / 10 - 0.5))
        yy, xx = np.nonzero(p["pw"])
        ok = (np.hypot(yy - cy, xx - cx) * 10 >= 400) & (yy > 30) & (yy < 170) & (xx > 30) & (xx < 170)
        for i in rng.choice(np.nonzero(ok)[0], min(PLACEBO_PER_WINDOW, ok.sum()), replace=False):
            rows.append((r.wid, int(yy[i]), int(xx[i])))
    d = pd.DataFrame(rows, columns=["wid", "cy", "cx"])
    d.to_parquet(fp)
    return d


def evaluate(prm, label=""):
    s = fs.sample()
    pl = placebo_points()
    rows = []
    for r in s.itertuples():
        p = prep(r.wid)
        cy, cx = int(round((r.y1 - r.y) / 10 - 0.5)), int(round((r.x - r.x0) / 10 - 0.5))
        rows.append(dict(wid=r.wid, size_class=r.size_class, kind="true", **run_point(p, cy, cx, r.NR_AREA_TOTAL, prm)))
        for q in pl[pl.wid == r.wid].itertuples():
            rows.append(dict(wid=r.wid, size_class=r.size_class, kind="placebo",
                             **run_point(p, q.cy, q.cx, r.NR_AREA_TOTAL, prm)))
    d = pd.DataFrame(rows)
    m = d.assign(hit=d.outcome == "match").groupby("kind").hit.mean()
    by = d[d.kind == "true"].assign(hit=lambda t: t.outcome == "match").groupby("size_class").hit.sum().to_dict()
    print(f"{label:38s} true {m['true']:.2f}  placebo {m['placebo']:.2f}  gap {m['true'] - m['placebo']:+.2f}  "
          f"true hits by size {by}", flush=True)
    return d


BASE = dict(start="nearest", start_m=200, reach_m=0, days=12, curve=0.08, snap=0.03)
VARIANTS = {
    "v3 as is": {},
    "v2 (no snapshot rule)": dict(snap=None),
    "start: best of fields <= 150 m": dict(start="best", start_m=150),
    "timing +-20 d": dict(days=20),
    "timing +-30 d": dict(days=30),
    "non-adjacent within 300 m": dict(reach_m=300),
    "non-adjacent 300 m, timing +-30 d": dict(reach_m=300, days=30),
    "snapshot 0.02": dict(snap=0.02),
    "snapshot 0.045": dict(snap=0.045),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    for n in names:
        evaluate(dict(BASE, **VARIANTS[n]), n)
