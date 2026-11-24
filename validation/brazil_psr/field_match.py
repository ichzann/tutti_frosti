"""Fast test: find the insured area around a PSR property point by seeded region growing on Sentinel-2.

    python -m validation.brazil_psr.field_match download     # S2 2 km windows for the sample
    python -m validation.brazil_psr.field_match match        # growth, outcome table, figures

Sample: wheat 2019 policies, random per insured-area class (small < 20 ha, mid 20-60 ha, large 60-200 ha).
Seed  = winter-crop pixel nearest the point (within SEED_MAX_M); the crop mask (data.crop_mask) is used
        for the seed only.
Grow  = priority flood: repeatedly add the candidate pixel whose pre-frost EVI2 curve (before FROST_NIGHT,
        so frost damage cannot steer the shape) is closest (mean abs diff) to the region's mean curve.
        Candidates are pixels within GAP_PX of the region, so roads / tracks up to ~20 m are crossed.
Read  = the cost of each added pixel, in order.  Inside one sowing unit the cost stays low; when the unit
        is used up the next pixel comes from a different field and the cost steps up.  natural_ha = area
        grown before the cost first exceeds STEP x the median cost of the first CORE_PX pixels (and
        >= MIN_STEP).  matched region = first insured-area pixels.
Outcome: ok if natural_ha within [LO, HI] x insured area; too_small / too_big otherwise; no_seed; no_data.
"""
from __future__ import annotations

import datetime as dt
import heapq
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
from shapely.geometry import Point, box, shape

from data.common import ROOT
from validation.brazil_psr.data import CASES, CRS, _offset, _read, case_policies, crop_mask, search_items

CASE = "wheat2019"
OUT = ROOT / "cache" / "brazil_psr" / "fieldmatch_wheat2019"
WIN_M = 2000
N_PX = WIN_M // 10
SEASON = ("2019-03-01", "2019-09-30")
FROST_NIGHT = "2019-07-05"
CLASSES = {"small": (0, 20, 20), "mid": (20, 60, 20), "large": (60, 200, 10)}   # lo, hi ha, n sampled
BANDS4 = ["B02", "B03", "B04", "B08"]
SEED_MAX_M = 200
GAP_PX = 3
GRID_DAYS = 8
CORE_PX = 200                 # 2 ha
STEP, MIN_STEP = 2.5, 0.05
LO, HI = 0.75, 1.33
SEED = 7


# ----------------------------------------------------------------------------------- sample --
def sample():
    fp = OUT / "sample.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    p = case_policies(CASE)
    p = p[p.lat.notna() & p.lon.notna() & p.NR_AREA_TOTAL.notna()]
    rng = np.random.default_rng(SEED)
    parts = []
    for cls, (lo, hi, n) in CLASSES.items():
        q = p[(p.NR_AREA_TOTAL >= lo) & (p.NR_AREA_TOTAL < hi)]
        parts.append(q.loc[rng.choice(q.index, min(n, len(q)), replace=False)].assign(size_class=cls))
    s = pd.concat(parts).reset_index(drop=True)
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(s.lon, s.lat)], crs=4326).to_crs(CRS)
    s["x"], s["y"] = pts.x.round(0), pts.y.round(0)
    s["wid"] = ["W%03d" % i for i in range(len(s))]
    s["x0"] = np.floor((s.x - WIN_M / 2) / 10) * 10
    s["y1"] = np.ceil((s.y + WIN_M / 2) / 10) * 10
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_parquet(fp)
    return s


def _bounds(r):
    return (r.x0, r.y1 - WIN_M, r.x0 + WIN_M, r.y1)


# --------------------------------------------------------------------------------- download --
def _item(item, s, bands=BANDS4):
    item = pc.sign(item)
    foot = gpd.GeoSeries([shape(item.geometry)], crs=4326).to_crs(CRS).iloc[0]
    out = []
    for r in s.itertuples():
        b = _bounds(r)
        if not foot.contains(box(*b)):
            continue
        scl = _read(item.assets["SCL"].href, b, (N_PX, N_PX))
        clear = np.isin(scl, [4, 5])
        if clear.mean() < 0.05:
            continue
        off = _offset(item)
        st = np.stack([np.clip(_read(item.assets[k].href, b, (N_PX, N_PX)).astype("float32") + off, 0, 65535)
                       for k in bands]).astype("uint16")
        out.append((r.wid, clear, st))
    return item, out


def _retry(item, s, bands=BANDS4, tries=4):
    for k in range(tries):
        try:
            return _item(item, s, bands)
        except Exception as e:
            if k == tries - 1:
                print(f"  FAILED {item.id}: {e!r}", flush=True)
                return item, []
            time.sleep(5 * (k + 1))


def download(workers=16, out=OUT, season=SEASON, bands=BANDS4, max_cloud=None, s=None):
    """S2 windows for the sample into out/raw/<wid>.npz (dates, clear, refl (T, len(bands), H, W))."""
    s = sample() if s is None else s
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)
    g = gpd.GeoDataFrame(s, geometry=[box(*_bounds(r)) for r in s.itertuples()], crs=CRS)
    items = search_items(g, season)
    if max_cloud is not None:
        items = [i for i in items if i.properties.get("eo:cloud_cover", 100) < max_cloud]
    print(f"{len(s)} windows, {len(items)} items", flush=True)
    best = {}
    for p in tmp.glob("*.npz"):
        w, d, n = p.stem.split("__")
        best[(w, d)] = (int(n), p)
    with ThreadPoolExecutor(workers) as ex:
        for k, (item, res) in enumerate(ex.map(lambda it: _retry(it, s, bands), items)):
            d = item.datetime.date().isoformat()
            for w, c, st in res:
                n = int(c.sum())
                if (w, d) in best and best[(w, d)][0] >= n:
                    continue
                fp = tmp / f"{w}__{d}__{n}.npz"
                np.savez_compressed(fp, clear=c, refl=st)
                if (w, d) in best:
                    best[(w, d)][1].unlink(missing_ok=True)
                best[(w, d)] = (n, fp)
            if (k + 1) % 50 == 0:
                print(f"  items {k + 1}/{len(items)}", flush=True)
    by = {}
    for (w, d), (_, p) in best.items():
        by.setdefault(w, []).append((d, p))
    for w in s.wid:
        lst = sorted(by.get(w, []))
        clear = np.zeros((len(lst), N_PX, N_PX), bool)
        refl = np.zeros((len(lst), len(bands), N_PX, N_PX), "uint16")
        for t, (d, p) in enumerate(lst):
            z = np.load(p)
            clear[t], refl[t] = z["clear"], z["refl"]
        np.savez_compressed(raw / f"{w}.npz", dates=np.array([d for d, _ in lst]), clear=clear, refl=refl)
        for _, q in lst:                                       # pieces are in raw now (disk filled up, 2026-11-06)
            q.unlink(missing_ok=True)
    (out / "download_done.json").write_text(json.dumps(dict(items=len(items), bands=bands, season=season)))


# ------------------------------------------------------------------------------------ match --
def _load(w):
    z = np.load(OUT / "raw" / f"{w}.npz")
    x = z["refl"].astype("float32") / 10000.0
    ev = 2.5 * (x[:, 3] - x[:, 2]) / (x[:, 3] + 2.4 * x[:, 2] + 1.0)
    ev = np.where(z["clear"], ev, np.nan).astype("float32")
    return z["dates"], ev, x, z["clear"]


def pre_frost_curves(dates, ev):
    """(H, W, G) EVI2 linearly interpolated onto an 8-day grid, season start .. day before frost."""
    d = np.array([dt.date.fromisoformat(s).toordinal() for s in dates])
    t0, t1 = dt.date.fromisoformat(SEASON[0]).toordinal(), dt.date.fromisoformat(FROST_NIGHT).toordinal() - 1
    grid = np.arange(t0, t1 + 1, GRID_DAYS)
    sel = d <= t1
    d, e = d[sel], ev[sel]
    H, W = e.shape[1:]
    out = np.full((H, W, len(grid)), np.nan, "float32")
    flat = e.reshape(len(d), -1)
    for j in range(flat.shape[1]):
        ok = np.isfinite(flat[:, j])
        if ok.sum() < 4:
            continue
        out.reshape(-1, len(grid))[j] = np.interp(grid, d[ok], flat[ok, j])
    return out


def grow(curves, seed, limit_px):
    """Priority flood from seed; returns pixel order and per-pixel cost."""
    H, W, G = curves.shape
    valid = np.isfinite(curves).all(-1)
    inreg = np.zeros((H, W), bool)
    queued = np.zeros((H, W), bool)
    s = curves[seed].astype("float64").copy()
    n = 0
    heap, order, cost = [], [], []

    def push_nb(r, c):
        for dr in range(-GAP_PX, GAP_PX + 1):
            for dc in range(-GAP_PX, GAP_PX + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and valid[rr, cc] and not queued[rr, cc]:
                    queued[rr, cc] = True
                    heapq.heappush(heap, (float(np.abs(curves[rr, cc] - s / max(n, 1)).mean()), rr, cc))

    queued[seed] = True
    heapq.heappush(heap, (0.0, *seed))
    while heap and len(order) < limit_px:
        _, r, c = heapq.heappop(heap)
        k = float(np.abs(curves[r, c] - s / max(n, 1)).mean()) if n else 0.0
        if heap and k > heap[0][0] + 1e-6:          # stale priority: re-queue with the current cost
            heapq.heappush(heap, (k, r, c))
            continue
        inreg[r, c] = True
        s += curves[r, c]
        n += 1
        order.append((r, c))
        cost.append(k)
        push_nb(r, c)
    return np.array(order), np.array(cost)


def natural_px(cost):
    if len(cost) <= CORE_PX:
        return len(cost)
    thr = max(STEP * float(np.median(cost[1:CORE_PX])), MIN_STEP)
    sm = pd.Series(cost).rolling(25, min_periods=1).median().to_numpy()
    over = np.nonzero(sm[CORE_PX:] > thr)[0]
    return int(CORE_PX + over[0]) if len(over) else len(cost)


def match_one(r):
    dates, ev, x, clear = _load(r.wid)
    if len(dates) < 5:
        return dict(wid=r.wid, outcome="no_data")
    mask = crop_mask(dates, ev, CASES[CASE])
    cy, cx = int(round((r.y1 - r.y) / 10 - 0.5)), int(round((r.x - r.x0) / 10 - 0.5))
    curves = pre_frost_curves(dates, ev)
    ok = mask & np.isfinite(curves).all(-1)
    rr, cc = np.nonzero(ok)
    res = dict(wid=r.wid, n_dates=len(dates), n_pre=int((dates < FROST_NIGHT).sum()),
               point_in_crop=bool(ok[cy, cx]))
    if not len(rr):
        return dict(res, outcome="no_seed", seed_dist_m=np.nan)
    dist = np.hypot(rr - cy, cc - cx) * 10
    i = int(np.argmin(dist))
    res["seed_dist_m"] = float(dist[i])
    if dist[i] > SEED_MAX_M:
        return dict(res, outcome="no_seed")
    target = int(round(r.NR_AREA_TOTAL * 100))
    order, cost = grow(curves, (int(rr[i]), int(cc[i])), limit_px=int(target * 2.5) + CORE_PX)
    nat = natural_px(cost)
    ratio = nat / target
    out = ("ok" if LO <= ratio <= HI else "too_small" if ratio < LO else "too_big")
    if len(order) < target:
        out = "too_small"
    np.savez_compressed(OUT / "match" / f"{r.wid}.npz", order=order, cost=cost, seed=np.array([rr[i], cc[i]]),
                        point=np.array([cy, cx]), mask=mask)
    return dict(res, outcome=out, natural_ha=nat / 100, grown_ha=len(order) / 100, ratio=ratio,
                core_cost=float(np.median(cost[1:CORE_PX])) if len(cost) > 2 else np.nan,
                matched_ha=min(target, len(order)) / 100)


def match():
    s = sample()
    (OUT / "match").mkdir(exist_ok=True)
    rows = []
    for r in s.itertuples():
        m = match_one(r)
        rows.append(m)
        print(r.wid, r.size_class, r.NR_AREA_TOTAL, m["outcome"], round(m.get("ratio", np.nan), 2), flush=True)
    res = s.merge(pd.DataFrame(rows), on="wid")
    res.to_parquet(OUT / "results.parquet")
    t = pd.crosstab(res.size_class, res.outcome, margins=True)
    print(t)
    return res



# ---------------------------------------------------------------------------------- figures --
def _rgb(x, clear, t):
    im = np.clip(x[t][[2, 1, 0]].transpose(1, 2, 0) / 0.15, 0, 1)
    im[~clear[t]] = im[~clear[t]] * 0.35 + np.array([1, 0, 1]) * 0.25
    return im


def _best_date(dates, clear, lo, hi):
    sel = [i for i, d in enumerate(dates) if lo <= d <= hi]
    return max(sel, key=lambda i: clear[i].mean()) if sel else None


def figure(r, fp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dates, ev, x, clear = _load(r.wid)
    m = np.load(OUT / "match" / f"{r.wid}.npz")
    order, cost = m["order"], m["cost"]
    target = int(round(r.NR_AREA_TOTAL * 100))
    nat = natural_px(cost)
    reg = np.zeros((N_PX, N_PX), bool)
    reg[tuple(order[:target].T)] = True
    natm = np.zeros((N_PX, N_PX), bool)
    natm[tuple(order[:nat].T)] = True
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for a, (lo, hi, lab) in zip(ax[:2], [("2019-06-01", "2019-07-04", "pre-frost"),
                                          ("2019-07-20", "2019-09-15", "after frost / peak")]):
        t = _best_date(dates, clear, lo, hi)
        if t is None:
            a.set_title(f"{lab}: no image")
            continue
        a.imshow(_rgb(x, clear, t), extent=(0, WIN_M, WIN_M, 0))
        e = np.arange(N_PX) * 10 + 5
        a.contour(e, e, natm.astype(float), [0.5], colors="cyan", linewidths=1.2, linestyles="--")
        a.contour(e, e, reg.astype(float), [0.5], colors="yellow", linewidths=1.8)
        a.plot(m["point"][1] * 10 + 5, m["point"][0] * 10 + 5, "rx", ms=12, mew=3)
        a.set_title(f"{lab} {dates[t]} (clear {clear[t].mean():.0%})")
        a.set_xticks([0, 500, 1000, 1500, 2000]); a.set_yticks([0, 500, 1000, 1500, 2000])
    ha = np.arange(1, len(cost) + 1) / 100
    ax[2].plot(ha, pd.Series(cost).rolling(25, min_periods=1).median(), color="k")
    ax[2].axvline(r.NR_AREA_TOTAL, color="goldenrod", lw=2, label=f"insured {r.NR_AREA_TOTAL:.1f} ha")
    ax[2].axvline(nat / 100, color="c", ls="--", lw=2, label=f"similar area {nat / 100:.1f} ha")
    ax[2].set_xlabel("area grown (ha)"); ax[2].set_ylabel("difference to region curve (EVI2)")
    ax[2].set_title("growth cost"); ax[2].legend()
    fig.suptitle(f"{r.wid}  {r.NM_MUNICIPIO_PROPRIEDADE}-{r.SG_UF_PROPRIEDADE}  {r.lat:.5f}, {r.lon:.5f}  "
                 f"insured {r.NR_AREA_TOTAL:.1f} ha  outcome {r.outcome}  claim {r.claim_class}")
    fig.tight_layout()
    fig.savefig(fp, dpi=90)
    plt.close(fig)


def figures(n=3):
    res = pd.read_parquet(OUT / "results.parquet")
    rng = np.random.default_rng(SEED + 1)
    (OUT / "fig").mkdir(exist_ok=True)
    picked = []
    for cls in ["small", "mid"]:
        q = res[(res.size_class == cls) & res.outcome.isin(["ok", "too_small", "too_big"])]
        picked += list(rng.choice(q.wid, min(n, len(q)), replace=False))
    for w in picked:
        figure(res[res.wid == w].iloc[0], OUT / "fig" / f"{w}.png")
    print("figures:", picked)
    return picked


if __name__ == "__main__":
    {"download": download, "match": match, "figures": figures}[sys.argv[1]]()
