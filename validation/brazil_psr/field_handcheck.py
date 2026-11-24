"""Hand-check set: wheat 2019-21 PSR policies, v3 field outlines whose area matches the insured area.

    python -m validation.brazil_psr.field_handcheck sample      # candidates by loss group, field ids F###
    python -m validation.brazil_psr.field_handcheck download    # S2 Mar-Nov (B02 B03 B04 B08 B11) per year
    python -m validation.brazil_psr.field_handcheck snapshots   # 3 off-season RGB scenes per window
    python -m validation.brazil_psr.field_handcheck ftw         # FTW inputs -> ftw env (ftw_infer.py)
    python -m validation.brazil_psr.field_handcheck match       # v3 segments + growth (field_segments, frozen)
    python -m validation.brazil_psr.field_handcheck pages       # blind figures of the area-matched fields

Groups (relative loss = (1 - coverage) + coverage x indemnity / insured value; user plan 2026-11-05):
  none    no claim of any kind (loss below the deductible)
  mild    frost paid, loss < 0.50;  medium 0.50-0.70;  heavy >= 0.70
Candidates: one wheat policy at the point that year (shared points have an ambiguous insured area),
3-150 ha, lon -54..-48 (one UTM grid).  heavy + medium: all; mild: N_MILD over years in proportion;
none: N_NONE spread over years like the damaged candidates, drawn from no-claim points within NONE_NEAR_KM
of a frost-paid policy of the same year (same frost exposure region).
Field ids are assigned in random order so the id says nothing about group or year.
Area match: matched outline within +-MATCH_TOL of the insured area (field_segments v3, nothing tuned here).
Figures are blind: outline + point on a pre-frost composite and the 3 off-season snapshots, no group,
loss, year or vegetation curves.
"""
from __future__ import annotations

import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from data.common import ROOT
from validation.brazil_psr import field_segments as fs
from validation.brazil_psr.data import CRS, policies
from validation.brazil_psr.field_match import N_PX, WIN_M, download as s2_download

OUT = ROOT / "cache" / "brazil_psr" / "handcheck_wheat"
YEARS = [2019, 2020, 2021]
AREA = (3, 150)
N_MILD, N_NONE = 70, 70
B2_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2024]          # batch 2 (user 2026-11-05: ~100 more kept fields)
B2_MILD, B2_NONE = 95, 110
NONE_NEAR_KM = 15
MATCH_TOL = 0.35
SHOW = {2019: ("2019-05-01", "2019-06-25"), 2020: ("2020-06-01", "2020-07-31"), 2021: ("2021-05-01", "2021-06-25")}
SHOW_DEFAULT = ("05-01", "06-25")
FTW_PY = "/opt/homebrew/Caskroom/miniforge/base/envs/ftw/bin/python"
SEED = 11
fs.V = "v3"


# ----------------------------------------------------------------------------------- sample --
def sample():
    """Batch 1 (sample.parquet), batch 2 (sample_b2.parquet) and batch 3 rounds (field_full, sample_b3_r<n>.parquet)."""
    parts = [_sample_b1().assign(batch=1)]
    if (OUT / "sample_b2.parquet").exists():
        parts.append(pd.read_parquet(OUT / "sample_b2.parquet").assign(batch=2))
    for fp in sorted(OUT.glob("sample_b3_r*.parquet")):
        parts.append(pd.read_parquet(fp).assign(batch=3))
    for fp in sorted(OUT.glob("sample_b4_r*.parquet")):                 # field_full batch 4 (new lockbox, 2026-11-09)
        parts.append(pd.read_parquet(fp).assign(batch=4))
    return pd.concat(parts, ignore_index=True)


def _candidates(years, pol=None):
    p = policies() if pol is None else pol
    p = p[(p.crop == "Trigo") & p.ANO_APOLICE.isin([str(y) for y in years]) & p.lat.notna()].copy()
    p["year"] = p.ANO_APOLICE.astype(int)
    p["loss"] = (1 - p.NivelDeCobertura) + p.NivelDeCobertura * p.loss_ratio
    paid = p.claim_class == "frost_paid"
    p["group"] = np.select([p.claim_class == "none", paid & (p.loss < 0.5), paid & (p.loss < 0.7), paid],
                           ["none", "mild", "medium", "heavy"], "drop")
    key = p.year.astype(str) + p.lat.round(5).astype(str) + p.lon.round(5).astype(str)
    p = p[key.map(key.value_counts()) == 1]
    p = p[(p.group != "drop") & p.NR_AREA_TOTAL.between(*AREA) & p.lon.between(-54, -48)]
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(p.lon, p.lat)], crs=4326, index=p.index).to_crs(CRS)
    p["x"], p["y"] = pts.x.round(0), pts.y.round(0)
    return p


def _near_none(p, none, share, n_total, rng):
    """No-claim points within NONE_NEAR_KM of a frost-paid policy of the same year, n per year by share."""
    take = []
    for y in sorted(share.index):
        ref = p[(p.year == y) & (p.group != "none")][["x", "y"]].to_numpy()
        q = none[none.year == y]
        if not len(ref) or not len(q):
            continue
        d = np.hypot(q.x.to_numpy()[:, None] - ref[:, 0], q.y.to_numpy()[:, None] - ref[:, 1]).min(1)
        q = q[d <= NONE_NEAR_KM * 1000]
        n = min(len(q), int(round(n_total * share.get(y, 0))))
        take.append(q.loc[rng.choice(q.index, n, replace=False)])
    return take


def _windows(s, first_id):
    s["wid"] = ["F%03d" % (first_id + i) for i in range(len(s))]
    s["x0"] = np.floor((s.x - WIN_M / 2) / 10) * 10
    s["y1"] = np.ceil((s.y + WIN_M / 2) / 10) * 10
    return s


def _sample_b1():
    fp = OUT / "sample.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    rng = np.random.default_rng(SEED)
    p = _candidates(YEARS)
    hm = p[p.group.isin(["heavy", "medium"])]
    take = [hm]
    mild = p[p.group == "mild"]
    avail = mild.year.value_counts()                           # year mix like heavy + medium, so group and
    n = {y: min(avail.get(y, 0), int(round(N_MILD * c / len(hm)))) for y, c in hm.year.value_counts().items()}
    n[avail.idxmax()] += N_MILD - sum(n.values())              # year are not confounded; rest from the big year
    for y, q in mild.groupby("year"):
        take.append(q.loc[rng.choice(q.index, min(len(q), n.get(y, 0)), replace=False)])
    dmg = pd.concat(take)
    none = p[p.group == "none"]
    take += _near_none(p, none, dmg.year.value_counts() / len(dmg), N_NONE, rng)
    s = _windows(pd.concat(take).sample(frac=1, random_state=SEED).reset_index(drop=True), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_parquet(fp)
    print(pd.crosstab(s.year, s.group, margins=True))
    return s


def batch2():
    """All unused heavy + medium of B2_YEARS, mild: all outside 2021 then 2021 up to B2_MILD, no-claim B2_NONE
    near frost-paid policies (year mix like this batch's damaged fields).  ids continue after batch 1."""
    fp = OUT / "sample_b2.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    rng = np.random.default_rng(SEED + 1)
    b1 = _sample_b1()
    p = _candidates(B2_YEARS)
    p = p[~p.ID_PROPOSTA.isin(b1.ID_PROPOSTA)]
    take = [p[p.group.isin(["heavy", "medium"])]]
    mild = p[p.group == "mild"]
    other = mild[mild.year != 2021]
    take.append(other)
    m21 = mild[mild.year == 2021]
    take.append(m21.loc[rng.choice(m21.index, min(len(m21), B2_MILD - len(other)), replace=False)])
    dmg = pd.concat(take)
    take += _near_none(_candidates(B2_YEARS), p[p.group == "none"], dmg.year.value_counts() / len(dmg), B2_NONE, rng)
    s = pd.concat(take).sample(frac=1, random_state=SEED + 1).reset_index(drop=True)
    s = _windows(s, int(b1.wid.str[1:].astype(int).max()) + 1)
    s.to_parquet(fp)
    print(pd.crosstab(s.year, s.group, margins=True))
    return s


def _years(s):
    for y in sorted(s.year.unique()):
        fs.set_year(y, OUT)
        yield y, s[s.year == y]


# ---------------------------------------------------------------------------------- steps --
def download():
    res = OUT / "results.parquet"
    done = set(pd.read_parquet(res).wid) if res.exists() else set()   # matched windows: raw may be deleted (user 2026-11-09)
    for y, s in _years(sample()):
        todo = s[[w not in done and not (OUT / "raw" / f"{w}.npz").exists() for w in s.wid]]
        print(y, len(todo), "windows to download", flush=True)
        if len(todo):
            s2_download(out=OUT, season=fs.SEASON, bands=fs.BANDS, s=todo)


def snapshots():
    for y, s in _years(sample()):
        fs.snapshots(s)


def ftw():
    d = OUT / "ftw"
    d.mkdir(parents=True, exist_ok=True)
    todo = []
    for y, s in _years(sample()):
        for r in s.itertuples():
            if (d / f"{r.wid}_prob.tif").exists() or not (OUT / "raw" / f"{r.wid}.npz").exists():
                continue                                       # done, or raw pruned (failed window, 2026-11-09)
            dates, clear, x = fs._load(r.wid)
            if not len(dates):                                 # no scenes for this window
                continue
            a, b = fs.composite(dates, clear, x, *fs.WIN_A), fs.composite(dates, clear, x, *fs.WIN_B)
            order = [2, 1, 0, 3]                               # B04 B03 B02 B08, as ftw-tools stacks
            st = np.nan_to_num(np.concatenate([a[order], b[order]]) * 10000, nan=0).clip(0, 65535).astype("uint16")
            inp = d / f"{r.wid}_in.tif"
            with rasterio.open(inp, "w", driver="GTiff", width=N_PX, height=N_PX, count=8, dtype="uint16",
                               crs=CRS, transform=from_origin(r.x0, r.y1, 10, 10)) as dst:
                dst.write(st)
            todo.append(str(inp))
    for i in range(0, len(todo), 40):
        subprocess.run([FTW_PY, str(ROOT / "validation" / "brazil_psr" / "ftw_infer.py"), *todo[i:i + 40]],
                       check=True)


def match():
    fp = OUT / "results.parquet"
    done = pd.read_parquet(fp) if fp.exists() else None
    rows = [] if done is None else done[[c for c in done.columns if c not in sample().columns or c == "wid"]
                                       ].drop(columns="area_ok", errors="ignore").to_dict("records")
    have = set() if done is None else set(done.wid)
    for y, s in _years(sample()):
        for r in s.itertuples():
            if r.wid in have:
                continue
            try:
                m = fs.match_one(r) if len(fs._load(r.wid)[0]) else dict(wid=r.wid, outcome="no_scenes")
            except Exception as ex:                            # noqa: BLE001  (e.g. no valid pixels in a window)
                m = dict(wid=r.wid, outcome="error")
                print(r.wid, "match error", repr(ex)[:120], flush=True)
            rows.append(m)
            print(r.wid, y, r.NR_AREA_TOTAL, m["outcome"], round(m.get("matched_ratio", np.nan), 2), flush=True)
    res = sample().merge(pd.DataFrame(rows), on="wid")
    res["area_ok"] = (res.matched_ratio - 1).abs() <= MATCH_TOL
    res.to_parquet(OUT / "results.parquet")
    print(pd.crosstab(res.group, res.area_ok, margins=True))
    print(pd.crosstab(res.year, res.area_ok, margins=True))
    return res


# -------------------------------------------------------------------------------- figures --
def _stretch(rgb):
    """(H, W, 3) reflectance -> 0..1 with one 2-98 % stretch over all three bands (keeps the colour balance)."""
    lo, hi = np.nanpercentile(rgb, [2, 98]) if np.isfinite(rgb).any() else (0, 1)
    return np.clip(np.nan_to_num((rgb - lo) / max(hi - lo, 1e-6)), 0, 1)


def _rgb(c):
    return _stretch(c[[2, 1, 0]].transpose(1, 2, 0))


def figure(r, fp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.set_year(int(r.year), OUT)
    dates, clear, x = fs._load(r.wid)
    m = np.load(fs._mdir() / f"{r.wid}.npz")
    lab = m["lab"]
    sz = np.load(OUT / "snaps" / f"{r.wid}.npz")
    pre = fs.composite(dates, clear, x, *SHOW.get(int(r.year), tuple(f"{int(r.year)}-{d}" for d in SHOW_DEFAULT)))
    panels = [(_rgb(pre), "in-season, before frost")]
    for i, d in enumerate(sz["dates"]):
        if d:
            panels.append((_stretch(sz["rgb"][i][::-1].transpose(1, 2, 0)), f"off-season {d}"))
    e = np.arange(N_PX) * 10 + 5
    fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    ax = np.atleast_1d(ax)
    for a, (img, t) in zip(ax, panels):
        a.imshow(img, extent=(0, WIN_M, WIN_M, 0))
        a.contour(e, e, np.isin(lab, m["matched"]).astype(float), [0.5], colors="yellow", linewidths=1.8)
        a.plot(m["point"][1] * 10 + 5, m["point"][0] * 10 + 5, "rx", ms=10, mew=2.5)
        a.set_title(t, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    ax[0].plot([100, 600], [1900, 1900], color="white", lw=3)
    ax[0].text(350, 1850, "500 m", color="white", ha="center", fontsize=9)
    fig.suptitle(f"{r.wid}    insured {r.NR_AREA_TOTAL:.1f} ha    outline {r.matched_ha:.1f} ha", fontsize=12)
    fig.tight_layout()
    fig.savefig(fp, dpi=80, pil_kwargs=dict(quality=80))
    plt.close(fig)


def pages():
    res = pd.read_parquet(OUT / "results.parquet")
    ok = res[res.area_ok & (res.batch == res.batch.max())].sort_values("wid")   # only the newest batch to check
    d = OUT / "pages"
    d.mkdir(exist_ok=True)
    for r in ok.itertuples():
        figure(r, d / f"{r.wid}.jpg")
    ok[["wid"]].to_csv(OUT / "handcheck_ids.csv", index=False)
    print(len(ok), "figures in", d)
    site(ok)


def site(ok):
    """OUT/site/index.html (handcheck_page.html + field list) and img/ copies, published as an Artifact."""
    import json
    import shutil
    d = OUT / "site"
    (d / "img").mkdir(parents=True, exist_ok=True)
    for w in ok.wid:
        shutil.copy(OUT / "pages" / f"{w}.jpg", d / "img" / f"{w}.jpg")
    rows = [dict(id=r.wid, ins=round(float(r.NR_AREA_TOTAL), 1), out=round(float(r.matched_ha), 1)) for r in ok.itertuples()]
    tpl = (ROOT / "validation" / "brazil_psr" / "handcheck_page.html").read_text()
    (d / "index.html").write_text(tpl.replace("__FIELDS__", json.dumps(rows)))


if __name__ == "__main__":
    {"sample": sample, "download": download, "snapshots": snapshots, "ftw": ftw, "match": match,
     "pages": pages, "batch2": batch2}[sys.argv[1]]()
