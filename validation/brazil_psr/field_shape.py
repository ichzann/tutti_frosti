"""Automatic shape rules for the field_handcheck outlines (no hand verdicts, no damage data used).

    python -m validation.brazil_psr.field_shape        # -> OUT/shape_metrics.parquet, prints the rule funnel
    python -m validation.brazil_psr.field_shape site   # -> OUT/shape_site (view page, every outline + its measures)

Per outline (matched labels, 10 m grid):
  parts      separate pieces (8-connected) after dropping crumbs < SPECK_HA (segmentation leftovers, 0.01-0.5 ha);
             rule: exactly 1, i.e. no second patch >= SPECK_HA anywhere in the window; metrics below on the cleaned outline
  hole_frac  share of the filled outline that is holes (tree islands, yards); rule <= HOLE_MAX
  solidity   area / convex-hull area of the filled outline; low = notches, arms, L-shapes; rule >= SOLIDITY_MIN
  corners    vertices after straightening the pixel staircase (Douglas-Peucker, SIMPLIFY_M); info only (was <= CORNERS_MAX)
  wiggle     straightened perimeter / convex-hull perimeter; 1 = no in-and-out edges; info only (was <= WIGGLE_MAX)
  centrality insurance point depth inside the outline / deepest inside depth (1 = most interior spot, 0 = edge,
             < 0 = outside); loose rule >= CENTRAL_MIN (point not on the outer rim)
  area_rel   |outline / insured - 1|; rule <= AREA_TOL
Thresholds fixed from the user's description ("simple forms, few curves, single part, area really close";
triangles / trapezoids / round fields allowed, no symmetry or right angles required) before any result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import Affine
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label
from shapely.geometry import shape

from validation.brazil_psr import field_handcheck as hc

SPECK_HA = 0.5
SIMPLIFY_M = 20
SOLIDITY_MIN = 0.80                     # 0.85 before user review (2026-11-06: simple fields rejected)
CORNERS_MAX = 12
WIGGLE_MAX = 1.15
HOLE_MAX = 0.05
AREA_TOL = 0.15
CENTRAL_MIN = 0.25
RULES = ["single_part", "no_holes", "solid", "central", "area"]          # corners + wiggle measured, not used (user, 2026-11-06:
                                                              # simple rounded fields failed the corner count)


def outline(wid, year, point=False):
    hc.fs.set_year(int(year), hc.OUT)
    m = np.load(hc.fs._mdir() / f"{wid}.npz")
    o = np.isin(m["lab"], m["matched"])
    return (o, tuple(int(v) for v in m["point"])) if point else o


def clean(o):
    lab, _ = label(o, structure=np.ones((3, 3)))
    n = np.bincount(lab.ravel())
    n[0] = 0
    return np.isin(lab, np.nonzero(n * 0.01 >= SPECK_HA)[0])


def metrics(o, pt):
    o = clean(o)
    _, parts = label(o, structure=np.ones((3, 3)))
    filled = binary_fill_holes(o)
    hole = 1 - o.sum() / filled.sum()
    lab, _ = label(filled, structure=np.ones((3, 3)))
    big = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1          # largest piece (the only one if parts == 1)
    polys = [shape(g) for g, v in features.shapes(big.astype("uint8"), mask=big, connectivity=8,
                                                     transform=Affine(10, 0, 0, 0, -10, 0)) if v == 1]
    p = max(polys, key=lambda q: q.area)
    s = p.simplify(SIMPLIFY_M, preserve_topology=True)
    hull = p.convex_hull
    inside = distance_transform_edt(filled)
    depth = inside[pt] if filled[pt] else -distance_transform_edt(~filled)[pt]
    return dict(centrality=float(depth / inside.max()), parts=int(parts), clean_ha=float(o.sum() * 0.01), hole_frac=float(hole), solidity=float(p.area / hull.area),
                corners=int(len(s.exterior.coords) - 1), wiggle=float(s.exterior.length / hull.exterior.length))


def table():
    r = pd.read_parquet(hc.OUT / "results.parquet")
    r = r[r.n_matched.fillna(0) > 0]
    rows = []
    for x in r.itertuples():
        o, pt = outline(x.wid, x.year, point=True)
        if clean(o).any():
            rows.append(dict(wid=x.wid, **metrics(o, pt)))
    m = r[["wid", "year", "group", "loss", "batch", "NR_AREA_TOTAL", "matched_ha", "matched_ratio", "outcome"]].merge(
        pd.DataFrame(rows), on="wid")
    m["area_rel"] = (m.clean_ha / m.NR_AREA_TOTAL - 1).abs()
    m["single_part"] = m.parts == 1
    m["no_holes"] = m.hole_frac <= HOLE_MAX
    m["solid"] = m.solidity >= SOLIDITY_MIN
    m["few_corners"] = m.corners <= CORNERS_MAX
    m["smooth"] = m.wiggle <= WIGGLE_MAX
    m["area"] = m.area_rel <= AREA_TOL
    m["central"] = m.centrality >= CENTRAL_MIN
    m["passes"] = m[RULES].all(axis=1)
    m.to_parquet(hc.OUT / "shape_metrics.parquet")
    return m


def funnel(m):
    G = ["heavy", "medium", "mild", "none"]
    keep = pd.Series(True, index=m.index)
    print(f"{'outlines':28s} {len(m):4d}  " + str(m.group.value_counts().reindex(G).fillna(0).astype(int).tolist()))
    for r in RULES:
        keep &= m[r]
        print(f"+ {r:26s} {keep.sum():4d}  " + str(m[keep].group.value_counts().reindex(G).fillna(0).astype(int).tolist())
              + f"   (rule alone fails {(~m[r]).sum()})")


def figure(r, fp):
    """field_handcheck.figure with the cleaned outline (the one the rules measure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = hc.fs
    fs.set_year(int(r.year), hc.OUT)
    dates, clear, x = fs._load(r.wid)
    m = np.load(fs._mdir() / f"{r.wid}.npz")
    o = clean(np.isin(m["lab"], m["matched"]))
    sz = np.load(hc.OUT / "snaps" / f"{r.wid}.npz")
    pre = fs.composite(dates, clear, x, *hc.SHOW.get(int(r.year), tuple(f"{int(r.year)}-{d}" for d in hc.SHOW_DEFAULT)))
    panels = [(hc._rgb(pre), "in-season")]
    for i, d in enumerate(sz["dates"]):
        if d:
            panels.append((hc._stretch(sz["rgb"][i][::-1].transpose(1, 2, 0)), f"off-season {d}"))
    e = np.arange(hc.N_PX) * 10 + 5
    fig, ax = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.5))
    ax = np.atleast_1d(ax)
    for a, (img, t) in zip(ax, panels):
        a.imshow(img, extent=(0, hc.WIN_M, hc.WIN_M, 0))
        a.contour(e, e, o.astype(float), [0.5], colors="yellow", linewidths=1.6)
        a.plot(m["point"][1] * 10 + 5, m["point"][0] * 10 + 5, "rx", ms=9, mew=2.2)
        a.set_title(t, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    ax[0].plot([100, 600], [1900, 1900], color="white", lw=3)
    ax[0].text(350, 1850, "500 m", color="white", ha="center", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(fp, dpi=72, pil_kwargs=dict(quality=72))
    plt.close(fig)


def _fig(args):
    r, fp = args
    r = pd.Series(r)
    if not fp.exists():
        figure(r, fp)


def site(sel=False):
    """OUT/shape_site/index.html (field_shape_page.html + rows) and img/<wid>.jpg for every measured outline;
    sel=True: OUT/selection_site with only the field_full selection."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    m = pd.read_parquet(hc.OUT / "shape_metrics.parquet").sort_values("wid")
    if sel:
        m = m[m.wid.isin(pd.read_parquet(hc.OUT / "selection.parquet").wid)]
    d = hc.OUT / ("selection_site" if sel else "shape_site")
    (d / "img").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(_fig, [(r._asdict(), d / "img" / f"{r.wid}.jpg") for r in m.itertuples()], chunksize=4))
    rows = [dict(id=r.wid, ins=round(r.NR_AREA_TOTAL, 1), out=round(r.clean_ha, 1), area=round(r.area_rel, 3),
                 parts=int(r.parts), holes=round(r.hole_frac, 3), sol=round(r.solidity, 3), corners=int(r.corners),
                 wiggle=round(r.wiggle, 3), central=round(r.centrality, 2), fail=[k for k in RULES if not getattr(r, k)]) for r in m.itertuples()]
    lim = dict(area=AREA_TOL, holes=HOLE_MAX, sol=SOLIDITY_MIN, corners=CORNERS_MAX, wiggle=WIGGLE_MAX, speck=SPECK_HA, central=CENTRAL_MIN)
    tpl = (Path(__file__).parent / "field_shape_page.html").read_text()
    (d / "index.html").write_text(tpl.replace("__ROWS__", json.dumps(rows)).replace("__LIM__", json.dumps(lim)))
    print(len(rows), "fields ->", d)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] in (["site"], ["site", "sel"]):
        site(sel=sys.argv[2:] == ["sel"])
    else:
        m = table()
        funnel(m)
        print(m[["parts", "hole_frac", "solidity", "corners", "wiggle", "area_rel"]].describe().round(3).to_string())
