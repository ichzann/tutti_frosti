"""Example figure for the README: two 2021 wheat fields, true colour, season curve, c_r4b damage map.

    python -m scripts.example_figure            -> docs/example_field.png

c_r4b is fitted exactly as in the scorer's leave-one-year-out fold: on development fields of all other years.
Needs the packaged field data (competition/data, not in the public repo).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from competition.field import Field, dev_labels, field_table
from methods.c_r4b import method as M

FIELDS = [("C0033", "Heavy loss (insurance: 80 % lost)"), ("C0195", "No claim")]
YEAR = 2021


def evi2(f):
    nir, red = f.band("B08"), f.band("B04")
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1)


def main_part(o):
    """Largest connected piece of the outline (display only; the method scores the whole outline)."""
    lab, n = ndimage.label(o)
    if n <= 1:
        return o
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def main():
    t = field_table()
    dev = t[(t.split == "dev") & (t.year != YEAR)]
    lab = dev_labels()
    M.fit([Field(r.field_id, r.year) for r in dev.itertuples()], lab)

    fig, ax = plt.subplots(2, 3, figsize=(13, 8.2), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    for i, (fid, title) in enumerate(FIELDS):
        f = Field(fid, YEAR)
        o = f.outline
        e = evi2(f)
        use = [k for k in range(len(f.dates)) if f.usable[k] and np.isfinite(e[k][o]).mean() > 0.5]
        curve = np.array([np.nanmean(e[k][o]) for k in use])
        peak = use[int(np.argmax(curve))]

        rgb = f.refl[peak][[2, 1, 0]].transpose(1, 2, 0)
        rgb = np.clip(np.nan_to_num(rgb) / 0.15, 0, 1)
        ax[i, 0].imshow(rgb)
        ax[i, 0].contour(main_part(o), levels=[0.5], colors="yellow", linewidths=1.2)
        ax[i, 0].set_title(f"{title}\nSentinel-2 true colour, {str(f.dates[peak])}", fontsize=10)

        a = ax[i, 1]
        a.plot(f.dates[use], curve, "o-", color="#2a7d2e", ms=4)
        a.set_ylabel("EVI2 (field mean)")
        a.set_ylim(0, 0.95)
        a.set_title("Season curve the method compares with known fields", fontsize=10)
        a.tick_params(axis="x", labelrotation=30, labelsize=8)
        a.grid(alpha=0.3)

        m = M.predict(f)
        score = np.nanmean(m)                    # field score as the scorer computes it
        show = main_part(o)
        m = np.where(show, m, np.nan)
        im = ax[i, 2].imshow(np.ma.masked_invalid(m), cmap="YlOrRd", vmin=0.2, vmax=0.8)
        ax[i, 2].contour(show, levels=[0.5], colors="black", linewidths=0.8)
        ax[i, 2].set_title(f"Damage map (c_r4b), field score {score:.2f}", fontsize=10)
        fig.colorbar(im, ax=ax[i, 2], fraction=0.046, pad=0.04, label="estimated harvest loss")
        for a in (ax[i, 0], ax[i, 2]):
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Southern Brazil wheat, 2021 season (frost in late June / July)", fontsize=12)
    fig.tight_layout()
    fig.savefig("docs/example_field.png", dpi=110)


if __name__ == "__main__":
    main()
