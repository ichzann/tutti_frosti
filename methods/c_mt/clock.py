"""Module A: satellite frost clock (label-free).  Built from DEV-split fields only (user decision 2026-11-09).

    python -m methods.c_mt.clock            # build cache + report + plot (once)

Cache: methods/c_mt/cache/clock_<hash>.pkl with H per field, per-year N, M, q, omega, omega_hat, per-field own
counts, omega_clim, delta, field table T2, pixel table Tpix.
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import sys

import numpy as np

from methods.c_mt import common as C

CFG = C.CFG
DAYS = np.arange(CFG["day_range"][0], CFG["day_range"][1] + 1)


def cache_path(e_sat=None):
    tag = "" if e_sat is None or e_sat == CFG["e_sat"] else f"_esat{e_sat}"
    return C.CACHE / f"clock_{C.cfg_hash()}{tag}.pkl"


def _extract(f):
    doy, n, r, v, e = C.field_series(f)
    H = C.heading(doy, r)
    fi = C.intervals(doy, n[:, None], r[:, None], v[:, None], e[:, None])
    P = int(f.outline.sum())
    rng = np.random.default_rng([CFG["pix_seed"], int(f.field_id[1:])])
    pix = np.sort(rng.choice(P, size=min(CFG["pix_per_field"], P), replace=False))
    pdoy, pn, pr, pv, pe = C.pixel_series(f, pix)
    pi = C.intervals(pdoy, pn, pr, pv, pe)
    f.drop()
    return dict(field_id=f.field_id, year=f.year, H=H, n_dates=len(doy), fi=fi, pi=pi)


def _cover(a, b):
    """(n, len(DAYS)) bool: day d in (a, b]."""
    return (DAYS[None, :] > a[:, None]) & (DAYS[None, :] <= b[:, None])


def omega_year(N, M, e_sat):
    """N, M (days,) -> q, omega (NaN where undefined)."""
    with np.errstate(all="ignore"):
        q = np.where(N >= CFG["n_min"], M / np.maximum(N, 1), np.nan)
    if not np.isfinite(q).any():
        return q, np.full_like(q, np.nan)
    e = np.maximum(0.0, q - np.nanmedian(q))
    h = CFG["smooth_days"] // 2
    es = np.full_like(e, np.nan)
    for i in np.flatnonzero(np.isfinite(e)):
        es[i] = np.nanmean(e[max(0, i - h): i + h + 1])
    return q, np.minimum(1.0, es / e_sat)


def shrink(N, om, clim):
    Nn = np.where(np.isfinite(om), N, 0.0)
    w = np.where(np.isfinite(om), om, 0.0)
    return (Nn * w + CFG["n0"] * clim) / (Nn + CFG["n0"])


def clock(ownN, ownM, years, e_sat):
    yrs = sorted(set(years))
    Y = {}
    for y in yrs:
        k = years == y
        N, M = ownN[k].sum(0), ownM[k].sum(0)
        q, om = omega_year(N, M, e_sat)
        Y[y] = dict(N=N, M=M, q=q, omega=om, n_fields=int(k.sum()))
    stack = np.array([Y[y]["omega"] for y in yrs])
    with np.errstate(all="ignore"):
        clim = np.where(np.isfinite(stack).any(0), np.nanmean(stack, 0), 0.0)
    for y in yrs:
        Y[y]["omega_hat"] = shrink(Y[y]["N"], Y[y]["omega"], clim)
    return Y, clim


def core_day(w):
    if not np.isfinite(w).any() or np.nanmax(w) <= 0:
        return np.nan
    m = w >= CFG["delta_core_frac"] * np.nanmax(w)
    return float((w[m] * DAYS[m]).sum() / w[m].sum())


def deltas(Y, clim):
    dref = core_day(clim)
    out = {}
    for y, r in Y.items():
        mx = float(np.nanmax(r["omega_hat"]))
        r["max_omega_hat"] = mx
        r["d_star"] = core_day(r["omega_hat"])
        if mx < CFG["delta_min_omega"] or not np.isfinite(dref):
            out[y] = 0
        else:
            out[y] = int(np.clip(round(r["d_star"] - dref), -CFG["delta_clip"], CFG["delta_clip"]))
    return out, dref


def lso_omega_hat(cache, field_id):
    """Leave-self-out omega_hat for a clock field; full-year omega_hat (dev clock) for any other field.
    Year with no clock fields: omega_clim (fallback)."""
    k = cache["index"].get(field_id)
    if k is None:
        return None
    y = cache["years"][k]
    r = cache["Y"][y]
    N = r["N"] - cache["ownN"][k]
    M = r["M"] - cache["ownM"][k]
    _, om = omega_year(N, M, cache["e_sat"])
    return shrink(N, om, cache["clim"])


def year_omega_hat(cache, year):
    r = cache["Y"].get(int(year))
    return r["omega_hat"] if r is not None else cache["clim"]


def field_omega_hat(cache, field_id, year):
    w = lso_omega_hat(cache, field_id)
    return w if w is not None else year_omega_hat(cache, year)


def _counts(F, T, e_sat_unused=None):
    """Per field own N / M over DAYS with table T; returns ownN, ownM and per-field z / shock rows."""
    n = len(F)
    ownN = np.zeros((n, len(DAYS)))
    ownM = np.zeros((n, len(DAYS)))
    rows = []
    for k, d in enumerate(F):
        fi = d["fi"]
        if len(fi["a"]) == 0:
            rows.append(None)
            continue
        s = (fi["a"] + fi["b"]) / 2 - d["H"]
        D = np.stack([fi["dN"], fi["dR"], fi["dV"]], 1)
        Dc = np.stack([fi["dNc"], fi["dRc"], fi["dVc"]], 1)
        z = C.zscore(T, s, C.dt_class(fi["dt"]), D, Dc)
        shock = fi["canopy"] & (z["z"] <= CFG["shock_z"]) & (z["pi"] == 1)
        cov = _cover(fi["a"][fi["canopy"]], fi["b"][fi["canopy"]])
        ownN[k] = cov.any(0)
        ownM[k] = _cover(fi["a"][shock], fi["b"][shock]).any(0)
        rows.append(dict(z=z["z"], pi=z["pi"], shock=shock))
    return ownN, ownM, rows


def _table_rows(F, key, mask_fn=None):
    s, dtc, D = [], [], []
    for d in F:
        r = d[key]
        if len(r["a"]) == 0:
            continue
        m = r["canopy"].copy()
        if mask_fn is not None:
            m &= mask_fn(d, r)
        s.append(((r["a"] + r["b"]) / 2 - d["H"])[m]); dtc.append(C.dt_class(r["dt"][m]))
        D.append(np.stack([r["dN"], r["dR"], r["dV"]], 1)[m])
    return np.concatenate(s), np.concatenate(dtc), np.concatenate(D)


def build(e_sat=None):
    e_sat = CFG["e_sat"] if e_sat is None else e_sat
    fields = C.clock_fields()
    with mp.get_context("fork").Pool(CFG["workers"]) as p:
        F = p.map(_extract, fields, chunksize=2)
    years = np.array([d["year"] for d in F])
    # pass 1
    T1 = C.build_table(*_table_rows(F, "fi"))
    n1, m1, _ = _counts(F, T1)
    Y1, clim1 = clock(n1, m1, years, e_sat)

    def nonevent(d, r):
        w = Y1[d["year"]]["omega_hat"]
        cov = _cover(r["a"], r["b"])
        mx = np.where(cov, w[None, :], 0).max(1) if cov.shape[0] else np.zeros(0)
        return mx < CFG["event_excl"]
    # pass 2
    T2 = C.build_table(*_table_rows(F, "fi", nonevent))
    ownN, ownM, rows = _counts(F, T2)
    Y, clim = clock(ownN, ownM, years, e_sat)
    delta, dref = deltas(Y, clim)

    def nonevent2(d, r):
        w = Y[d["year"]]["omega_hat"]
        cov = _cover(r["a"], r["b"])
        mx = np.where(cov, w[None, :], 0).max(1) if cov.shape[0] else np.zeros(0)
        return mx < CFG["event_excl"]
    Tpix = C.build_table(*_table_rows(F, "pi", nonevent2))
    cache = dict(e_sat=e_sat, days=DAYS, field_ids=[d["field_id"] for d in F], years=years,
                 index={d["field_id"]: k for k, d in enumerate(F)}, H={d["field_id"]: d["H"] for d in F},
                 n_dates={d["field_id"]: d["n_dates"] for d in F},
                 ownN=ownN, ownM=ownM, Y=Y, clim=clim, delta=delta, d_ref=dref, T1=T1, T2=T2, Tpix=Tpix,
                 Y_pass1={y: dict(max_omega_hat=float(np.nanmax(r["omega_hat"]))) for y, r in Y1.items()},
                 n_intervals=dict(field=int(sum(len(d["fi"]["a"]) for d in F)),
                                  pixel=int(sum(len(d["pi"]["a"]) for d in F))))
    with open(cache_path(e_sat), "wb") as fh:
        pickle.dump(cache, fh)
    return cache


def load(e_sat=None):
    p = C.CACHE / "clock_active.txt"
    name = p.read_text().strip() if p.exists() else cache_path(e_sat).name
    with open(C.CACHE / name, "rb") as fh:
        return pickle.load(fh)


def ranges(days):
    if len(days) == 0:
        return []
    out, st, pr = [], days[0], days[0]
    for d in days[1:]:
        if d != pr + 1:
            out.append((st, pr)); st = d
        pr = d
    out.append((st, pr))
    return out


def report(cache, year_dates=True):
    import datetime as dt
    lines = [f"Module A (clock: dev-split imagery only, user decision 2026-11-09), e_sat {cache['e_sat']}, "
             f"d_ref {cache['d_ref']:.1f}, intervals field {cache['n_intervals']['field']} pixel-sample "
             f"{cache['n_intervals']['pixel']}, H NaN {sum(np.isnan(v) for v in cache['H'].values())} / "
             f"{len(cache['H'])}",
             "year | n dev fields | days q defined | max N | max omega_hat (pass 1) | max omega_hat | d*_y | delta_y | "
             "dates omega_hat >= 0.5"]
    for y in sorted(cache["Y"]):
        r = cache["Y"][y]
        hi = DAYS[r["omega_hat"] >= 0.5]
        rg = []
        for a, b in ranges(list(hi)):
            da = dt.date(int(y), 1, 1) + dt.timedelta(days=int(a) - 1)
            db = dt.date(int(y), 1, 1) + dt.timedelta(days=int(b) - 1)
            rg.append(f"{da:%d %b}-{db:%d %b} (doy {a}-{b})")
        ds = f"{r['d_star']:.1f}" if np.isfinite(r["d_star"]) else "nan"
        lines.append(f"{y} | {r['n_fields']} | {int(np.isfinite(r['q']).sum())} | {int(r['N'].max())} | "
                     f"{cache['Y_pass1'][y]['max_omega_hat']:.3f} | {r['max_omega_hat']:.3f} | {ds} | "
                     f"{cache['delta'][y]} | {'; '.join(rg) if rg else '-'}")
    return "\n".join(lines)


def plot(cache, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ys = sorted(cache["Y"])
    fig, axs = plt.subplots(len(ys), 1, figsize=(10, 2.0 * len(ys)), sharex=True)
    for ax, y in zip(axs, ys):
        r = cache["Y"][y]
        ax.plot(DAYS, r["q"], color="0.5", lw=1, label="q_y (shock share)")
        ax.plot(DAYS, r["omega_hat"], color="tab:red", lw=1.5, label="omega_hat_y")
        ax.plot(DAYS, cache["clim"], color="tab:blue", lw=0.8, ls="--", label="omega_clim")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{y}\nn={r['n_fields']}")
        ax.text(0.01, 0.8, f"delta {cache['delta'][y]} d, max omega_hat {r['max_omega_hat']:.2f}",
                transform=ax.transAxes, fontsize=8)
    axs[0].legend(fontsize=7, loc="upper right")
    axs[-1].set_xlabel("day of year")
    fig.suptitle("c_mt Module A clock (dev-split imagery only)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    e = float(sys.argv[1]) if len(sys.argv) > 1 else None
    c = build(e)
    (C.CACHE / "clock_active.txt").write_text(cache_path(e).name)
    rep = report(c)
    print(rep)
    tag = "" if e is None else f"_esat{e}"
    (C.HERE / "report" / f"module_a{tag}.txt").write_text(rep + "\n")
    plot(c, C.HERE / "report" / f"module_a_clock{tag}.png")


def delta_table(axis, perm_seed=None):
    """year -> delta for Module B.  calendar: {} (no shift); frost: dev-clock delta_y; perm: delta_y permuted across
    the clock years with a fixed seed (falsification F6).  Years without a clock entry get 0 (climatology rule)."""
    if axis == "calendar":
        return {}
    d = load()["delta"]
    if perm_seed is None:
        return dict(d)
    ys = sorted(d)
    p = np.random.default_rng(perm_seed).permutation(len(ys))
    return {ys[i]: d[ys[p[i]]] for i in range(len(ys))}
