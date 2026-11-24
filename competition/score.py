"""Fixed scoring script (FRAME.md section 6).

    python -m competition.score dev <method>             # leave-one-year-out on development fields
    python -m competition.score lockbox <method>         # harness only: fit on all dev, score the lockbox ONCE
    python -m competition.score compare <a> <b> [split]  # paired bootstrap of the M1 difference a - b

<method> = a package methods/<name>/ (method.py with predict(field) and optional fit(fields, labels)), or a
baseline in competition/baselines/ (ndmi_drop, low_peak_evi2), or era5_coldest (weather reference, harness only).
Dev: per held-out year, the module is reloaded (fresh state), fit on the other years' fields and labels, then
predicts the held-out year.  Field score = mean of the map over outline pixels with a value (scorer, not method);
a field with no value anywhere scores 0 (no damage found) and is counted in n_nan.  Paid fields with unknown
loss (group paid_unknown) enter only (paid, none) pairs and are left out of the group metrics M3 / M6.
Output: competition/scores/<method>__<split>.parquet (one row per field) and .json (metrics).
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import sys
import time
import traceback

import numpy as np
import pandas as pd

from competition.field import Field, dev_labels, field_table
from competition.harness import C, GROUPS, HARN

SCORES = C / "scores"
BOOT = 2000
BOOT_SEED = 7
DMG = 0.35                                    # damaged pixel / "none above threshold" level (claims start at 0.35)
WORKERS = 8
_M = {}


# ------------------------------------------------------------------------------------- running --
def _module(name):
    for mod in (f"methods.{name}.method", f"competition.baselines.{name}"):
        try:
            importlib.import_module(mod.rsplit(".", 1)[0])
        except ModuleNotFoundError:
            continue
        m = importlib.import_module(mod)
        return importlib.reload(m)
    raise SystemExit(f"no method {name}")


def _one(f: Field):
    t0 = time.time()
    try:
        y = np.asarray(_M["m"].predict(f), dtype="float32")
        o = f.outline
        if y.shape != o.shape:
            raise ValueError(f"map shape {y.shape} != outline {o.shape}")
        v = y[o]
        if not _M["raw"] and (np.nanmin(v, initial=0) < 0 or np.nanmax(v, initial=0) > 1):
            raise ValueError(f"values outside 0..1: {np.nanmin(v)} .. {np.nanmax(v)}")
        ok = np.isfinite(v)
        r = dict(field_id=f.field_id, score=float(v[ok].mean()) if ok.any() else np.nan,
                 dmg_share=float((v[ok] >= DMG).mean()) if ok.any() else np.nan, n_valid=int(ok.sum()),
                 n_outline=int(o.sum()), error="")
    except Exception:                                            # noqa: BLE001 - record and continue
        r = dict(field_id=f.field_id, score=np.nan, dmg_share=np.nan, n_valid=0, n_outline=0,
                 error=traceback.format_exc(limit=3)[-400:])
    f.drop()
    r["seconds"] = time.time() - t0
    return r


def _predict(m, fields, raw):
    _M.update(m=m, raw=raw)
    with mp.get_context("fork").Pool(WORKERS) as p:             # fork: workers share the fitted state
        return list(p.imap_unordered(_one, fields, chunksize=1))


def run(name, split):
    if name == "era5_coldest":
        return _era5(split)
    t = field_table()
    lab = dev_labels()
    raw = name in ("ndmi_drop", "low_peak_evi2")                 # baselines: field number, not 0..1
    rows = []
    t0 = time.time()
    if split == "dev":
        dev = t[t.split == "dev"]
        for y in sorted(dev.year.unique()):
            m = _module(name)
            tr = dev[dev.year != y]
            if hasattr(m, "fit"):
                m.fit([Field(r.field_id, r.year) for r in tr.itertuples()],
                      lab[lab.field_id.isin(tr.field_id)].reset_index(drop=True))
            rows += [dict(r, fold=int(y)) for r in
                     _predict(m, [Field(r.field_id, r.year) for r in dev[dev.year == y].itertuples()], raw)]
            print(f"  fold {y}: {int((dev.year == y).sum())} fields, {time.time() - t0:.0f} s", flush=True)
    else:
        led = HARN / "lockbox_runs.json"
        done = json.loads(led.read_text()) if led.exists() else {}
        if name in done:
            sys.exit(f"{name} already scored on the lockbox ({done[name]}); one run only (FRAME rule 4)")
        m = _module(name)
        dev = t[t.split == "dev"]
        if hasattr(m, "fit"):
            m.fit([Field(r.field_id, r.year) for r in dev.itertuples()], lab)
        rows = _predict(m, [Field(r.field_id, r.year) for r in t[t.split == "lockbox"].itertuples()], raw)
        done[name] = time.strftime("%Y-%m-%d %H:%M")
        led.write_text(json.dumps(done, indent=1))
    s = pd.DataFrame(rows)
    print(f"{name} {split}: {len(s)} fields in {time.time() - t0:.0f} s, errors {int((s.error != '').sum())}")
    return _save(name, split, s)


def _era5(split):
    h = pd.read_parquet(HARN / "split.parquet")
    h = h[h.split == split]
    s = pd.DataFrame(dict(field_id=h.field_id, score=-h.era5_tmin, dmg_share=np.nan, n_valid=1, n_outline=1,
                          error=""))
    return _save("era5_coldest", split, s)


def _labels(split):
    if split == "dev":
        return dev_labels()
    h = pd.read_parquet(HARN / "split.parquet")
    h = h[h.split == "lockbox"]
    claim = h.group != "none"
    return pd.DataFrame(dict(field_id=h.field_id, year=h.year, group=h.group, claim=claim,
                             loss=np.where(claim, h.loss, np.nan)))


def _save(name, split, s):
    SCORES.mkdir(parents=True, exist_ok=True)
    d = _labels(split).merge(s, on="field_id")
    d["n_nan"] = d.score.isna()
    d["score"] = d.score.fillna(0.0)
    d.to_parquet(SCORES / f"{name}__{split}.parquet")
    met = metrics(d)
    met.update(method=name, split=split, n_errors=int((d.error != "").sum()), n_nan=int(d.n_nan.sum()))
    (SCORES / f"{name}__{split}.json").write_text(json.dumps(met, indent=1))
    print(report(met))
    return met


def load(name, split):
    return pd.read_parquet(SCORES / f"{name}__{split}.parquet")


# ------------------------------------------------------------------------------------- metrics --
def _pairs(d):
    """A[i, j] = 1 if field i truly lost more than j (comparable pair); K = concordance credit of the scores."""
    claim, loss, s = d.claim.to_numpy(bool), d.loss.to_numpy(float), d.score.to_numpy()
    known = np.isfinite(loss)                                    # paid_unknown: only in (paid, none) pairs
    pp = (claim & known)[:, None] & (claim & known)[None, :] & (np.nan_to_num(loss)[:, None] > np.nan_to_num(loss)[None, :])
    pn = claim[:, None] & ~claim[None, :]
    A = (pp | pn).astype("float64")
    K = A * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))
    return A, K, pn.astype("float64")


def _boot_w(n, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(BOOT)]).astype("float64")


def _c(A, K, W=None):
    if W is None:
        return float(K.sum() / A.sum()) if A.sum() else np.nan
    num = ((W @ K) * W).sum(1)
    den = ((W @ A) * W).sum(1)
    return num / np.where(den > 0, den, np.nan)


def _ci(x):
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [np.nan, np.nan]


def _jt(d):
    """Jonckheere-Terpstra over none < mild < medium < heavy: statistic, z, one-sided p."""
    from scipy.stats import norm
    g = [d.score[d.group == k].to_numpy() for k in GROUPS]
    J = sum(((b[None, :] > a[:, None]) + 0.5 * (b[None, :] == a[:, None])).sum()
            for i, a in enumerate(g) for b in g[i + 1:])
    n = np.array([len(x) for x in g])
    N = n.sum()
    mu = (N ** 2 - (n ** 2).sum()) / 4
    var = (N ** 2 * (2 * N + 3) - (n ** 2 * (2 * n + 3)).sum()) / 72
    z = (J - mu) / np.sqrt(var) if var > 0 else np.nan
    return float(J), float(z), float(norm.sf(z))


def metrics(d):
    d = d.reset_index(drop=True)
    A, K, PN = _pairs(d)
    same = (d.year.to_numpy()[:, None] == d.year.to_numpy()[None, :]).astype("float64")
    W = _boot_w(len(d))
    paid = d[d.claim]
    none = d[~d.claim]
    jt = _jt(d)
    m = dict(
        n=len(d), n_paid=int(d.claim.sum()), n_none=int((~d.claim).sum()),
        n_paid_unknown=int((d.group == "paid_unknown").sum()),
        M1_C=_c(A, K), M1_ci=_ci(_c(A, K, W)),
        M2_C_within_year=_c(A * same, K * same), M2_ci=_ci(_c(A * same, K * same, W)),
        M3_median={g: float(d.score[d.group == g].median()) for g in GROUPS},
        M3_jt_z=jt[1], M3_jt_p=jt[2],
        M4_AUC=_c(PN, K * PN), M4_ci=_ci(_c(PN, K * PN, W)),
        M5_mae_paid=float((paid.score - paid.loss).abs().median()) if paid.loss.notna().any() else np.nan,
        M5_none_ge_035=float((none.score >= DMG).mean()) if len(none) else np.nan,
        M6_dmg_share={g: float(d.dmg_share[d.group == g].median()) for g in GROUPS},
    )
    per = {}
    for y, g in d.groupby("year"):
        a, k, _ = _pairs(g.reset_index(drop=True))
        per[int(y)] = dict(n=len(g), n_paid=int(g.claim.sum()), C=_c(a, k))
    m["per_year_C"] = per
    return m


def compare(a, b, split="dev"):
    """Paired bootstrap over fields: M1(a) - M1(b), share of resamples with a > b."""
    da, db = load(a, split), load(b, split)
    da = da.set_index("field_id")
    db = db.set_index("field_id").loc[da.index]
    A, Ka, _ = _pairs(da.reset_index())
    _, Kb, _ = _pairs(db.reset_index())
    W = _boot_w(len(da))
    diff = _c(A, Ka, W) - _c(A, Kb, W)
    r = dict(a=a, b=b, split=split, M1_a=_c(A, Ka), M1_b=_c(A, Kb), diff=_c(A, Ka) - _c(A, Kb),
             diff_ci=_ci(diff), p_a_gt_b=float(np.nanmean(diff > 0)))
    print(json.dumps(r, indent=1))
    return r


def report(m):
    f = lambda x: f"{x:.3f}"                                   # noqa: E731
    ci = lambda k: f"[{f(m[k][0])}, {f(m[k][1])}]"              # noqa: E731
    lines = [f"{m.get('method', '')} {m.get('split', '')}: n {m['n']} (paid {m['n_paid']}, none {m['n_none']}), "
             f"errors {m.get('n_errors', 0)}, no-value fields {m.get('n_nan', 0)}",
             f"  M1 C {f(m['M1_C'])} {ci('M1_ci')} | M2 within-year C {f(m['M2_C_within_year'])} {ci('M2_ci')} "
             f"| M4 AUC {f(m['M4_AUC'])} {ci('M4_ci')}",
             "  M3 median " + " ".join(f"{g} {f(v)}" for g, v in m["M3_median"].items())
             + f" | JT z {m['M3_jt_z']:.2f} p {m['M3_jt_p']:.4f}",
             f"  M5 MAE paid {f(m['M5_mae_paid'])} | none >= 0.35 {f(m['M5_none_ge_035'])} | M6 dmg share "
             + " ".join(f"{g} {f(v)}" for g, v in m["M6_dmg_share"].items()),
             "  per year C " + " ".join(f"{y}:{f(v['C'])}(n{v['n']})" for y, v in m["per_year_C"].items())]
    return "\n".join(lines)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "compare":
        compare(*sys.argv[2:])
    else:
        run(sys.argv[2], cmd)
