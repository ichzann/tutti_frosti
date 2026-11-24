"""Batch 3: wheat candidates from the full PSR download -> outlines -> automatic shape rules, in rounds until
QUOTA fields pass (user 2026-11-06: automatic selection, cap 200, reuse the batch 1+2 outlines first).

    python -m validation.brazil_psr.field_full check    # is the download complete? (size, last line, rows)
    python -m validation.brazil_psr.field_full run      # wait for the download, then rounds until QUOTA passes

The full CSV is read in place (SRC, the user's curl target) through a symlink RAW_FULL; only data.KEEP columns are
read (no names / CPF).  Candidates as field_handcheck._candidates (single policy at the point, 3-150 ha,
lon -54..-48), all years, minus every policy already sampled.  Per round and group, the number drawn =
remaining quota / observed pass rate of that group (prior YIELD_PRIOR), scaled to ROUND_MAX windows;
no-claim points within NONE_NEAR_KM of a frost-paid policy of the same year (field_handcheck._near_none).
Each round: S2 download, off-season snapshots, FTW, v3 match, field_shape rules.  Passing fields of batch 1-3
count towards QUOTA; selection.parquet = passing fields, at most QUOTA per group (random wid order).
"""
from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.common import ROOT
from validation.brazil_psr import field_handcheck as hc
from validation.brazil_psr import field_shape as fsh
from validation.brazil_psr.data import policies

SRC = Path.home() / "dados_abertos_psr.csv"
RAW_FULL = ROOT / "data" / "dados_abertos_psr_2016a2024_full.csv"
POL_FULL = ROOT / "cache" / "brazil_psr" / "policies_full.parquet"
YEARS = list(range(2016, 2025))
QUOTA = dict(heavy=110, medium=90, mild=100, none=100)           # 400 passing fields (user 2026-11-07; was 70/50/40/40)
ROUND_MAX = 300
MAX_ROUNDS = 16
CORE_YEARS = (2019, 2020, 2021)                                  # 158 of the first 189 fields
OTHER_YEAR_WEIGHT = 3.0                                          # draw weight of damaged candidates from other years
YIELD_PRIOR = 0.10                                               # 50 of 480 batch 1+2 outlines pass
G = ["heavy", "medium", "mild", "none"]


def complete(fp=SRC):
    """(ok, info): file ends with a newline and its last line has as many fields as the header."""
    with open(fp, "rb") as f:
        head = f.readline()
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(size - 65536, 0))
        tail = f.read()
    last = tail.rstrip(b"\n").split(b"\n")[-1]
    nl = tail.endswith(b"\n")
    nf, nh = last.count(b";") + 1, head.count(b";") + 1
    return nl and nf == nh, f"{size / 1e6:.0f} MB, ends with newline {nl}, last line fields {nf}/{nh}"


def _curl_running():
    return subprocess.run(["pgrep", "-f", "dados_abertos_psr"], capture_output=True).returncode == 0


def wait():
    while _curl_running():
        time.sleep(60)
    ok, info = complete()
    print("download finished:", info, flush=True)
    if not ok:
        sys.exit("download stopped before the end - resume it with the same curl -C - command")
    if not RAW_FULL.exists():
        RAW_FULL.symlink_to(SRC)


def candidates():
    pol = policies(RAW_FULL, POL_FULL)
    c = hc._candidates(YEARS, pol)
    print("full file policies", len(pol), "| wheat candidates", len(c), c.group.value_counts().reindex(G).tolist(),
          flush=True)
    return c


def passing():
    m = fsh.table()
    m = m[m.batch <= 3]                                          # batch 4 is a separate set (run_b4)
    return m, m[m.passes].group.value_counts().reindex(G).fillna(0).astype(int)


def draw(c, m, ok, rnd):
    rng = np.random.default_rng(hc.SEED + 10 + rnd)
    used = set(hc.sample().ID_PROPOSTA)
    pool = c[~c.ID_PROPOSTA.isin(used)]
    rate = (m[m.passes].group.value_counts() / m.group.value_counts()).reindex(G).fillna(YIELD_PRIOR).clip(lower=0.03)
    want = {g: max(QUOTA[g] - ok[g], 0) / rate[g] for g in G}
    avail = {g: int((pool.group == g).sum()) for g in G}
    n = {g: min(avail[g], math.ceil(want[g])) for g in G}
    k = min(1.0, ROUND_MAX / max(sum(n.values()), 1))
    n = {g: int(round(n[g] * k)) for g in G}
    print(f"round {rnd}: passing {ok.to_dict()}, available {avail}, drawing {n}", flush=True)
    take = []
    for g in ["heavy", "medium", "mild"]:
        q = pool[pool.group == g]
        w = np.where(q.year.isin(CORE_YEARS), 1.0, OTHER_YEAR_WEIGHT)
        take.append(q.sample(n[g], weights=w, random_state=rng.integers(1e9)))
    dmg = pd.concat(take)
    share = (dmg if len(dmg) else c[c.group != "none"]).year.value_counts(normalize=True)
    if n["none"]:
        take += hc._near_none(c, pool[pool.group == "none"], share, n["none"], rng)
    s = pd.concat(take).sample(frac=1, random_state=hc.SEED + 10 + rnd).reset_index(drop=True)
    if not len(s):
        return 0
    s = hc._windows(s, int(hc.sample().wid.str[1:].astype(int).max()) + 1)
    s.to_parquet(hc.OUT / f"sample_b3_r{rnd}.parquet")
    print(pd.crosstab(s.year, s.group, margins=True), flush=True)
    return len(s)


def run():
    wait()
    c = candidates()
    rnd = len(list(hc.OUT.glob("sample_b3_r*.parquet")))
    res = hc.OUT / "results.parquet"
    if not res.exists() or not set(hc.sample().wid) <= set(pd.read_parquet(res).wid):
        rnd -= 1                                                 # last drawn round not finished: finish it first
    m, ok = passing()
    while (ok < pd.Series(QUOTA)).any() and rnd < MAX_ROUNDS:
        if not (hc.OUT / f"sample_b3_r{rnd}.parquet").exists() and not draw(c, m, ok, rnd):
            print("no candidates left", flush=True)
            break
        for step in (hc.download, hc.snapshots, hc.ftw, hc.match):
            print("--", step.__name__, flush=True)
            step()
        m, ok = passing()
        fsh.funnel(m)
        rnd += 1
    sel = pd.concat([m[m.passes & (m.group == g)].sort_values("wid").head(QUOTA[g]) for g in G])
    sel.to_parquet(hc.OUT / "selection.parquet")
    print("selection", len(sel), sel.group.value_counts().reindex(G).tolist(), flush=True)


# ------------------------------------------------------------------------------- batch 4 --
# User 2026-11-09: new lockbox from unused candidates, same strict rules, damage first (every remaining heavy and
# medium candidate, mild and none to B4_MILD / B4_NONE), ~5 h; raw + FTW input of failing windows deleted per round.
B4_MILD, B4_NONE = 485, 484
B4_ROUND = 300


def draw_b4(c):
    """All batch-4 windows, drawn once -> b4_draw.parquet (written to sample_b4_r<k>.parquet one round at a time)."""
    fp = hc.OUT / "b4_draw.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    rng = np.random.default_rng(hc.SEED + 400)
    used = set(hc.sample().ID_PROPOSTA)
    pool = c[~c.ID_PROPOSTA.isin(used)]
    q = pool[pool.group == "mild"]
    w = np.where(q.year.isin(CORE_YEARS), 1.0, OTHER_YEAR_WEIGHT)
    take = [pool[pool.group.isin(["heavy", "medium"])],
            q.loc[rng.choice(q.index, min(B4_MILD, len(q)), replace=False, p=w / w.sum())]]
    share = pd.concat(take).year.value_counts(normalize=True)
    take += hc._near_none(c, pool[pool.group == "none"], share, B4_NONE, rng)
    s = pd.concat(take).sample(frac=1, random_state=hc.SEED + 400).reset_index(drop=True)
    s = hc._windows(s, int(hc.sample().wid.str[1:].astype(int).max()) + 1)
    s.to_parquet(fp)
    print(pd.crosstab(s.year, s.group, margins=True), flush=True)
    return s


def _prune(m):
    """Delete raw + FTW input of matched batch-4 windows that fail the rules (no outline or rules not all met)."""
    res = pd.read_parquet(hc.OUT / "results.parquet")
    b4 = set(res[res.batch == 4].wid)
    ok = set(m[m.passes].wid)
    n = 0
    for w in sorted(b4 - ok):
        for fp in (hc.OUT / "raw" / f"{w}.npz", hc.OUT / "ftw" / f"{w}_in.tif"):
            if fp.exists():
                fp.unlink()
                n += fp.suffix == ".npz"
    print("pruned raw of", n, "failing batch-4 windows", flush=True)


def _report(m):
    m4 = m[m.batch == 4]
    fsh.funnel(m4)
    m4[m4.passes].to_parquet(hc.OUT / "selection_b4.parquet")
    print("selection_b4", int(m4.passes.sum()), m4[m4.passes].group.value_counts().reindex(G).fillna(0).astype(int).tolist(),
          flush=True)


def run_b4():
    """Rounds of B4_ROUND windows; resumable (steps skip finished windows, round files are kept)."""
    s = draw_b4(candidates())
    for k in range(math.ceil(len(s) / B4_ROUND)):
        fp = hc.OUT / f"sample_b4_r{k:02d}.parquet"
        if not fp.exists():
            s.iloc[k * B4_ROUND:(k + 1) * B4_ROUND].to_parquet(fp)
        print(f"== batch 4 round {k}", time.strftime("%H:%M"), flush=True)
        for step in (hc.download, hc.snapshots, hc.ftw, hc.match):
            print("--", step.__name__, flush=True)
            step()
        m = fsh.table()
        _prune(m)
        _report(m)


if __name__ == "__main__":
    {"check": lambda: print(complete()), "run": run, "run_b4": run_b4}[sys.argv[1]]()
