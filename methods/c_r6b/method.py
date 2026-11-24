"""c_r6b: spectral unmixing + elastic registration of the fraction season course.

1. Every clear pixel-date spectrum (10 bands) is unmixed by fully constrained least squares (non-negative, sum to one)
   into green vegetation (GV), non-photosynthetic vegetation (NPV, straw / dead tissue), soil and shade; the three
   material fractions are shade-normalised.  Endmembers are fixed spectra (settings.yaml, label-free).
2. Each pixel's fraction series (GV, NPV, soil) on a day-of-year grid is registered by dynamic time warping to one
   template: the mean fraction course of the training fields (label-free).  This splits the season into
   phase (when the pixel reached each template stage, in grid steps relative to the template) and amplitude (the
   pixel's fractions and ln band reflectances read at each template stage).
3. Pixel loss = clip(b + w . [phase, amplitude], 0, 1); w, b from ridge on the training fields' mean registered
   features (alpha by inner leave-one-year-out C).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import yaml

CFG = yaml.safe_load(Path(__file__).with_name("settings.yaml").read_text())
STATE = {}
CH = {"gv": [0], "npv": [1], "soil": [2], "ln_bands": list(range(3, 13))}


# ------------------------------------------------------------------------------------------ unmixing --
def _endmembers():
    return np.stack([np.array(CFG["endmember_gv"]), np.array(CFG["endmember_npv"]),
                     np.array(CFG["endmember_soil"]), np.zeros(10)])


def unmix(X):
    """X (N, 10) -> (N, 4) fractions GV, NPV, SOIL, SHADE; fully constrained LS by enumerating active sets."""
    E = _endmembers()
    k, N = len(E), len(X)
    best = np.full(N, np.inf)
    F = np.zeros((N, k))
    for r in range(1, k + 1):
        for S in itertools.combinations(range(k), r):
            S = list(S)
            M = E[S]
            if r == 1:
                f = np.ones((N, 1))
            else:
                A = (M[:-1] - M[-1]).T
                sol = np.linalg.lstsq(A, (X - M[-1]).T, rcond=None)[0]
                f = np.concatenate([sol.T, 1 - sol.sum(0)[:, None]], 1)
            feas = (f >= -1e-9).all(1)
            res = ((X - f @ M) ** 2).sum(1)
            upd = feas & (res < best - 1e-12)
            best[upd] = res[upd]
            Fs = np.zeros((N, k))
            Fs[:, S] = f
            F[upd] = Fs[upd]
    return F


# ------------------------------------------------------------------------------------------ curves --
def _grid():
    return np.arange(CFG["grid_start_doy"], CFG["grid_end_doy"], CFG["grid_step_days"], dtype="float64")


def pixel_curves(f, idx=None):
    """(P, G, 13) curves of outline pixels (all, or the subset idx): GV, NPV, SOIL, ln B02..B12; plus pixel mask."""
    o = f.outline
    r = f.refl[:, :, o]                                                  # (T, 10, P)
    clr = np.isfinite(r[:, 6]).mean(1)
    use = f.usable & (clr >= CFG["usable_share"])
    doy = ((f.dates - f.dates.astype("datetime64[Y]")).astype(int) + 1.0)[use]
    r = r[use].astype("float64")
    if idx is not None:
        r = r[:, :, idx]
    T, B, P = r.shape
    X = np.transpose(r, (0, 2, 1)).reshape(-1, B)
    ok = np.isfinite(X).all(1)
    Fr = np.full((len(X), 4), np.nan)
    if ok.any():
        Fr[ok] = unmix(X[ok])
    sun = 1 - Fr[:, 3]
    Fr = Fr[:, :3] / np.where(sun >= CFG["min_sunlit"], sun, np.nan)[:, None]
    V = np.concatenate([Fr, np.log(np.maximum(X, CFG["refl_floor"]))], 1).reshape(T, P, 13)
    g = _grid()
    out = np.full((P, len(g), 13), np.nan)
    for p in range(P):
        for c in range(13):
            m = np.isfinite(V[:, p, c])
            if m.sum() >= CFG["min_clear_dates"]:
                out[p, :, c] = np.interp(g, doy[m], V[m, p, c])
    return out


# ------------------------------------------------------------------------------------------ registration --
def register(Q, T, W, rc, ac):
    """Q (P, G, 13) scaled curves, T (G, len(rc)) template -> (P, G + G * len(ac)) phase + amplitude features."""
    Qr, Qa = Q[:, :, rc], Q[:, :, ac]
    P, G, _ = Qr.shape
    cost = ((Qr[:, :, None, :] - T[None, None, :, :]) ** 2).sum(-1)    # (P, Gq, Gt)
    R = np.full((P, G + 1, G + 1), np.inf)
    R[:, 0, 0] = 0
    for i in range(1, G + 1):
        for j in range(max(1, i - W), min(G, i + W) + 1):
            R[:, i, j] = cost[:, i - 1, j - 1] + np.minimum(np.minimum(R[:, i - 1, j], R[:, i, j - 1]),
                                                            R[:, i - 1, j - 1])
    ph = np.zeros((P, G))
    cnt = np.zeros((P, G))
    amp = np.zeros((P, G, Qa.shape[2]))
    i = np.full(P, G)
    j = np.full(P, G)
    ar = np.arange(P)
    act = np.ones(P, bool)
    while act.any():
        a = ar[act]
        ph[a, j[a] - 1] += i[a] - 1
        cnt[a, j[a] - 1] += 1
        amp[a, j[a] - 1] += Qa[a, i[a] - 1]
        mv = np.argmin(np.stack([R[a, i[a] - 1, j[a] - 1], R[a, i[a] - 1, j[a]], R[a, i[a], j[a] - 1]]), 0)
        i[a] -= np.where(mv == 2, 0, 1)
        j[a] -= np.where(mv == 1, 0, 1)
        act = (i > 0) & (j > 0)
    ph = ph / cnt - np.arange(G)[None]
    amp = amp / cnt[..., None]
    return np.concatenate([ph, amp.reshape(P, -1)], 1)


def _channels():
    rc = sum((CH[c] for c in CFG["register_on"]), [])
    ac = sum((CH[c] for c in CFG["amplitude_channels"]), [])
    return rc, ac


def features(curves):
    """(P, G, 13) raw curves -> (P, F) registered features with the fitted scale and template; NaN rows kept."""
    rc, ac = _channels()
    ok = np.isfinite(curves).all((1, 2))
    q = curves[ok] / STATE["sd"]
    out = np.full((len(curves), STATE["n_feat"]), np.nan)
    ch = CFG["pixel_chunk"]
    for s in range(0, len(q), ch):
        out[np.where(ok)[0][s:s + ch]] = register(q[s:s + ch], STATE["template"], CFG["dtw_band"], rc, ac)
    return out


# ------------------------------------------------------------------------------------------ fit --
def _train_curves(f):
    n = int(f.outline.sum())
    idx = np.random.default_rng(0).choice(n, min(CFG["train_pixels"], n), replace=False)
    c = pixel_curves(f, idx)
    f.drop()
    return c


def _train_feats(c):
    if not np.isfinite(c).all((1, 2)).any():
        return np.full(STATE["n_feat"], np.nan)
    return np.nanmean(features(c), 0)


def _cindex(s, claim, loss):
    known = np.isfinite(loss)
    pp = (claim & known)[:, None] & (claim & known)[None, :] & (np.nan_to_num(loss)[:, None] > np.nan_to_num(loss)[None, :])
    A = pp | (claim[:, None] & ~claim[None, :])
    K = A * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))
    return K.sum() / max(A.sum(), 1)


def _ridge(Z, y, alpha):
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return w, ym


def fit(fields, labels):
    import multiprocessing as mp
    lab = labels.set_index("field_id").loc[[f.field_id for f in fields]]
    with mp.get_context("fork").Pool(8) as p:
        curves = p.map(_train_curves, fields)
    rc, ac = _channels()
    allp = np.concatenate([c.reshape(-1, 13) for c in curves])
    sd = np.nanstd(allp, 0) + 1e-9
    fm = np.array([np.nanmean(c, 0) for c in curves]) / sd              # (n, G, 13) field-mean curves
    okf = np.isfinite(fm).all((1, 2))
    G = fm.shape[1]
    STATE.update(sd=sd, template=fm[okf].mean(0)[:, rc], n_feat=G + G * len(ac))
    with mp.get_context("fork").Pool(8) as p:
        X = np.array(p.map(_train_feats, curves))
    keep = np.isfinite(X).all(1)
    X, lab = X[keep], lab[keep]
    claim = lab.claim.to_numpy(bool)
    loss = lab.loss.to_numpy(float)
    y = np.where(claim, np.where(np.isfinite(loss), loss, CFG["heavy_nan_target"]), CFG["none_target"])
    yr = lab.year.to_numpy()
    best = None
    for a in CFG["alpha_grid"]:                                         # inner leave-one-year-out C
        s = np.zeros(len(y))
        for v in np.unique(yr):
            tr = yr != v
            mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-9
            w, b = _ridge((X[tr] - mu) / sg, y[tr], a)
            s[~tr] = b + ((X[~tr] - mu) / sg) @ w
        c = _cindex(s, claim, loss)
        if best is None or c > best[0]:
            best = (c, a)
    mu, sg = X.mean(0), X.std(0) + 1e-9
    w, b = _ridge((X - mu) / sg, y, best[1])
    STATE.update(mu=mu, sg=sg, w=w, b=b, alpha=best[1], inner_c=best[0])


def predict(f):
    if "w" not in STATE:
        raise RuntimeError("fit first")
    o = f.outline
    F = features(pixel_curves(f))
    s = STATE["b"] + ((F - STATE["mu"]) / STATE["sg"]) @ STATE["w"]
    out = np.full(o.shape, np.nan, dtype="float32")
    out[o] = np.clip(s, 0.0, 1.0)
    return out
