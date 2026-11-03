"""Scorecard from the per-pixel table of a validation run (method-independent)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

MISSED = -1e9   # score given to an injected pixel the method no longer returns as valid


def _auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def _r(v, nd=4):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def case_table(pix, thr, mcfg):
    """One row per case: field-level null / injected statistics."""
    p = pix.assign(fl_null=pix.valid_null & (pix.score_null >= thr),
                   fl_inj=pix.valid_inj & (pix.score_inj >= thr),
                   s_null=pix.score_null.where(pix.valid_null),
                   s_inj=pix.score_inj.where(pix.valid_inj))
    g = p.groupby("case_id", observed=True)
    c = pd.DataFrame(dict(n_px=g.size(), n_valid_null=g.valid_null.sum(), n_valid_inj=g.valid_inj.sum(),
                          n_fl_null=g.fl_null.sum(), n_fl_inj=g.fl_inj.sum(),
                          mean_null=g.s_null.mean(), mean_inj=g.s_inj.mean()))
    mv = mcfg["field_min_valid_px"]
    c["share_null"] = c.n_fl_null / c.n_valid_null.where(c.n_valid_null >= mv)
    c["share_inj"] = c.n_fl_inj / c.n_valid_inj.where(c.n_valid_inj >= mv)
    c["flag_null"] = c.share_null >= mcfg["field_flag_share"]
    c["flag_inj"] = c.share_inj >= mcfg["field_flag_share"]   # NaN share (too few valid) -> not flagged
    if "score_haze" in pix:                                      # nuisance run: haze on the first obs after
        fh = pix.valid_haze & (pix.score_haze >= thr)
        gh = fh.groupby(pix.case_id, observed=True)
        c["n_valid_haze"] = pix.valid_haze.groupby(pix.case_id, observed=True).sum()
        c["n_fl_haze"] = gh.sum()
        c["share_haze"] = c.n_fl_haze / c.n_valid_haze.where(c.n_valid_haze >= mv)
        c["flag_haze"] = c.share_haze >= mcfg["field_flag_share"]
    return c


def _headline(pix, cases, thr, crit, mcfg):
    """The metrics the criteria use (also bootstrapped)."""
    v = pix[pix.valid_null]
    pos = v[(v.w > 0) & (v.drop_rel > 0)]
    out = dict(coverage=float(pix.valid_null.mean()),
               fpr_px=float((v.score_null >= thr).mean()) if len(v) else np.nan)
    cn = cases[cases.share_null.notna()]
    out["fpr_field"] = float(cn.flag_null.mean()) if len(cn) else np.nan
    pl = pos[pos.level == crit["I1_auc_level"]]
    s = np.where(pl.valid_inj, pl.score_inj, MISSED)
    out["auc_px@I1"] = _auc(s, v.score_null.values)
    pl = pos[pos.level == crit["I2_tpr_level"]]
    out["tpr_px@I2"] = float((pl.valid_inj & (pl.score_inj >= thr)).mean()) if len(pl) else np.nan
    ci = cases[(cases.level == crit["I3_field_level"]) & (cases.pattern == "whole") & cases.share_null.notna()]
    out["field_det@I3"] = float(ci.flag_inj.mean()) if len(ci) else np.nan
    pi = pos[pos.level == crit["I3_field_level"]]
    out["validity_ratio@I3"] = float(pi.valid_inj.mean()) if len(pi) else np.nan
    if "score_haze" in pix:
        h = pix[pix.valid_haze]
        out["fpr_px_haze"] = float((h.score_haze >= thr).mean()) if len(h) else np.nan
        ch = cases[cases.share_haze.notna()]
        out["fpr_field_haze"] = float(ch.flag_haze.mean()) if len(ch) else np.nan
    return out


def _bootstrap(pix, cases, thr, crit, mcfg, n, seed=0):
    fields = cases.field_id.unique()
    pix_by = {f: i for f, i in pix.groupby("field_id", observed=True).indices.items()}
    case_by = {f: i for f, i in cases.reset_index().groupby("field_id").indices.items()}
    rng = np.random.default_rng(seed)
    reps = []
    cr = cases.reset_index()
    for _ in range(n):
        fs = rng.choice(fields, len(fields))
        pi = np.concatenate([pix_by[f] for f in fs if f in pix_by])
        ci = np.concatenate([case_by[f] for f in fs])
        reps.append(_headline(pix.iloc[pi], cr.iloc[ci], thr, crit, mcfg))
    r = pd.DataFrame(reps)
    return {k: [_r(r[k].quantile(0.025)), _r(r[k].quantile(0.975))] for k in r}


def probabilistic(pix):
    """Calibration of a predictive distribution (only if the method returns predicted/pred_std/observed)."""
    v = pix[pix.valid_null & np.isfinite(pix.pred) & (pix.pstd > 0)]
    if not len(v):
        return None

    def stats(d):
        z = (d.obs - d.pred) / d.pstd
        crps = d.pstd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        pit = np.histogram(norm.cdf(z), bins=10, range=(0, 1))[0] / len(z)
        return dict(n_px=len(d), z_mean=_r(z.mean(), 3), z_sd=_r(z.std(), 3), z_median=_r(z.median(), 3),
                    cov95=_r((np.abs(z) <= 1.96).mean()), share_z_le_m2=_r((z <= -2).mean()),
                    bias_evi2=_r((d.obs - d.pred).mean()), mae_evi2=_r((d.obs - d.pred).abs().mean()),
                    median_pred_std=_r(d.pstd.median()), crps=_r(crps.mean()),
                    pit_deciles=[_r(x, 3) for x in pit])
    return dict(all=stats(v), by_stage={s: stats(d) for s, d in v.groupby("stage", observed=True)})


def scorecard(pix, cases_meta, thr, vcfg):
    mcfg, crit = vcfg["metrics"], vcfg["criteria"]
    ct = case_table(pix, thr, mcfg).join(cases_meta.set_index("case_id"))
    head = _headline(pix, ct, thr, crit, mcfg)
    ci = _bootstrap(pix, ct, thr, crit, mcfg, mcfg["bootstrap"]) if mcfg["bootstrap"] else {}
    v = pix[pix.valid_null]
    neg = v.score_null.values
    q = float(np.quantile(neg, 1 - mcfg["fpr_target"])) if len(neg) else np.nan

    # null, by stage
    null_stage = {}
    for s, d in pix.groupby("stage", observed=True):
        dv = d[d.valid_null]
        cs = ct[(ct.stage == s) & ct.share_null.notna()]
        null_stage[s] = dict(n_cases=int((ct.stage == s).sum()), coverage=_r(d.valid_null.mean()),
                             fpr_px=_r((dv.score_null >= thr).mean()) if len(dv) else None,
                             fpr_field=_r(cs.flag_null.mean()) if len(cs) else None)

    # injection, pixel level
    pos = v[(v.w > 0) & (v.drop_rel > 0)]
    s_all = np.where(pos.valid_inj, pos.score_inj, MISSED)
    pos = pos.assign(s=s_all)

    def px_block(d, negs):
        return dict(n_px=len(d), auc=_r(_auc(d.s.values, negs)),
                    tpr_at_thr=_r((d.s >= thr).mean()), tpr_at_fpr=_r((d.s > q).mean()),
                    validity=_r(d.valid_inj.mean()), median_drop_abs=_r(d.drop_abs.median()))
    by_level = {}
    for (pat, lev), d in pos.groupby(["pattern", "level"], observed=True):
        by_level.setdefault(pat, {})[str(lev)] = px_block(d, neg)
    by_level_stage = {}
    for (st, lev), d in pos.groupby(["stage", "level"], observed=True):
        by_level_stage.setdefault(st, {})[str(lev)] = px_block(d, v[v.stage == st].score_null.values)
    bins = mcfg["drop_bins"]
    pos["drop_bin"] = pd.cut(pos.drop_abs, bins)
    by_drop = {str(b): px_block(d, neg) for b, d in pos.groupby("drop_bin", observed=True)}
    det_limit = {}
    for key in ("tpr_at_thr", "tpr_at_fpr"):
        for need in (0.5, 0.8):
            hit = [b for b in by_drop if by_drop[b]["n_px"] >= 200 and by_drop[b][key] >= need]
            det_limit[f"{key}>={need}"] = hit[0] if hit else None

    # injection, field level and localisation
    field_level = {}
    cn_mean = ct.mean_null[ct.share_null.notna()].values
    for (pat, lev), d in ct[ct.share_null.notna()].groupby(["pattern", "level"]):
        m_inj = d.mean_inj.fillna(MISSED).where(d.n_valid_inj >= mcfg["field_min_valid_px"], MISSED)
        field_level.setdefault(pat, {})[str(lev)] = dict(
            n_cases=len(d), detected=_r(d.flag_inj.mean()), auc_mean_score=_r(_auc(m_inj.values, cn_mean)))
    by_half = {}
    if "half_days" in pos:
        for (hd, lev), d in pos.groupby(["half_days", "level"], observed=True):
            by_half.setdefault(str(hd), {})[str(lev)] = px_block(d, neg)
    haze = None
    if "score_haze" in pix:
        haze = {}
        for hb, d in pix[pix.valid_haze].groupby("haze_blue", observed=True):
            cs = ct[(ct.haze_blue == hb) & ct.share_haze.notna()]
            haze[str(hb)] = dict(n_px=len(d), fpr_px=_r((d.score_haze >= thr).mean()),
                                 fpr_field=_r(cs.flag_haze.mean()) if len(cs) else None,
                                 validity=_r(pix[pix.haze_blue == hb].valid_haze.mean()))
    loc = {}
    lowc = pix[(pix.pattern == "low") & pix.valid_inj]
    for lev, d in lowc.groupby("level", observed=True):
        loc[str(lev)] = dict(auc_in_vs_out=_r(_auc(d.score_inj[d.w > 0].values, d.score_inj[d.w == 0].values)),
                             fpr_outside=_r((d.score_inj[d.w == 0] >= thr).mean()))

    # criteria
    prob = probabilistic(pix)
    cr = {}
    cr["N1_coverage"] = head["coverage"] >= crit["N1_coverage_min"]
    cr["N2_fpr_px"] = head["fpr_px"] <= crit["N2_fpr_px_max"]
    cr["N3_fpr_field"] = head["fpr_field"] <= crit["N3_fpr_field_max"]
    if prob:
        lo, hi = crit["N4_cov95_range"]
        cr["N4_cov95"] = lo <= prob["all"]["cov95"] <= hi
    else:
        cr["N4_cov95"] = None
    cr["I1_auc"] = head["auc_px@I1"] >= crit["I1_auc_min"]
    cr["I2_tpr"] = head["tpr_px@I2"] >= crit["I2_tpr_min"]
    cr["I3_field"] = head["field_det@I3"] >= crit["I3_field_min"]
    vr = head["validity_ratio@I3"]
    cr["I4_validity"] = vr >= crit["I4_validity_ratio_min"]
    if "fpr_field_haze" in head and "N5_haze_fpr_field_max" in crit:
        cr["N5_haze_field"] = head["fpr_field_haze"] <= crit["N5_haze_fpr_field_max"]
    cr = {k: (None if v is None else bool(v)) for k, v in cr.items()}

    return dict(
        threshold=thr, n_fields=int(ct.field_id.nunique()), n_cases=len(ct),
        n_cases_by_stage=ct.stage.value_counts().to_dict(),
        headline={k: _r(v) for k, v in head.items()}, headline_ci95=ci,
        criteria=cr, passed=all(v for v in cr.values() if v is not None),
        null=dict(score_q95=_r(q), by_stage=null_stage, probabilistic=prob),
        injection=dict(px_by_pattern_level=by_level, px_by_stage_level=by_level_stage,
                       px_by_drop_abs=by_drop, detection_limit_drop_abs=det_limit,
                       field_by_pattern_level=field_level, localisation_low=loc,
                       px_by_half_days_level=by_half),
        haze=haze), ct
