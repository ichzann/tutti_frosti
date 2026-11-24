# c_mt shared code (max-theory method, design: max_theory_method.md)

Outside the champion loop. Tested 2026-11-09 per section 4 of the design doc. Section 5 (single-field window clock)
not run (user instruction).

**Clock: dev-split imagery only, user decision 2026-11-09.** Module A (stage anchors H_k, field and pixel
normal-change tables, per-year event rates q_y, omega_hat_y, delta_y) is built from fields with split == 'dev' in
competition/data/fields.parquet only. Lockbox imagery never enters Module A or any imagery table, not even
unlabeled (`common.clock_fields`, `settings.yaml: clock_split: dev`). Under leave-one-year-out the held-out year's
clock comes from that year's dev fields' unlabeled imagery (no labels). A dev field's own evidence / exposure uses
the leave-self-out clock. Any other field (e.g. a later lockbox) gets its year's dev clock; a year without dev clock
entries gets omega_clim and delta 0. Years where q_y is never defined (N_y < 8 on every day): 2016 (4 dev fields),
2018 (7), 2023 (4) -> omega_hat = omega_clim, delta 0 (shrinkage fallback, doc A7). 2024 (8 dev fields): q defined on
7 days only.

Files: `common.py` (indices, series, intervals, tables, z, persistence, S), `clock.py` (Module A; `python -m
methods.c_mt.clock`), `prior.py` + `ridge4.py` (Module B: copies of c_r6a method.py / ridge4.py, hook DOY' = DOY -
delta_y; ridge cache key includes time axis + delta table; field-level OOF prior for Module D), `evidence.py`
(Module C; `python -m methods.c_mt.evidence`, switch configs for F1-F5), `fchecks.py` (F1-F5, F8), `fuse.py`
(Module D), `settings.yaml` (Module A / C values, all from doc 3.6), `prior_settings.yaml` / `ridge4_settings.yaml`
(copies of c_r6a). Caches in `cache/` (label-free). Reports in `report/`.

Implementation choices not spelled out in the doc (made before any score): cell merge = repeatedly merge the
smallest-count stage group with its smaller neighbour until all >= 30; 5-day smoothing = nanmean over defined days,
omega defined only where q is; leave-self-out clock keeps omega_clim fixed; pixel series = 3x3 smoothed value on
dates where the pixel itself is clear; F5 changes the evidence composite only (clock unchanged); in c_mt_core pixels
with E_p NaN are NaN; field with all-NaN E gets E-bar 0 in fits; X NaN (H NaN) -> training mean.

## Module A output (numbers only; report/module_a.txt, plot report/module_a_clock.png)
e_sat 0.10 (no fallback: every year max omega_hat >= 0.3). d_ref 175.7. Intervals: field 6094, pixel sample
1,217,016. H_k NaN 16 / 275. Pixel table sd_c 0.853, field table sd_c 0.862.

| year | n dev fields | days q defined | max omega_hat | d*_y | delta_y | dates omega_hat >= 0.5 (doy) |
|---|---|---|---|---|---|---|
| 2016 | 4 | 0 | 1.000 | 175.7 | 0 | 76-85, 87-121, 124-131, 219-228, 261-263, 267-268, 271-293 |
| 2017 | 20 | 73 | 1.000 | 184.8 | +9 | 76-85, 87-121, 124-131, 215-241, 261-263, 267-268, 271-293 |
| 2018 | 7 | 0 | 1.000 | 175.7 | 0 | 76-85, 87-121, 124-131, 219-228, 261-263, 267-268, 271-293 |
| 2019 | 79 | 119 | 1.000 | 179.2 | +4 | 76-85, 87-121, 124-131, 185-200, 217-227, 261-263, 267-268, 271-293 |
| 2020 | 55 | 171 | 1.000 | 206.4 | +24 | 76-85, 112-121, 124-131, 246-293 |
| 2021 | 86 | 177 | 1.000 | 173.9 | -2 | 76-85, 87-121, 124-131, 199-205, 222-228, 236-245, 271-293 |
| 2022 | 12 | 56 | 1.000 | 181.8 | +6 | 76-85, 87-121, 124-131, 209-234, 261-263, 267-268, 271-293 |
| 2023 | 4 | 0 | 1.000 | 175.7 | 0 | 76-85, 87-121, 124-131, 219-228, 261-263, 267-268, 271-293 |
| 2024 | 8 | 7 | 1.000 | 175.4 | 0 | 76-85, 87-121, 124-131, 220-228, 261-263, 267-268, 271-293 |

## F1-F5, F8 (report/fchecks.txt): single-feature C of field-mean E, all 275 dev fields, no fit
| config | C | C within-year | AUC | median std. E none / mild / medium / heavy | JT z |
|---|---|---|---|---|---|
| true clock | 0.6044 | 0.6113 | 0.6586 | -0.757 / -0.198 / 0.079 / 0.248 | 4.44 |
| F1 clock -35 d | 0.6144 | 0.6211 | 0.6784 | -0.711 / -0.277 / 0.040 / 0.224 | 4.85 |
| F1 clock +35 d | 0.5015 | 0.4916 | 0.5404 | -0.479 / -0.108 / -0.200 / -0.375 | 0.14 |
| F2 flat clock | 0.6059 | 0.6121 | 0.6656 | -0.671 / -0.167 / 0.073 / 0.213 | 4.55 |
| F3 flat S | 0.5725 | 0.5837 | 0.6422 | -0.657 / 0.206 / -0.131 / 0.081 | 3.15 |
| F4 no persistence | 0.5854 | 0.5757 | 0.6417 | -0.674 / 0.081 / 0.007 / 0.080 | 3.52 |
| F5 NDMI only | 0.5918 | 0.5943 | 0.6660 | -0.830 / 0.093 / -0.184 / 0.221 | 3.83 |
| exposure X alone | 0.3958 | 0.3729 | 0.4034 | 0.267 / 0.147 / -0.192 / -0.428 | -4.79 |
