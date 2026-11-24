# Attempts log

Numbers only (no verdicts). One entry per scored method. Dev = leave-one-year-out on development fields.
Columns: round, name, role, idea (1-2 lines), code, dev M1 C [95 %], M2 within-year C, M4 AUC, JT z,
compare vs champion (diff [95 %], p), champion after round.

## Baselines (round 0)

Package 2026-11-08: 391 fields (dev 275: none 66 / mild 68 / medium 64 / heavy 60 / paid_unknown 17; lockbox 116).
Harness fix 2026-11-08 (after round 1 hand-in of c_r1a): 22 paid fields (17 dev, 5 lockbox) have no coverage level
in PSR, so relative loss is unknown; they were labelled heavy with loss NaN and ranked below all paid fields in M1.
Now group paid_unknown: only (paid, none) pairs, not in M3 / M6. All numbers below are after the fix.

| name | M1 C [95 %] | M2 within-year C | M4 AUC | JT z | MAE paid | none >= 0.35 |
|---|---|---|---|---|---|---|
| low_peak_evi2 | 0.480 [0.439, 0.524] | 0.462 | 0.497 | see json | see json | see json |
| ndmi_drop | 0.434 [0.390, 0.479] | 0.438 | 0.393 | see json | see json | see json |
| era5_coldest (weather reference, not a competitor) | 0.489 [0.439, 0.536] | 0.473 | 0.492 | - | - | - |

Full metrics: competition/scores/<name>__dev.json.
Champion after round 0: low_peak_evi2 (code competition/baselines/low_peak_evi2.py).

## Round 1

### c_r1a (improver)
- Idea: per-pixel EVI2 profile on a fixed day-of-year grid (day 100-324, 16 d steps, linear interpolation over
  clear usable dates); pixel loss = clip(b + w . profile, 0, 1), ridge on field-mean profiles.
- Learned from labels: 15 weights + intercept; target loss (paid), 0.85 (paid, loss unknown), 0.15 (no claim);
  ridge alpha by inner leave-one-year-out C (all-dev fit: alpha 3000). Alpha grid was widened once after a dev score
  (grid [10..10000] gave M1 0.612).
- Code: methods/c_r1a/
- Dev: M1 0.658 [0.616, 0.700] | M2 0.674 [0.628, 0.719] | M4 0.671 [0.595, 0.748] (full: scores/c_r1a__dev.json)
- Compare vs low_peak_evi2: diff +0.178 [0.117, 0.239], p 1.000

### c_r1b (challenger)
- Idea: per-pixel season centre (day of year) = midpoint of the half-amplitude rise and fall crossings of the
  red-edge chlorophyll index CIre = B07/B05 - 1 (clear usable dates); pixel loss = hi / (1 + exp((centre - m) / s)).
- Learned from labels: m, s, hi by Nelder-Mead on censored squared error (paid: (score-loss)^2; paid, loss unknown:
  min(score-0.35,0)^2; no claim: max(score-0.35,0)^2); all-dev fit m 241.6, s 30.5, hi 0.671. Index (CIre),
  crossing fraction 0.5, no smoothing chosen on dev screens (table in methods/c_r1b/README.md).
- Code: methods/c_r1b/
- Dev: M1 0.647 [0.605, 0.692] | M2 0.699 [0.653, 0.746] | M4 0.651 [0.563, 0.735] (full: scores/c_r1b__dev.json)
- Compare vs low_peak_evi2: diff +0.167 [0.109, 0.224], p 1.000

Round 1 result: both beat low_peak_evi2. c_r1a vs c_r1b: diff +0.011 [-0.018, 0.040], p 0.762; field-score
Spearman 0.806. Champion after round 1: c_r1a (higher dev M1). no_win = 0.

## Round 2

### c_r2a (improver)
- Idea: c_r1a with EVI2 + CIre + NDMI calendar profiles (45 values per pixel), ridge on field-mean profiles;
  alpha = largest alpha with inner leave-one-year-out C within 0.02 of the best.
- Learned from labels: 45 weights + intercept, targets as c_r1a; all-dev fit alpha 30000.
- Chosen on dev scores: index set (screen evi2 0.658 / evi2+cire+ndmi 0.659 / all 7 0.638 ...), tie rule 0.02
  (0: 0.659, 0.015: 0.668, 0.02: 0.669), alpha 30000 added to grid. Many further variants screened (table in
  methods/c_r2a/README.md), range 0.598-0.661.
- Code: methods/c_r2a/
- Dev: M1 0.669 [0.627, 0.710] | M2 0.682 [0.633, 0.728] | M4 0.680 [0.600, 0.759]
- Compare vs c_r1a: diff +0.010 [-0.004, 0.025], p 0.926 (not a win)

### c_r2b (challenger)
- Idea: per-pixel ln NIR (B08) retention curve r(t) = ln B08(t) - max_t ln B08 on the day 100-324 / 16 d grid;
  pixel loss = clip(b + w . r, 0, 1), ridge on field-mean curves (alpha by inner leave-one-year-out C).
- Learned from labels: 15 weights + intercept, targets as c_r1a; all-dev fit alpha 3000.
- Chosen on dev scores: signal from ~20 screened families (EVI2 0.647, NDMI 0.642, CIre 0.645, B08 0.675,
  ln B08 0.678, all bands 0.646, neighbourhood 0.587-0.607, peak-aligned 0.46-0.55, extra trees 0.63-0.67,
  NMF 0.61-0.674); two post-score variants not kept (B07/B08/B8A mean 0.673, top-3 reference 0.675).
- Code: methods/c_r2b/
- Dev: M1 0.674 [0.635, 0.713] | M2 0.672 [0.626, 0.713] | M4 0.699 [0.630, 0.765]
- Compare vs c_r1a: diff +0.016 [-0.008, 0.040], p 0.904 (not a win)

Round 2 result: no candidate beats c_r1a. Champion after round 2: c_r1a. no_win = 1.

## Round 3

### c_r3a (improver)
- Idea: pixel loss = 0.5 x c_r1a map (ridge on EVI2 calendar profile, clipped) + 0.5 x sigmoid(v . r + c) with r =
  ln NIR retention curve (c_r2b's signal); v, c fitted so field-mean pixel sigmoid matches the label (L2 penalty).
- Learned from labels: ridge part as c_r1a (all-dev alpha 3000); sigmoid part 15 weights + intercept, lambda from
  [30, 100, 300] by inner leave-one-year-out C (all-dev lambda 300). Targets as c_r1a.
- Chosen on dev scores: despiking (not used), combination type (joint ridge 0.633 / average of ridges 0.670 /
  ridge + pixel sigmoid 0.673 kept), lambda grid from a fixed-lambda screen (1..3000: 0.520-0.680), all pixels
  vs 400 per field (0.671, p 0.955). Weight 0.5 fixed before scoring.
- Code: methods/c_r3a/
- Dev: M1 0.672 [0.634, 0.712] | M2 0.676 [0.630, 0.719] | M4 0.693 [0.622, 0.762]
- Compare vs c_r1a: diff +0.014 [-0.001, 0.030], p 0.968 (beats under the rule)
- Compare vs c_r2b (info): diff -0.002 [-0.014, 0.010], p 0.389

### c_r3b (challenger)
- Idea: per pixel, date of own EVI2 maximum, then lowest-EVI2 date after it; 10-band reflectance on that date
  ("residue spectrum") -> ridge; pixel loss = 0.5 x clip(residue ridge) + 0.5 x clip(c_r1a profile ridge).
  Residue part alone: 0.653 (offline leave-one-year-out, alpha 300).
- Learned from labels: residue ridge 10 weights + intercept (alpha fixed 3000); c_r1a part as c_r1a. Targets as c_r1a.
- Chosen on dev scores: signal family from screens (hysteresis 0.42-0.56, spatial 0.47-0.52, spectra at
  phenology points 0.44-0.66, post-season median 0.60-0.64, trees 0.63-0.64); blend rule (grid 0.656, NNLS
  0.650/0.661, joint ridge 0.668, fixed 0.5 kept); residue alpha 100..10000 (0.671-0.679, p 0.82-0.999), 3000 kept.
- Code: methods/c_r3b/
- Dev: M1 0.679 [0.638, 0.718] | M2 0.694 [0.649, 0.740] | M4 0.691 [0.617, 0.763]
- Compare vs c_r1a: diff +0.020 [0.006, 0.036], p 0.998 (beats under the rule)
- Compare (info): vs c_r3a +0.006 [-0.011, 0.023] p 0.779; vs c_r2b +0.004 [-0.018, 0.026] p 0.656

Round 3 result: c_r3a (p 0.968) and c_r3b (p 0.998) beat c_r1a; champion after round 3: c_r3b (higher dev M1).
no_win = 0.

## Round 4

### c_r4a (improver)
- Idea: c_r3b's two parts (EVI2 calendar profile ridge, EVI2-min residue spectrum ridge) + spectrum at each pixel's
  lowest-NIR (B08) date after its own EVI2 peak (ridge) + ln NIR retention curve (c_r2b, ridge); pixel loss =
  equal-weight mean of the four clipped parts.
- Learned from labels: four ridges (15+1, 15+1, 10+1, 10+1); profile / retention alpha by inner LOYO C (all-dev
  3000 / 3000); spectrum alphas fixed 3000. Targets as c_r1a.
- Chosen on dev scores (offline copy of c_r3b, 400 px / field): signal (NIR-min spectrum 0.648-0.669 alone, other
  residue variants 0.45-0.661, all-band retention 0.52-0.679, targets/weights/joint models 0.638-0.679, blends of
  existing scores 0.664-0.683), component set (3 parts 0.694 p 0.965 / 4 parts 0.695 p 0.985), NIR-min alpha
  3000 vs 1000. One scorer run, unchanged after.
- Code: methods/c_r4a/
- Dev: M1 0.694 [0.657, 0.731] | M2 0.697 [0.653, 0.738] | M4 0.725 [0.657, 0.790]
- Compare vs c_r3b: diff +0.016 [0.001, 0.031], p 0.983 (beats under the rule)

### c_r4b (challenger, no blending allowed)
- Idea: nearest-neighbour season analogues. Training fields stored as field-mean ln reflectance trajectories (10 bands,
  day 90-322, 8 d grid); each pixel's 10-band trajectory matched by multivariate DTW (shared path, band +-6 steps);
  pixel loss = mean target of the 40 nearest reference fields.
- Learned from labels: no fitted weights; references + targets (as c_r1a) stored in fit; per-band scale label-free.
- Chosen on dev scores: signal family (EVI2 0.585-0.652, NDMI 0.579-0.637, retention 0.635-0.689, ln NIR
  0.657-0.694, single bands 0.59-0.70), pixel query band sets (0.681-0.703), DTW band W 4/6/8 and k 20/40/80
  (0.689-0.704); inner-LOYO selection of W, k gave 0.694 (not used). Variants not kept 0.679-0.704.
- Code: methods/c_r4b/
- Dev: M1 0.705 [0.667, 0.742] | M2 0.700 [0.656, 0.741] | M4 0.732 [0.664, 0.797]
- Compare vs c_r3b: diff +0.026 [-0.004, 0.057], p 0.952 (beats under the rule)
- Compare (info): vs c_r4a +0.011 [-0.015, 0.035] p 0.791; field-score Spearman with c_r4a 0.853

Round 4 result: c_r4a (p 0.983) and c_r4b (p 0.952) beat c_r3b; champion after round 4: c_r4b (higher dev M1).
no_win = 0.

## Round 5

### c_r5b (challenger, no blending allowed)
- Idea: per-pixel 10-band ln reflectance series (day 90-322, 4 d grid); 300 random dilated MiniRocket kernels
  (1-3 bands each), features = share of season above 25/50/75 % thresholds (label-free); ridge on field means.
- Learned from labels: 900 weights + intercept; alpha [1000..30000] by inner LOYO C (all-dev 30000). Targets as c_r1a.
- Chosen on dev scores: family (k-means codebook 0.606-0.652, spectral-state occupancy 0.508-0.565, path
  signatures 0.594-0.653, kernel ridge 0.520-0.651, MLP 0.574-0.658, stage x calendar 0.548-0.607), kernel
  feature type (0.647-0.677), alpha grid moved up once (1-1000: 0.608-0.671). One scorer run.
- Code: methods/c_r5b/
- Dev: M1 0.653 [0.611, 0.695] | M2 0.668 [0.621, 0.712] | M4 0.678 [0.601, 0.752]
- Compare vs c_r4b: diff -0.052 [-0.081, -0.021], p 0.001 (not a win)

### c_r5a (improver)
- Idea: c_r4b analogues with field-median (not mean) reference trajectories; pixel loss = 0.5 x clipped analogue
  map + 0.5 x c_r4a map (copied unchanged as ridge4.py).
- Learned from labels: analogue references + targets; ridge part as c_r4a.
- Chosen on dev scores (offline replica, 100 px / field): reference type (mean 0.705 / mean log 0.708 / median
  0.709), blend partner (0.5/0.5 with each earlier method 0.692-0.711, c_r4a kept), blend weight (0.707-0.714).
  Variants not kept 0.668-0.711 (table in methods/c_r5a/README.md). One scorer run.
- Code: methods/c_r5a/
- Dev: M1 0.712 [0.675, 0.748] | M2 0.709 [0.666, 0.748] | M4 0.743 [0.675, 0.806]
- Compare vs c_r4b: diff +0.007 [-0.002, 0.017], p 0.941 (not a win)

Round 5 result: no candidate beats c_r4b. Champion after round 5: c_r4b. no_win = 1.

## Round 6

### c_r6b (challenger, no blending allowed)
- Idea: fully constrained spectral unmixing of each clear pixel-date (fixed label-free endmembers) into green
  vegetation / dead matter / soil / shade; pixel GV-NPV-soil season DTW-aligned (W 4) to a label-free mean template;
  features = stage phase (30) + amplitudes at template stages (30 x 13: fractions + ln bands); ridge.
- Learned from labels: 420 weights + intercept; alpha [3000, 10000, 30000] by inner LOYO C. Targets as c_r1a.
- Chosen on dev scores: family (stage-relative fraction scalars ridge 0.605-0.629; within-field spatial structure
  single C 0.50-0.54), alignment variants on field-mean curves (0.677-0.698) and pixel level (0.635-0.700), alpha
  grid from that screen. Not kept: boosting on PCA 0.62-0.65, PCA regression 0.668-0.693. One scorer run.
- Code: methods/c_r6b/
- Dev: M1 0.684 [0.644, 0.723] | M2 0.686 [0.640, 0.729] | M4 0.707 [0.633, 0.773]
- Compare vs c_r4b: diff -0.021 [-0.046, 0.004], p 0.050 (not a win)

c_r6a (improver): first run stopped unfinished at usage limit 2026-11-08; rerun from a fresh c_r4b copy.

### c_r6a (improver, rerun)
- Idea: c_r5a code (field-median analogue references + c_r4a ridge map) with ridge weight 0.3 instead of 0.5.
- Learned from labels: analogue references + targets; ridge part as c_r4a.
- Chosen on dev scores (offline replica, 24-150 px / field): ridge weight 0 / 0.3 / 0.4 / 0.5 -> 0.708 / 0.711 /
  0.712 / 0.712. Structural variants not kept 0.55-0.713 (table in methods/c_r6a/README.md). One scorer run.
- Code: methods/c_r6a/
- Dev: M1 0.711 [0.675, 0.747] | M2 0.705 [0.662, 0.746] | M4 0.742 [0.677, 0.805]
- Compare vs c_r4b: diff +0.006 [0.000, 0.013], p 0.983 (meets beat rule)
- Compare (info): vs c_r5a diff -0.001, p 0.372

Round 6 result: c_r6a p 0.983 vs c_r4b (diff +0.006). User decision 2026-11-08: loop stopped after round 6.

## Lockbox (run once, 2026-11-08; n 116, paid 87, none 29; log competition/scores/lockbox_final.log)

| method | M1 C | M2 within-year C | M4 AUC | M5 MAE paid | none >= 0.35 |
|---|---|---|---|---|---|
| c_r6a | 0.694 [0.633, 0.751] | 0.726 [0.656, 0.791] | 0.741 [0.642, 0.833] | 0.125 | 1.000 |
| c_r4b | 0.683 [0.619, 0.742] | 0.715 [0.643, 0.782] | 0.722 [0.618, 0.819] | 0.128 | 0.897 |
| low_peak_evi2 | 0.448 [0.393, 0.509] | 0.419 [0.348, 0.495] | 0.478 [0.374, 0.583] | 0.230 | 0.241 |
| ndmi_drop | 0.505 [0.437, 0.571] | 0.493 [0.414, 0.575] | 0.484 [0.352, 0.605] | 0.298 | 0.276 |
| era5_coldest | 0.510 [0.434, 0.584] | 0.533 [0.445, 0.624] | 0.485 [0.343, 0.627] | - | - |

Paired M1 lockbox: c_r6a - c_r4b +0.011 [0.001, 0.024] p 0.983; c_r6a - low_peak_evi2 +0.246 [0.160, 0.332] p 1.000;
c_r6a - ndmi_drop +0.189 [0.105, 0.282] p 1.000; c_r6a - era5_coldest +0.184 [0.083, 0.281] p 1.000;
c_r4b - low_peak_evi2 +0.235 p 1.000; c_r4b - ndmi_drop +0.178 p 1.000; c_r4b - era5_coldest +0.173 p 1.000.
M3 medians c_r6a none 0.462 mild 0.464 medium 0.487 heavy 0.540 (JT z 5.61); c_r4b 0.459 / 0.464 / 0.482 / 0.550 (z 5.24).

Lockbox, added 2026-11-08 after the table above (user asked for top 5 approaches; next-best distinct approaches by dev,
c_r5a skipped as near-copy of c_r6a; methods unchanged; log competition/scores/lockbox_top5.log):
| c_r4a | 0.703 [0.641, 0.761] | 0.740 | 0.740 | 0.134 | 1.000 |
| c_r6b | 0.687 [0.628, 0.742] | 0.709 | 0.729 | 0.129 | 0.828 |
| c_r3b | 0.683 [0.621, 0.741] | 0.710 | 0.725 | 0.135 | 1.000 |
Paired M1 lockbox: c_r4a - c_r6a +0.009 p 0.686; c_r4a - c_r4b +0.020 p 0.824; c_r6b - c_r4b +0.004 p 0.594.
Jury page: https://claude.ai/artifact/21m7N4mt4PT99KbjKv5qqt (scripts in session scratchpad).

## Round 7 (extra round, user request 2026-11-09; loop stops after this round whatever the result)

Champion going in: c_r6a (dev M1 0.711). Note: lockbox numbers above were already seen; agents told not to use them for any choice.

### c_r7a (improver)
- Idea: c_r6a (0.7 x DTW season analogues + 0.3 x c_r4a ridge map) with DTW cell cost = (squared scaled 10-band ln
  distance) ** 1.25 (c_r6a: power 1); pixels above the field's 80 % quantile of Euclidean distance to the field-median
  trajectory get the mean loss of the field's other pixels.
- Learned from labels: as c_r6a (analogue references + targets; c_r4a ridges). Cost power and quantile chosen on dev scores.
- Chosen on dev scores (offline LOYO replica, 120 px / field; replica c_r6a 0.7116): families screened 0.609-0.7149
  (band-group DTW 0.639-0.7115, k / weighting 0.691-0.712, residual analogues 0.709-0.713, derivative DTW 0.657-0.705,
  cost functions 0.697-0.7143, atypical quantile 0.6-0.9 0.7138-0.7149; table in methods/c_r7a/README.md). One scorer run.
- Code: methods/c_r7a/
- Dev: M1 0.714 [0.677, 0.750] | M2 0.711 [0.667, 0.751] | M4 0.744 [0.678, 0.807]
- Compare vs c_r6a: diff +0.003 [-0.003, 0.009], p 0.828 (not a win)

### c_r7b (challenger, no blending allowed)
- Idea: diffusion map (alpha 1, width = median distance, 15 coordinates) of pairwise multivariate DTW distances (band
  +-6 steps) between training field-mean ln reflectance trajectories (10 bands, day 90-322, 8 d); ridge (alpha 1000) on
  standardised coordinates; each pixel trajectory placed on the map by Nystrom extension from its DTW distances.
- Learned from labels: 15 ridge weights + intercept. Targets as c_r1a.
- Chosen on dev scores: families (pixel boosting 0.614-0.615, multiple-instance MLP 0.590, phenology scalars 0.50-0.62,
  shapelets 0.598-0.643, elastic proximity forest 0.670-0.674; field DTW kNN reference 0.704), diffusion-map settings
  at field level 0.604-0.710, pixel ridge alpha 100-10000 0.529-0.706. Range 0.53-0.71. One scorer run.
- Code: methods/c_r7b/
- Dev: M1 0.707 [0.668, 0.744] | M2 0.709 [0.665, 0.750] | M4 0.744 [0.677, 0.808]
- Compare vs c_r6a: diff -0.005 [-0.022, 0.013], p 0.305 (not a win); field-score Spearman vs c_r6a 0.898

Round 7 result: no candidate beats c_r6a. Champion after round 7: c_r6a. Loop stopped (user decision 2026-11-09:
round 7 was the last round). No lockbox run for round 7 candidates.

## Max-theory test (outside the champion loop)

Design: max_theory_method.md (method family c_mt), tested 2026-11-09 per its section 4; section 5 not run.
Clock: dev-split imagery only, user decision 2026-11-09 (Module A and every imagery table built from dev fields
only; lockbox imagery not used). Module A table, F1-F8: methods/c_mt/README.md; plot methods/c_mt/report/module_a_clock.png.
delta_y (Module A): 2016 0, 2017 +9, 2018 0, 2019 +4, 2020 +24, 2021 -2, 2022 +6, 2023 0, 2024 0 (2016 / 2018 / 2023:
q never defined, omega_hat = omega_clim). Rule R1: c_mt_full on calendar axis. No lockbox run.

### c_mt_h1 (H1)
- Idea: c_r6a with DOY' = DOY - delta_y (analogue curves and ridge grids). Clock: dev-split imagery only, user decision 2026-11-09.
- Learned from labels: as c_r6a. One scorer run.
- Code: methods/c_mt_h1/ (shared methods/c_mt/)
- Dev: M1 0.694 [0.654, 0.729] | M2 0.691 [0.647, 0.732] | M4 0.720 [0.649, 0.785] | JT z 8.39
- Compare vs c_r6a: diff -0.018 [-0.029, -0.007], p 0.0005

### c_mt_core (evidence alone)
- Idea: pixel loss = clip(b0 + b1 E_p + b2 X~_k); E_p = stage-weighted, clock-locked, persistent abnormal canopy change
  (NDMI, ln B07/B05, ln VIS) per pixel. Clock: dev-split imagery only, user decision 2026-11-09.
- Learned from labels: b0, b1, b2 (b2 = 0 in all 9 folds; b1 0.031-0.095). One scorer run.
- Code: methods/c_mt_core/
- Dev: M1 0.585 [0.539, 0.627] | M2 0.612 [0.554, 0.661] | M4 0.624 [0.545, 0.698] | JT z 3.63
- Compare vs c_r6a: diff -0.127 [-0.181, -0.075], p 0.000

### c_mt_full (candidate)
- Idea: sigmoid(logit c_r6a-prior (calendar, R1) + beta1 E_p + beta2 X~_k). Clock: dev-split imagery only, user decision 2026-11-09.
- Learned from labels: prior as c_r6a; beta by OOF-prior C grid: (0.1, 0) in all 9 folds. One scorer run.
- Code: methods/c_mt_full/
- Dev: M1 0.712 [0.675, 0.745] | M2 0.715 [0.670, 0.755] | M4 0.759 [0.696, 0.817] | JT z 9.08 | none >= 0.35 0.985
- Compare vs c_r6a: diff +0.001 [-0.016, 0.017], p 0.499

### c_mt_perm (F6)
- Idea: c_mt_h1 with delta_y permuted across years (seed 20261109). Clock: dev-split imagery only, user decision 2026-11-09.
- Learned from labels: as c_r6a. One scorer run.
- Code: methods/c_mt_perm/
- Dev: M1 0.697 [0.659, 0.733] | M2 0.699 [0.654, 0.740] | M4 0.724 [0.654, 0.790] | JT z 8.41
- Compare vs c_r6a: diff -0.014 [-0.025, -0.004], p 0.0015; c_mt_h1 vs c_mt_perm: diff -0.003 [-0.015, 0.008], p 0.280

Single-feature C, field-mean E, all dev (F1-F5): true clock 0.604; clock -35 d 0.614; +35 d 0.502; flat clock 0.606;
flat S 0.573; no persistence 0.585; NDMI only 0.592; exposure X alone 0.396.
F7 per-year C c_r6a -> c_mt_h1 (delta): 2016 0.500->0.500 (0), 2017 0.875->0.893 (+9), 2018 0.312->0.188 (0),
2019 0.736->0.731 (+4), 2020 0.650->0.606 (+24), 2021 0.694->0.684 (-2), 2022 0.836->0.836 (+6), 2023 0.750->0.500 (0),
2024 0.722->0.722 (0). F8 median std. E none / mild / medium / heavy -0.757 / -0.198 / 0.079 / 0.248 (JT z 4.44);
c_mt_core M3 JT z 3.63.
