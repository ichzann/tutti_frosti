# c_r2a: learned calendar profiles of EVI2 + CIre + NDMI per pixel (round 2 improver of c_r1a)

## Idea
Grows from c_r1a. Every outline pixel gets its calendar profile (doy 100..324, every 16 days, linear interpolation
between the pixel's clear dates on usable scenes) of three indices: EVI2, CIre (B07/B05 - 1) and NDMI
((B08 - B11)/(B08 + B11)), concatenated to 45 values. Pixel loss = clip(b + w . profile_px, 0, 1); ridge on
field-mean profiles (= what the scorer averages, up to clipping). Ridge alpha: inner leave-one-year-out C on the
training years, with a tie rule: the largest alpha whose inner C is within 0.02 of the best inner C.

## Learned from labels (fit, development folds only)
- 45 weights + intercept (ridge on standardised field-mean profiles). Targets as c_r1a: paid = relative loss,
  paid with unknown loss -> 0.85, no claim -> 0.15.
- Ridge alpha from the grid [10 .. 30000] by inner LOYO C + tie rule. Fit on all dev: alpha 30000, inner C at that
  alpha 0.620 (best inner C 0.635).

## Settings chosen by looking at dev scores (own LOYO script with the scorer's C, field-mean features)
1. Indices. Screen (c_r1a fit, alpha by inner LOYO, no tie rule), M1 C:
   evi2 0.658, cire 0.651, ndmi 0.654, evi2+cire 0.613, evi2+ndmi 0.657, evi2+cire+ndmi 0.659, evi2+ndre+ndmi 0.622,
   evi2+cire+ndmi+nbr2 0.641, 7 indices (+ndre, nbr2, red, swir) 0.638.
2. Tie rule alpha_tie (and grid + 30000). M1 C / paired p vs c_r1a:
   evi2: 0.0 0.658, 0.005 0.655, 0.01 0.658, 0.015 0.667 / 0.923, 0.02 0.665 / 0.862, 0.03 0.665, 0.04 0.665.
   evi2+cire+ndmi: 0.0 0.659 / 0.543, 0.005 0.656, 0.01 0.656, 0.015 0.668 / 0.916, 0.02 0.669 / 0.926,
   0.025 0.666, 0.03 0.666, 0.04 0.666.
   Chosen: evi2+cire+ndmi, 0.02.

## Other variants screened on dev and not used (M1 C, own LOYO script)
- Ridge with added features (c_r1a fit): + CIre season centre 0.656; + phenology (peak, timing, length, amplitude of
  3 indices) 0.649; + within-field spread per date 0.606; + window-neighbourhood profile 0.638; + pixel drops 0.657;
  10 band profiles 0.623; stage-aligned profile (relative to pixel green-up) + EVI2 0.626.
- Objectives on EVI2 profile: censored target (iterative) 0.661; 2nd-difference penalty 0.598; pairwise logistic 0.623;
  unit-weighted signs 0.643; bagged ridge 0.635; alpha-averaged ridge 0.632.
- Ridge with slope/intercept refit after the ridge (fixed alpha 1000 / 10000): EVI2 0.642 / 0.643.
- Pixel-level gradient boosting (300 pixels/field, target = field label): 0.610 - 0.641.
- Pixel masking by EVI2 amplitude >= 0.2 / 0.3 / 0.4 / 0.5: 0.619 / 0.658 / 0.617 / 0.614.
- Field-score averages c_r1a + c_r1b (0.5 / 0.7 weight): 0.658 / 0.660.
- Univariate field-level C (sign as is): CIre season centre 0.355 (within-year 0.307); neighbourhood-relative
  CIre centre 0.498.

## Assumes / fixed
As c_r1a: scene used if usable and >= 50 % of outline clear; pixel needs >= 3 clear dates; no weather; no
damage-free set (no-claim fields only through the 0.15 target). `cache/` = label-free per-field mean profiles.

## Dev results (leave one year out, competition.score)
```
c_r2a dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.669 [0.627, 0.710] | M2 within-year C 0.682 [0.633, 0.728] | M4 AUC 0.680 [0.600, 0.759]
  M3 median none 0.480 mild 0.483 medium 0.500 heavy 0.513 | JT z 7.31 p 0.0000
  M5 MAE paid 0.139 | none >= 0.35 1.000 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:0.750(n4) 2017:0.598(n20) 2018:0.250(n7) 2019:0.619(n79) 2020:0.625(n55) 2021:0.764(n86) 2022:0.709(n12) 2023:0.250(n4) 2024:0.222(n8)
compare c_r2a vs c_r1a dev: M1 0.669 vs 0.658, diff +0.010 [-0.004, 0.025], p(a > b) 0.926
```
