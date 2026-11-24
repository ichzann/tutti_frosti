# c_r1a: learned calendar EVI2 profile per pixel (round 1 improver of low_peak_evi2)

## Idea
The champion low_peak_evi2 scores a field by 1 - season maximum of the field-median EVI2, i.e. one fixed weight
on the EVI2 level at one moment. Here every outline pixel gets its EVI2 profile on a fixed day-of-year grid
(doy 100..324, every 16 days, linear interpolation between the pixel's clear dates on usable scenes), and a ridge
regression on that profile, learned from development labels, gives the pixel's loss:
pixel loss = clip(b + w . profile_px, 0, 1). The scorer's field mean is then (up to clipping) the same linear
model applied to the field-mean profile, which is what fit() trains on.

## Learned from labels (fit, development folds only)
- 15 weights + intercept (ridge on standardised field-mean profiles).
  Target: paid fields = relative loss (paid heavy with missing loss value -> 0.85); no-claim fields -> 0.15
  (stand-in for censored loss < 0.35).
- Ridge alpha: chosen from the settings grid by inner leave-one-year-out censored concordance on the training years.
- Fit on all development fields: alpha 3000, inner C 0.636; weights > 0 for doy 132-196 and 308-324, < 0 for
  doy 212-292.

## Assumes / fixed
Index EVI2 (as champion); scene used if usable flag and >= 50 % of outline clear; pixel needs >= 3 clear dates.
Grid step sensitivity (field-median profiles, alpha 1000, dev LOYO C): 8 d 0.628, 16 d 0.630, 32 d 0.630.
No weather, no damage-free training set (no-claim fields enter only through the 0.15 stand-in target).
`cache/` holds label-free per-field mean profiles written by fit() (speed only).

## Dev results (leave one year out)
```
c_r1a dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.652 [0.613, 0.691] | M2 within-year C 0.669 [0.624, 0.713] | M4 AUC 0.671 [0.595, 0.748]
  M3 median none 0.474 mild 0.481 medium 0.501 heavy 0.528 | JT z 6.33 p 0.0000
  M5 MAE paid 0.135 | none >= 0.35 0.985 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
compare c_r1a vs low_peak_evi2 dev: diff 0.160 [0.102, 0.218], p(a > b) 1.0
```
Alpha grid [10, 100, 1000, 10000] (first run): M1 0.612 [0.572, 0.655], M2 0.622, M4 0.612.
