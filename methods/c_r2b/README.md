# c_r2b: canopy-scattering retention per pixel (round 2 challenger)

## Idea
Per outline pixel, ln NIR reflectance (B08) is interpolated on a fixed day-of-year grid (doy 100..324, 16 d, clear
pixels on usable scenes with >= 50 % of the outline clear), and the pixel's own seasonal maximum of that curve is
subtracted: r(t) = ln NIR(t) - max ln NIR <= 0, i.e. the share of the pixel's best NIR scattering left at each date
(log scale). The curve is scale-free per pixel (absolute brightness level removed). Pixel loss =
clip(b + w . r_px, 0, 1). No weather input; no damage-free reference set.

## Learned from labels (fit, development folds only)
- 15 weights + intercept, ridge on standardised field-mean retention curves. Targets as c_r1a: paid = relative
  loss, paid with unknown loss = 0.85, no claim = 0.15.
- Ridge alpha from [10..10000] by inner leave-one-year-out censored C on the training years (paid_unknown only in
  (paid, none) pairs). Fit on all dev: alpha 3000, inner C 0.680, intercept 0.415; weights > 0 for doy 132-196,
  < 0 for doy 212-324.

## Settings chosen by looking at dev scores
1. Signal (ln NIR retention vs own maximum): chosen after screening on dev (field-median curves, LOYO ridge C) of
   about 20 feature families: EVI2 0.647, NDMI 0.642, NDTI 0.641, CIre 0.645, NDRE 0.627, PSRI 0.615, single bands
   0.52-0.675 (B08 0.675), ln B08 0.678, ln B08 minus own max 0.679 (alpha 3000), all 10 bands 0.646, quantile /
   within-field spread curves 0.61-0.65, neighbourhood (window outside outline) curves 0.587-0.607,
   phenology-aligned (own peak / fall) curves 0.46-0.55, extra trees on field curves 0.63-0.67, pixel-level extra
   trees (bands + EVI2 + NDMI, 300 px per field) 0.655, pairwise logistic ranking 0.61-0.63, NMF archetype
   abundances 0.61-0.674, two-window ln NIR contrast with per-fold window search 0.665, single late window
   retention with per-fold window search 0.648, crop-purity pixel masks 0.675-0.676.
2. Variants tried in the method after the first dev score (one look each, not kept):
   bands B07+B08+B8A geometric mean: M1 0.673 (compare p 0.875); reference = mean of top 3 grid values: M1 0.675
   (compare p 0.916). Kept: B08 only, reference = maximum (ref_top_k 1).
3. Grid, usable share, min clear dates, targets 0.15 / 0.85, alpha grid: taken from c_r1a unchanged.

## Dev results (leave one year out)
```
c_r2b dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.674 [0.635, 0.713] | M2 within-year C 0.672 [0.626, 0.713] | M4 AUC 0.699 [0.630, 0.765]
  M3 median none 0.471 mild 0.478 medium 0.498 heavy 0.555 | JT z 7.47 p 0.0000
  M5 MAE paid 0.128 | none >= 0.35 1.000 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:0.750(n4) 2017:0.768(n20) 2018:0.312(n7) 2019:0.689(n79) 2020:0.637(n55) 2021:0.670(n86) 2022:0.782(n12) 2023:0.500(n4) 2024:0.333(n8)
compare c_r2b vs c_r1a dev: diff +0.016 [-0.008, 0.040], p(a > b) 0.9035
```
`cache/` holds label-free per-field mean retention curves (speed only).
