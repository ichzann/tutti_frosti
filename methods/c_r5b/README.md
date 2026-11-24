# c_r5b: random convolutional kernel transform of pixel spectral time series (round 5 challenger)

## Idea
Every outline pixel's season is a 10-band log reflectance series on a day-of-year grid (doy 90..322, 4 d steps;
clear values on usable scenes with >= 50 % of the outline clear, linear interpolation). A fixed bank of 300 random
dilated convolution kernels (MiniRocket shapes: 9 taps, -1 x 6 / +2 x 3, zero sum; each kernel mixes 1-3 random bands
with random weights; dilation 1..4) runs along time. Each kernel output gives 3 PPV features: share of the season
in which the output exceeds a threshold (thresholds = 25/50/75 % quantiles of that kernel's output on training
pixels, label-free). 900 PPV features per pixel. Pixel loss = clip(b + w . ppv_px, 0, 1). Field score (scorer) =
mean of the pixel map. No weather, no damage-free reference set, no other method's map or model.

## Learned from labels (fit, development folds only)
- Ridge (standardised features) on field-mean PPV of 400 sampled pixels per training field: 900 weights + intercept.
  Targets as c_r1a: paid = relative loss, paid with unknown loss = 0.85, no claim = 0.15.
- Ridge alpha from [1000, 3000, 10000, 30000] by inner leave-one-year-out censored C. All-dev fit: alpha 30000,
  inner C 0.677, intercept 0.673.
- Label-free from training fields: per-band mean / sd of log reflectance, kernel thresholds.

## Settings chosen by looking at dev scores (own LOYO script, 400 px / field, scorer's C, field-mean ridge)
1. Other main ideas screened first and not used (best M1 per family): pixel-trajectory k-means codebook +
   bounded least squares cluster losses (K 16..128) 0.606-0.652; single-date spectral-state occupancy (k-means
   states, ridge) 0.508-0.565; path signatures (depth 3-4, PCA bands +- time channel, ridge) 0.594-0.653; RBF /
   Laplacian kernel ridge on field-mean curves 0.520-0.651; MLP on field-mean curves 0.574-0.658; stage x calendar
   bilinear exposure model (ln B07 stage) 0.548-0.607.
2. Kernel transform variants (alpha 300..30000, unclipped field means): MiniRocket ppv 300 kernels 0.661-0.677
   (alpha 1..1000 first: 0.608-0.671, grid then moved up); ppv+max 1000 kernels 0.651-0.673; random normal weights
   ppv+max 1000 0.647-0.667; ppv in 3 calendar windows 500 kernels 0.650-0.659; MLP readout on 300-kernel ppv
   0.656-0.675. Kept: MiniRocket ppv, 300 kernels, alpha grid [1000..30000] by inner LOYO.
3. Grid, usable share, min clear dates, targets, kernel count 300, max channels 3, quantiles: set before the
   first screen, not varied. One scorer run, no change after it.

## Dev results (leave one year out, competition.score)
```
c_r5b dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.653 [0.611, 0.695] | M2 within-year C 0.668 [0.621, 0.712] | M4 AUC 0.678 [0.601, 0.752]
  M3 median none 0.461 mild 0.484 medium 0.501 heavy 0.570 | JT z 6.44 p 0.0000
  M5 MAE paid 0.141 | none >= 0.35 0.864 | M6 dmg share none 0.999 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:0.500(n4) 2017:0.795(n20) 2018:0.562(n7) 2019:0.702(n79) 2020:0.633(n55) 2021:0.652(n86) 2022:0.673(n12) 2023:0.500(n4) 2024:0.389(n8)
compare c_r5b vs c_r4b dev: M1 0.653 vs 0.705, diff -0.052 [-0.081, -0.021], p(a > b) 0.001
```
Dev run 302 s (9 fits + 275 fields, 8 workers).

## Assumes / fixed
One crop season inside the grid window; log reflectance (floor 0.001); pixels with < 3 clear usable dates get NaN.
