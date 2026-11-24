# c_r4b: season analogues by elastic matching of pixel trajectories (round 4 challenger)

## Idea
Nearest-neighbour analogues. Every training field is kept as one reference trajectory: log of its field-mean
reflectance in all 10 bands on a day-of-year grid (doy 90..322, 8 d steps; clear values on usable scenes with
>= 50 % of the outline clear, linear interpolation). Every outline pixel of a new field gets its own 10-band log
reflectance trajectory on the same grid and is matched to all reference trajectories by multivariate dynamic time
warping (one shared warping path for all bands, squared Euclidean cost on per-band standardised log reflectance,
Sakoe-Chiba band +-6 grid steps = +-48 d). Pixel loss = mean label target of its k = 40 nearest reference fields.
Field score (scorer) = mean of the pixel map. No weather input, no damage-free reference set, no other method's map.

## Learned from labels (fit, development folds only)
- No weights: fit() stores the training fields' reference trajectories and their targets. Targets as c_r1a:
  paid = relative loss, paid with unknown loss = 0.85, no claim = 0.15.
- Per-band scale (standard deviation of the reference trajectories) from the training fields (label-free).

## Settings chosen by looking at dev scores (own LOYO script on 100 / 400 sampled pixels or field means,
scorer's C; p = paired bootstrap share vs c_r3b, own replica of the scorer's bootstrap)
1. Signal family, field-mean curves, kNN on DTW (W 0..8, k 10..80) vs Euclidean: EVI2 0.585-0.652,
   NDMI 0.579-0.637, ln NIR retention 0.635-0.689, ln NIR 0.657-0.694; per band ln reflectance (W 4):
   B02-B05 0.62-0.65, B06-B8A 0.68-0.70, B11/B12 0.59-0.61; summed per-band DTW: all 10 0.669-0.690,
   B05-B8A 0.691-0.694; joint (shared path) B05-B8A W 2/4/6: 0.673-0.683 / 0.688-0.695 / 0.686-0.699.
2. Pixel queries vs field-mean references (100 px / field, W 6, k 20/40/80): B05-B8A 0.691/0.703/0.701;
   all 10 bands 0.694/0.703/0.698; B08 only 0.681/0.687/0.693. Kept: all 10 bands (no band selection).
3. 400 px / field, all 10 bands, k 20/40/80: W 4 0.696/0.701/0.694; W 6 0.694/0.704/0.701; W 8 0.689/0.697/0.698.
   Kept W 6, k 40. Inner leave-one-year-out selection over k {10,20,40,80} x W {4,6,8}: k 40 in all 9 folds,
   W 6 (6 folds) / 8 (2) / 4 (1); outer C 0.694 (p 0.83) - not used, fixed W 6 / k 40 kept.
4. Variants not kept (400 px, W 6 unless noted): Gaussian neighbour weights (bandwidth = 10th/20th/40th
   neighbour distance) 0.699/0.703/0.703; averaging k {20,40,80} 0.700, k {10..80} 0.699; L1 cost W 6 / W 4
   0.689-0.696 / 0.689-0.694; added first-difference channels 0.679-0.686; season-mean-centred curves
   0.698-0.704; 3 x 3 spatial smoothing of pixel reflectance 0.695-0.704.
5. Grid (doy 90..322, 8 d), usable share, min clear dates, targets: set before the first screen / from c_r1a, not varied.

## Dev results (leave one year out, competition.score)
```
c_r4b dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.705 [0.667, 0.742] | M2 within-year C 0.700 [0.656, 0.741] | M4 AUC 0.732 [0.664, 0.797]
  M3 median none 0.441 mild 0.462 medium 0.492 heavy 0.550 | JT z 8.80 p 0.0000
  M5 MAE paid 0.135 | none >= 0.35 1.000 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:0.250(n4) 2017:0.875(n20) 2018:0.312(n7) 2019:0.737(n79) 2020:0.627(n55) 2021:0.689(n86) 2022:0.855(n12) 2023:0.750(n4) 2024:0.722(n8)
compare c_r4b vs c_r3b dev: M1 0.705 vs 0.679, diff +0.026 [-0.004, 0.057], p(a > b) 0.9515
```
Info: vs c_r1a diff +0.047 p 0.9965; vs c_r3a +0.033 p 0.9825; vs c_r2b +0.031 p 0.9805.
Field-score Spearman with c_r3b 0.794. Dev run 150 s (8 workers).

## Assumes / fixed
One crop season inside the grid window; log reflectance (floor 0.001). Pixels with < 3 clear usable dates get NaN.
