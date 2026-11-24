# c_r4a: four-component pixel map, adding the residue spectrum at the pixel's darkest-NIR post-peak date (round 4 improver of c_r3b)

## Idea
Starts from champion c_r3b (EVI2 calendar profile ridge + residue spectrum at the pixel's lowest-EVI2 date after its
EVI2 maximum). New component: for every outline pixel, the 10-band spectrum on its **lowest-NIR (B08) date after its
own EVI2 maximum** (the pixel's darkest post-season state in canopy scattering, no calendar). Also added, from runner-up
c_r2b: the ln NIR retention curve (ln B08 on the doy grid minus the pixel's own maximum). Each component is a ridge on
field-mean features applied per pixel and clipped to 0..1; pixel loss = mean of the four:

    loss_px = 1/4 [clip(prof) + clip(ret) + clip(res) + clip(resn)]

Pixels without a post-peak date get the field mean of that residue part; pixels with < 3 clear usable dates are NaN.
No weather input, no damage-free set.

## Learned from labels (fit, development folds only)
- 4 ridges on standardised field-mean features: prof 15 + 1, ret 15 + 1, res 10 + 1, resn 10 + 1 weights.
- Alpha: prof and ret by inner leave-one-year-out C over [10..10000] (c_r1a / c_r2b rule); res 3000 (c_r3b);
  resn 3000 (copied from res, not tuned on its own).
- All-dev fit: prof alpha 3000, ret alpha 3000; resn intercept 0.680, weights B02..B12 -0.125 -0.202 -0.110 -0.182
  -0.223 -0.194 -0.177 -0.171 +0.015 +0.055 (raw reflectance scale after standardisation; VNIR negative).
- Targets as c_r1a: paid = relative loss, paid with unknown loss = 0.85, no claim = 0.15.

## Settings chosen by looking at dev scores
All screens offline: own LOYO replica (field-mean ridge, 400-pixel sample per field for the pixel clip, scorer's C,
paired p vs a c_r3b replica; replica M1 0.678 vs scored 0.679, field-score r 0.9999).
1. Signals screened and not used (M1 alone / M1 of 0.5 prof + 0.5 x / p vs c_r3b replica):
   residue variants at alpha 300/1000/3000: ln residue 0.639-0.655 / 0.674-0.679; residue minus pre-season spectrum
   0.600-0.635 / 0.668-0.671; residue band shape 0.591-0.627 / 0.662-0.663; residue indices (NDTI, NDMI, NDVI, CIre,
   ...) 0.550-0.651 / 0.658-0.671; pre-season spectrum 0.45 / 0.640-0.653; residue + peak spectrum 0.639-0.661 /
   0.677-0.680; residue + phenology timings 0.646-0.653 / 0.665-0.673; field-common residue date (lowest field-median
   EVI2 post-peak, 1/2/3 dates) 0.594-0.632 / 0.659-0.675; per-band post-peak minimum 0.638-0.652 / 0.676-0.677,
   per-band season minimum 0.45-0.46, per-band post-peak 20 % quantile / median 0.53-0.61; EVI2-min date >= 30 d
   after peak 0.626-0.649 / 0.673-0.678; residue at **lowest-NIR post-peak date 0.648-0.669 / 0.687-0.688** (kept).
   Residue as multiple-instance sigmoid (lambda 10..300) 0.597-0.654 / 0.672-0.679. ln-band retention curves (own
   max or own min) for all 10 bands 0.52-0.679; NDTI profile 0.650. Pixel aggregation of c_r3b (median, quartiles,
   trimmed mean) 0.674-0.677. None target 0 / 0.1 / 0.2 / 0.25 / 0.3: 0.674 / 0.677 / 0.679 / 0.644 / 0.646.
   Sample weights (no paid_unknown, class balance) 0.649-0.674. Block-scaled joint ridge 0.638-0.671, PLS 1-3
   0.647-0.676. Field-score averages of existing methods' dev scores with c_r3b 0.664-0.683 (p <= 0.72).
   MIL sigmoid on ln NIR retention (c_r3a part B) with prof + res: lambda 100 0.685, 300 0.688, inner [30,100,300] 0.681.
2. Component set (equal weights, fixed rule): prof+res+resn 0.694 (p 0.965 at resn alpha 3000; 0.691 / p 0.86 at
   1000); prof+res+resn+ret 0.695 (p 0.985 at 3000; 0.696 / p 0.965 at 1000); res+resn 0.657-0.666;
   prof+resn 0.688; 0.5 prof + 0.25 res + 0.25 resn 0.687-0.690. Kept: prof+res+resn+ret, resn alpha 3000 (= res).
3. One run of `competition.score dev c_r4a` (below); no change after it.

## Assumes / fixed
Grid, usable share, min clear dates, targets, profile alpha grid from c_r1a; residue definition and alpha from c_r3b;
nir floor and retention definition from c_r2b. `cache/` = label-free per-field mean features (speed only).

## Dev results (leave one year out, competition.score)
```
c_r4a dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.694 [0.657, 0.731] | M2 within-year C 0.697 [0.653, 0.738] | M4 AUC 0.725 [0.657, 0.790]
  M3 median none 0.482 mild 0.492 medium 0.498 heavy 0.525 | JT z 8.52 p 0.0000
  M5 MAE paid 0.132 | none >= 0.35 1.000 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:0.750(n4) 2017:0.839(n20) 2018:0.188(n7) 2019:0.700(n79) 2020:0.675(n55) 2021:0.701(n86) 2022:0.745(n12) 2023:0.500(n4) 2024:0.500(n8)
compare c_r4a vs c_r3b dev: M1 0.694 vs 0.679, diff +0.016 [0.001, 0.031], p(a > b) 0.9825
compare (info): vs c_r3a diff +0.022, p 0.998; vs c_r2b diff +0.020, p 0.998
```
