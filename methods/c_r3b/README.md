# c_r3b: post-season residue spectrum per pixel (round 3 challenger)

## Idea
For every outline pixel (clear pixels on usable scenes with >= 50 % of the outline clear) the date of its own EVI2
maximum is found, then the date of its lowest EVI2 after that maximum: the pixel's own post-season state (crop
dead / harvested), located without any calendar. The 10-band reflectance on that date is the pixel's residue
spectrum. A ridge on field-mean residue spectra maps it to loss. Second component: the c_r1a EVI2 calendar profile
ridge, unchanged. Pixel loss = 0.5 clip(b_r + w_r . residue_px, 0, 1) + 0.5 clip(b_e + w_e . profile_px, 0, 1).
Pixels without a post-peak date get the field mean of the residue part. No weather input, no damage-free set.

## Learned from labels (fit, development folds only)
- Residue ridge: 10 weights + intercept on standardised field-mean residue spectra, alpha 3000 (fixed).
  All-dev fit: intercept 0.639, weights B02..B12 -0.115 -0.129 -0.089 -0.102 -0.120 -0.113 -0.103 -0.101 -0.015
  +0.025 (per unit reflectance after standardisation back to raw scale; all VNIR bands negative).
- EVI2 profile ridge (c_r1a): 15 weights + intercept, alpha by inner leave-one-year-out C (all-dev: 3000).
- Targets as c_r1a: paid = relative loss, paid with unknown loss = 0.85, no claim = 0.15.

## Settings chosen by looking at dev scores
1. Signal family (residue spectrum at the pixel's own post-peak EVI2 minimum), from exploratory screens on dev
   (400 sampled pixels / field medians; C of single feature, sign either way): hysteresis of bands/indices between
   rising and falling EVI2 leg 0.42-0.56; within-field spatial correlation between dates / spread 0.47-0.52;
   colour at matched falling-EVI2 levels 0.39-0.57 (reversed 0.43-0.61); phenology-anchored 10-band spectra,
   ridge LOYO: pre-season minimum 0.44-0.47, peak 0.49-0.53, falling 50 % 0.55-0.59, post-season minimum 0.65,
   peak + post 0.66. Post-season state as median over dates below 10/20/30 % of the post-peak amplitude: 0.60-0.64
   (single minimum date kept). Extra trees / gradient boosting on residue spectra 0.63-0.64 (not kept).
2. Blend rule. Tried in the method (dev M1): blend weight by inner LOYO C over a grid 0.656; NNLS stacking with
   rescaling 0.650; NNLS weight normalised 0.661 (all with residue alpha by inner LOYO C). Offline replicas: joint
   ridge on profile + residue (nested alpha) 0.668; residue alpha + weight jointly by inner C 0.654; residue alpha
   by blend-aware inner C 0.657; tie rule 0.02 (from c_r2a) at weight 0.5 0.675. Kept: fixed weight 0.5.
3. Residue alpha fixed, screened offline at weight 0.5 (M1 / paired p vs c_r1a): 100 0.671/0.82, 300 0.674/0.89,
   1000 0.678/0.97, 3000 0.679/0.998, 10000 0.672/0.999. Kept 3000.
4. Grid, usable share, min clear dates, targets, profile alpha grid: from c_r1a unchanged.

## Dev results (leave one year out)
```
c_r3b dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.679 [0.638, 0.718] | M2 within-year C 0.694 [0.649, 0.740] | M4 AUC 0.691 [0.617, 0.763]
  M3 median none 0.485 mild 0.489 medium 0.500 heavy 0.522 | JT z 7.90 p 0.0000
  M5 MAE paid 0.133 | none >= 0.35 1.000 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:0.250(n4) 2017:0.759(n20) 2018:0.125(n7) 2019:0.663(n79) 2020:0.639(n55) 2021:0.744(n86) 2022:0.782(n12) 2023:0.500(n4) 2024:0.389(n8)
compare c_r3b vs c_r1a dev: diff +0.020 [0.006, 0.036], p(a > b) 0.9975
```
Residue component alone (offline LOYO, field means, alpha 300): C 0.653; inner LOYO C on all dev at alpha 3000:
0.634. `cache/` holds label-free per-field mean features (speed only).
