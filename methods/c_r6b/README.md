# c_r6b: spectral unmixing + elastic registration of the fraction season course (round 6 challenger)

## Idea
1. Each clear pixel-date spectrum (10 bands) is unmixed by fully constrained least squares (non-negative, sum to one)
   into green vegetation (GV), non-photosynthetic vegetation (NPV), soil and shade; GV / NPV / soil are
   shade-normalised. Endmembers are fixed label-free spectra (settings.yaml).
2. Each pixel's GV / NPV / soil course (doy 90-322, 8 d grid, linear interpolation over clear usable dates) is
   registered by dynamic time warping (band +-4 steps) to one template: the mean fraction course of the training
   fields (label-free). The warp splits the season into phase (when the pixel reached each template stage, relative
   to the template) and amplitude (fractions and ln band reflectances read at each template stage).
3. Pixel loss = clip(b + w . [phase (30), amplitude (30 x 13)], 0, 1), ridge on training-field means of the
   registered pixel features (400 px / training field).

## Learned from labels
Ridge weights (420) + intercept; alpha from [3000, 10000, 30000] by inner leave-one-year-out C. Targets: loss (paid),
0.85 (paid, loss unknown), 0.15 (no claim), from c_r1a. Template, channel scales and endmembers are label-free.

## Assumptions
Three materials + shade explain the canopy spectrum; one template season with elastic timing fits all fields.
No weather input.

## Settings chosen by looking at dev scores (own offline screens, 400 px / field, field-mean ridge, scorer's C)
1. Main family. Screened first: stage-relative scalar features of the unmixed fractions (peak GV, green / fill
   durations, GV retention, NPV / soil 8-64 d after GV peak, max GV drop, NPV rise, pre-peak dip; 29 features):
   single C 0.50-0.63, ridge 0.605-0.629. Within-field spatial structure (rank correlation of pixel EVI2 between
   consecutive green dates, early-late GV pattern correlation, within-field GV SD): single C 0.50-0.54.
2. Registration (field-mean curves, ridge alpha 30..100000): ln 10-band registration both blocks W 3/4/6/8/10 best
   0.683-0.692 (raw unregistered ln curves 0.637-0.644); fraction registration amplitude-only 0.677-0.691;
   fractions + ln 0.689-0.698.
3. Pixel-level registration (per-fold template), M1 / paired p vs c_r4b: fractions only W 4: both 0.677-0.681,
   amp 0.655-0.695; fractions + ln (registered on both) both 0.685-0.699, amp 0.690-0.701; register on GV only,
   amplitude all: 0.684-0.694; register on fractions, amplitude all (kept): both 0.685/0.695/0.700/0.700 at
   alpha 1000/3000/10000/30000 (p 0.08-0.34); CLR-transformed fractions 0.635-0.699.
4. Not kept: gradient boosting on PCA of registered features 0.62-0.65, PCA regression 0.668-0.693; within-field
   phase SD added 0.692-0.698; 25 % / 75 % pixel quantiles of registered features 0.691-0.698.
Kept: register on GV/NPV/soil, amplitude of fractions + ln bands, W 4, alpha grid [3000, 10000, 30000] (inner LOYO).
Endmember rules, grid, min_sunlit, train_pixels set before scoring, not varied.

## Result (one scorer run)
```
c_r6b dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.684 [0.644, 0.723] | M2 within-year C 0.686 [0.640, 0.729] | M4 AUC 0.707 [0.633, 0.773]
  M3 median none 0.465 mild 0.493 medium 0.507 heavy 0.568 | JT z 7.92 p 0.0000
  M5 MAE paid 0.130 | none >= 0.35 0.924 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:0.500(n4) 2017:0.857(n20) 2018:0.312(n7) 2019:0.725(n79) 2020:0.678(n55) 2021:0.651(n86) 2022:0.782(n12) 2023:1.000(n4) 2024:0.667(n8)
compare c_r6b vs c_r4b dev: M1 0.684 vs 0.705, diff -0.021 [-0.046, 0.004], p(a > b) 0.050
```
Offline replica at fixed alpha 10000 (400 px / field, one global channel scale): 0.700. Dev run 110 s (8 workers).
