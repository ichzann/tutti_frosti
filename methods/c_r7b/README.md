# c_r7b: season manifold regression (round 7 challenger, no blending)

## Idea
The training fields' season trajectories (log of field-mean reflectance, 10 bands, doy 90..322 / 8 d; trajectory
code copied from c_r4b) are laid out on a label-free low-dimensional manifold: a diffusion map (Coifman & Lafon,
density normalisation alpha 1) of the pairwise multivariate DTW distances (band +-6 steps) between the trajectories.
Loss is a ridge function of the first 15 diffusion coordinates (standardised), fitted to the field labels.
Each outline pixel's own 10-band trajectory is placed on the same manifold by the Nystrom out-of-sample extension of
its DTW distances to the training trajectories; pixel loss = ridge(coordinates), clipped 0..1. NaN outside the
outline and for pixels with < 3 clear usable dates. No weather, no damage-free set, no other method's map.

## Learned from labels (fit, development folds only)
- 15 ridge weights + intercept on the diffusion coordinates. Targets as c_r1a: paid = relative loss, paid with
  unknown loss = 0.85, no claim = 0.15. Ridge alpha fixed 1000 (chosen on dev, below).
- Label-free from the training fields: the reference trajectories, per-band scale, kernel width (median DTW distance),
  diffusion eigenvectors, coordinate scale.

## Settings chosen by looking at dev scores
Own leave-one-year-out script, scorer's C (`competition.score._pairs/_c`); p = paired bootstrap share vs scored
c_r6a. "Field" = field-mean curves as queries; "pixel" = 100 sampled outline pixels per field as queries.
1. Families screened before this one (none kept):
   - pixel-instance learners on the 300-value pixel trajectory (60 px / field, field target per pixel): histogram
     gradient boosting 0.614-0.615; numpy multiple-instance MLP (bag mean of pixel sigmoids, MSE + censored pairwise
     rank loss, time-shift / gain augmentation) 0.590.
   - phenometric scalars (EVI2 peak doy, largest post-peak drop doy / size, end-of-season EVI2, within-field IQRs),
     single-feature C 0.501-0.621.
   - shapelet transform (1500 random 3/5/8-step windows of field curves, min distance anywhere / +-2 / +-4 steps,
     top 20/50/100 by train C, ridge): 0.598-0.643.
   - elastic proximity forest (trees splitting by nearest exemplar under DTW, 5 band sets x W 0/3/6, 200 trees):
     0.670-0.674.
   - reference: kNN on the same field DTW matrix (k 40) all bands W 6 / red-edge W 3 / NIR W 6: 0.704 / 0.690 / 0.685.
2. Diffusion map + regression, field level: density alpha {1, 0.5, 0}, kernel width quantile {0.05, 0.1, 0.25, 0.5,
   1.0}, coordinates {3, 5, 10, 15, 20, 25, 30, 50}, ridge alpha {0.3 .. 1000} or kNN on coordinates:
   0.604-0.710 (kNN on coordinates 0.604-0.687). Censored pairwise-rank fit instead of ridge (lambda 0.001-0.1):
   0.694-0.700. Distances summed over band sets / DTW bands: 0.690-0.703.
3. Pixel level (alpha 1, width 0.5, m 15 unless noted): ridge alpha 100 / 300 / 1000 / 3000 / 10000: 0.696 / 0.700 /
   0.706 / 0.656 / 0.529 (m 20: 0.699 / 0.654 / 0.530 at 300 / 3000 / 10000); width 0.25 m 20, alpha 1000: 0.704;
   alpha 0 variants 0.680-0.704; unstandardised coordinates (alpha 0.01-10): 0.694; inner leave-one-year-out choice
   of alpha over [30, 100, 300, 1000]: 0.680 (fold choices 30-1000). Pixel kNN reference on the same pixels 0.705.
   **Kept: alpha 1, width 0.5, m 15, standardised coordinates, ridge alpha 1000 fixed.**
4. One run of `competition.score dev c_r7b` (below); no change after it.

## Dev results (leave one year out, competition.score)
```
c_r7b dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.707 [0.668, 0.744] | M2 within-year C 0.709 [0.665, 0.750] | M4 AUC 0.744 [0.677, 0.808]
  M3 median none 0.482 mild 0.492 medium 0.494 heavy 0.516 | JT z 8.74 p 0.0000
  M5 MAE paid 0.132 | none >= 0.35 1.000 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:0.250(n4) 2017:0.884(n20) 2018:0.438(n7) 2019:0.755(n79) 2020:0.658(n55) 2021:0.684(n86) 2022:0.800(n12) 2023:0.500(n4) 2024:0.778(n8)
compare c_r7b vs c_r6a dev: M1 0.7068 vs 0.7114, diff -0.005 [-0.022, 0.013], p(a > b) 0.305
compare (info) c_r7b vs c_r4b dev: diff +0.002, p 0.584
```
Field-score Spearman with c_r6a 0.898, with c_r4b 0.894. Dev run 321 s (8 workers).

## Assumes / fixed
As c_r4b (grid, usable share 0.5, min 3 clear dates, reflectance floor 0.001, DTW band 6, targets from c_r1a).
One crop season inside the grid window.
