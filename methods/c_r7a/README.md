# c_r7a: c_r6a with powered DTW cell cost and atypical-pixel replacement (round 7 improver)

## Idea
Starts from champion c_r6a (pixel loss = 0.7 x season-analogue map + 0.3 x c_r4a ridge map; analogues = each outline
pixel's 10-band log reflectance trajectory, doy 90..322 / 8 d, matched by multivariate DTW, band +-6, to every
training field's field-median trajectory; analogue loss = mean target of the 40 nearest reference fields).
Two changes:
1. **DTW cell cost** = (squared distance of the scaled 10-band log vector) ** 1.25 (c_r6a: power 1). Large
   per-date spectral departures weigh more in the warped distance than small ones.
2. **Atypical pixels**: per field, each pixel's scaled log trajectory is compared (plain Euclidean, no warping) with
   the median trajectory of the field's own pixels. Pixels above the field's 80 % quantile of that distance get the
   mean loss of the other 80 % of the field (their own value is replaced, they keep a value).
Everything else as c_r6a (c_r4a ridge code unchanged in `ridge4.py`, settings `ridge4_settings.yaml`, label-free
feature cache `cache/`). No weather input, no damage-free set.

## Learned from labels (fit, development folds only)
- Analogue part: no weights; references + targets (paid = loss, paid with unknown loss = 0.85, no claim = 0.15).
- Ridge part: as c_r4a (4 ridges; prof / ret alpha by inner leave-one-year-out C, res / resn alpha 3000).
- Cost power and atypical quantile: chosen on dev scores (below), not fitted.

## Settings chosen by looking at dev scores
Offline replica (`_scratch/`): 120 sampled outline pixels per dev field, DTW to all dev field-median references,
leave-one-year-out by reference year, ridge parts refitted per fold on field means; scorer's C and paired bootstrap
(p = share of resamples above scored c_r6a). Replica of c_r6a: 0.7116 (scored 0.7114).
Values: analogue part alone C / blend (0.7 analogue + 0.3 ridge) C.

| Screen | Variants | C range |
|---|---|---|
| band-group DTW (vis / red edge / NIR / SWIR, separate warping), weighted sums 0/1/2 per group (80 combos) | alone 0.609-0.708, blend 0.639-0.7115 | no gain |
| averaging group-wise analogue maps | blend 0.705-0.713 | |
| k 10 / 20 / 40 / 80 / 160; Gaussian distance-weighted kNN; multi-k mean | blend 0.691-0.712 | |
| residual analogues: ridge map + mean of the k nearest references' nested-LOYO ridge residuals (k 20-160, gain 0.5 / 1) | alone 0.696-0.712, with analogue 0.709-0.713 | |
| reference editing (drop 5-30 % of references with largest neighbour disagreement) / target smoothing toward neighbour mean | blend 0.688-0.713 | |
| class-conditional distance ratio (paid vs none nearest) | 0.628-0.641 | |
| target encodings (loss ranks, binary, none target 0 / 0.3) | blend 0.686-0.711 | |
| pixel aggregation (power 0.5-6 on pixel map, field quantile 0.25-0.9) | 0.696-0.713 | |
| derivative DTW (log + beta x first difference, beta 0.5 / 1 / 2 / only) | blend 0.657-0.705 | |
| DTW band W 4 / 8, field-mean references, ensembles of these | blend 0.707-0.711 | |
| spatially smoothed query pixels (3x3 / 5x5 clear-pixel mean) | blend 0.712 | |
| pairwise-logistic (rank) ridge parts instead of ridge, lambda 0.01-1 | blend 0.706-0.711 | |
| NNLS stacker of analogue + 4 ridge parts (nested LOYO) | 0.705 (weights: analogue only) | |
| reference dropping by within-field spread (10-30 %) | blend 0.708-0.712 | |
| **DTW cell cost**: log1p / sqrt / pseudo-Huber (delta^2 1, 4, 16) / power 1 / 1.25 / 1.5 / 2 | alone 0.692 / 0.698 / 0.703-0.707 / 0.7085 / 0.7109 / 0.7109 / 0.7098; blend 0.697-0.7143 | **1.25 kept** (blend 0.7141, p 0.878) |
| **atypical pixels** quantile 0.6 / 0.7 / 0.8 / 0.9 (power 1) | 0.7124 / 0.7130 / 0.7129 / 0.7120 | |
| power 1.25 + atypical quantile 0.6 / 0.7 / 0.8 / 0.9 | 0.7143 / 0.7148 / 0.7149 / 0.7138 | **0.8 kept** (p 0.865) |

One run of `competition.score dev c_r7a` (below); no change after it.

## Dev results (leave one year out, competition.score)
```
c_r7a dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.714 [0.677, 0.750] | M2 within-year C 0.711 [0.667, 0.751] | M4 AUC 0.744 [0.678, 0.807]
  M3 median none 0.451 mild 0.471 medium 0.496 heavy 0.545 | JT z 9.19 p 0.0000
  M5 MAE paid 0.131 | none >= 0.35 1.000 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:0.250(n4) 2017:0.929(n20) 2018:0.312(n7) 2019:0.738(n79) 2020:0.659(n55) 2021:0.700(n86) 2022:0.836(n12) 2023:0.750(n4) 2024:0.722(n8)
compare c_r7a vs c_r6a dev: M1 0.7142 vs 0.7114, diff +0.003 [-0.003, 0.009], p(a > b) 0.828
```
Dev run 382 s (8 workers). Scratch scripts of the offline replica: `_scratch/` (pre*.py, dtw*.py, t*.py).

## Assumes / fixed
As c_r6a (grids, usable share, min clear dates, floors, targets, k 40, W 6, ridge weight 0.3).
