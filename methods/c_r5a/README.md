# c_r5a: c_r4b season analogues with field-median references, averaged with the c_r4a ridge map (round 5 improver)

## Idea
Starts from champion c_r4b (nearest-neighbour season analogues by multivariate DTW of 10-band log reflectance
trajectories, doy 90..322 / 8 d, band +-6 steps, k = 40). Two changes:
1. Reference trajectory of a training field = log of its field-**median** reflectance per usable date (c_r4b: log of
   the field mean). Per-band scale still from the field-mean curves (as c_r4b). Pixel queries unchanged.
2. Pixel loss = 0.5 x analogue map (clipped 0..1) + 0.5 x c_r4a map (mean of four clipped ridge maps: EVI2 calendar
   profile, ln NIR retention, residue spectrum at lowest-EVI2 and at lowest-NIR post-peak date). c_r4a code copied
   unchanged to `ridge4.py` with its settings in `ridge4_settings.yaml` and its own label-free feature cache `cache/`.
   Where one part has no value the other is used; NaN outside the outline.
DTW implementation stores only the cells inside the band (same values as c_r4b's dtw, checked on random data: max
difference 0). No weather input, no damage-free set.

## Learned from labels (fit, development folds only)
- Analogue part: no weights; references + targets (paid = loss, paid with unknown loss = 0.85, no claim = 0.15).
- Ridge part: as c_r4a (4 ridges, prof / ret alpha by inner leave-one-year-out C, res / resn alpha 3000).

## Settings chosen by looking at dev scores
Offline replica: 100 sampled pixels per dev field, DTW to all 275 dev field references (global per-band scale),
leave-one-year-out by reference year, scorer's C and bootstrap; replica of c_r4b C 0.7050 (scored 0.7050, field-score
r 0.9985). p = share of bootstrap resamples above the c_r4b replica / scored c_r4b.
1. Analogue variants (not kept unless noted), C / p:
   - k-means centroids of training-field pixels (4 per field) as references: k 40/80/160/240 0.692/0.700/0.706/0.703;
     field mean + centroids k 80/160 0.699/0.707.
   - pixel aggregation in the field score (power 0.5/2/4, quantile 0.25/0.5/0.75/0.9): 0.696-0.706.
   - year-centred reference targets (lambda 0.5/1): 0.703/0.701. Rank targets: 0.706. None target 0/0.1/0.25/0.35
     (field-mean query proxy): 0.702-0.705.
   - reference editing (drop 10/20/30 % worst leave-one-out refs) 0.668-0.692; target smoothing 0.702-0.704 (field proxy).
   - DTW band W 4/8 0.702/0.698, W-ensemble 4/6/8 0.703.
   - joint distance DTW + residue spectrum (res / resn, log or raw, beta 0.25/0.5/1): 0.668-0.697; residue alone 0.64-0.65.
   - per-band weights and per-time-segment weights in the DTW cost (coordinate search, in-sample, field proxy):
     no change from 1.0 improved C.
   - reference = mean of log 0.708 (p 0.964); **reference = median 0.709 (p 0.933, kept)**; mean + meanlog + median
     refs k 120 0.707. Pixel + field-query combined distance 0.707; field query alone 0.696.
   - regression-adjusted analogues (kNN mean + (g(x) - mean g(neighbours)), g = c_r4a mean): lambda 0.5/1/1.5
     0.708/0.709/0.708; single-component g 0.700-0.711.
2. Blend partner (0.5 / 0.5 field-score averages of c_r4b scored map with each scored method): c_r1a 0.701, c_r1b
   0.692, c_r2a 0.710, c_r2b 0.702, c_r3a 0.704, c_r3b 0.708, **c_r4a 0.711 (kept)**; c_r4b + c_r4a weights
   0.25/0.5/0.75: 0.707/0.711/0.708. Replica median analogue + c_r4a: w 0.5 0.713 (p 0.962), w 0.6 0.714, w 0.8 0.708;
   three-way with c_r2a / c_r3b etc. 0.698-0.713. Kept w = 0.5.
3. One run of `competition.score dev c_r5a` (below); no change after it.

## Dev results (leave one year out, competition.score)
```
c_r5a dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.712 [0.675, 0.748] | M2 within-year C 0.709 [0.666, 0.748] | M4 AUC 0.743 [0.675, 0.806]
  M3 median none 0.462 mild 0.475 medium 0.498 heavy 0.538 | JT z 9.15 p 0.0000
  M5 MAE paid 0.135 | none >= 0.35 1.000 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:0.500(n4) 2017:0.884(n20) 2018:0.188(n7) 2019:0.734(n79) 2020:0.658(n55) 2021:0.702(n86) 2022:0.836(n12) 2023:0.500(n4) 2024:0.778(n8)
compare c_r5a vs c_r4b dev: M1 0.712 vs 0.705, diff +0.007 [-0.002, 0.017], p(a > b) 0.941
compare (info) c_r5a vs c_r4a dev: diff +0.018, p 0.9685
```
Dev run 215 s (8 workers, machine shared with another agent).

## Assumes / fixed
As c_r4b and c_r4a (grids, usable share, min clear dates, floors, targets).
