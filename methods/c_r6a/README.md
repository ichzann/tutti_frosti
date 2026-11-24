# c_r6a: c_r4b season analogues (field-median references) 0.7 + c_r4a ridge map 0.3 (round 6 improver)

## Idea
Starts from champion c_r4b (nearest-neighbour season analogues: each outline pixel's 10-band log reflectance
trajectory, doy 90..322 / 8 d, matched by multivariate DTW, band +-6 steps, to every training field's reference
trajectory; pixel analogue loss = mean target of the k = 40 nearest reference fields). Code taken from c_r5a:
1. Reference trajectory = log of the field-**median** reflectance per usable date (c_r4b: field mean). Per-band scale
   from the field-mean curves (as c_r4b).
2. Pixel loss = 0.7 x analogue map (clipped 0..1) + 0.3 x c_r4a map (mean of four clipped ridge maps: EVI2 calendar
   profile, ln NIR retention, residue spectrum at lowest-EVI2 and lowest-NIR post-peak date; c_r4a code copied
   unchanged to `ridge4.py`, settings `ridge4_settings.yaml`, label-free feature cache `cache/`).
   Where one part has no value the other is used; NaN outside the outline.
Only difference to c_r5a: blend weight 0.3 on the ridge part (c_r5a: 0.5). No weather input, no damage-free set.

## Learned from labels (fit, development folds only)
- Analogue part: no weights; references + targets (paid = loss, paid with unknown loss = 0.85, no claim = 0.15).
- Ridge part: as c_r4a (4 ridges; prof / ret alpha by inner leave-one-year-out C, res / resn alpha 3000).

## Settings chosen by looking at dev scores
Offline replica: sampled outline pixels per dev field (24-150), DTW to the dev field references, leave-one-year-out
by reference year, scorer's C and paired bootstrap (p = share of resamples above scored c_r4b). Replica of c_r4b
(150 px) C 0.7049 (scored 0.7050).
1. Structural screens (none kept), C / p vs c_r4b:
   - window context (outside-outline pixels, 3 px gap; all / 30 % / 10 % most similar to the field curve):
     context-only kNN 0.574-0.643; joint pixel + context DTW distance (beta 0.25 / 0.5 / 1) 0.657-0.696;
     field curve minus context or concatenated, field-level kNN 0.551-0.686.
   - band transforms: PCA whitening (3 / 5 / 10 comp.) 0.690-0.697; 2 PCs 0.701; per time x band scale 0.658;
     per-date band-mean removed (spectral shape) + brightness x 0 / 0.25 / 0.5 / 1 / 2: 0.673-0.699.
   - similarity-space ridge on mean pixel DTW kernels (bandwidth 0.1-1 x median distance, lambda 0.01-10):
     0.547-0.687; averaged with kNN 0.653-0.702.
   - pixel-to-pixel analogues (24 reference pixels per training field), k 100-3000: 0.676-0.706 (k 1000: 0.706 vs
     0.700 for the field-reference replica on the same 24 px); with multiple-instance label refinement
     (1 / 3 iterations, gain 1 / 2) 0.694-0.697; pixel-to-field set distance (quantile 0 / 0.1 / 0.25 / 0.5 of
     pixel distances, k 20 / 40 / 80) 0.683-0.703.
   - hurdle (share paid among kc nearest x mean loss among kl nearest paid; kc, kl 20 / 40 / 80) 0.686-0.706.
   - phenology-weighted DTW cost (pre-peak weight 0 / 0.3, peak = reference NIR / EVI2 max, offset -4 / 0 / 3;
     post-peak x 2) 0.663-0.709 (60 px; replica 0.706).
   - observation-gap-weighted DTW cost (tau 4 / 8 / 16 d, floor 0.1 / 0.3) 0.676-0.698.
   - random 5-band subspace ensemble (24 members, W 4 / 6 / 8): members 0.663-0.709, ensemble 0.705.
   - 4 d grid doy 76-328 / 88-328 (W 12) 0.689 / 0.701; 8 d grid from doy 76 0.686.
   - k-means zones (2-32 per field) as queries 0.703-0.707; interior-only pixels (edge distance >= 2..6 px)
     0.704-0.706; year-stratified neighbours (top 3 / 5 / 10 per training year) 0.667-0.693.
   - field-level label propagation (graph k 10 / 20 / 40, alpha 0.5-0.95) 0.660-0.686; early-season analogue
     counterfactual deficit (split doy 180-240) 0.49-0.64 alone, 0.628-0.686 with kNN; kNN in a PLS subspace
     (2-8 comp.) 0.614-0.637, with DTW kNN 0.664-0.674.
2. Blends (field-score level; c_r4a = its scored dev scores): median references alone 0.7084 (p 0.897); median +
   pixel-to-pixel 0.7086; median refs (1-w) + c_r4a w: w 0.3 0.7112 (p 0.972), 0.4 0.7118 (p 0.956), 0.5 0.7124
   (p 0.946); with pixel-to-pixel part 0.7105-0.7130 (p 0.919-0.954); three-way with c_r2a / c_r3b / c_r6b /
   c_r1b 0.7055-0.7131; earlier scored methods 0.5 / 0.5 with c_r4b 0.684-0.711. **Kept: median refs, w 0.3.**
3. One run of `competition.score dev c_r6a` (below); no change after it.

## Dev results (leave one year out, competition.score)
```
c_r6a dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.711 [0.675, 0.747] | M2 within-year C 0.705 [0.662, 0.746] | M4 AUC 0.742 [0.677, 0.805]
  M3 median none 0.453 mild 0.470 medium 0.495 heavy 0.543 | JT z 9.07 p 0.0000
  M5 MAE paid 0.133 | none >= 0.35 1.000 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:0.500(n4) 2017:0.875(n20) 2018:0.312(n7) 2019:0.736(n79) 2020:0.650(n55) 2021:0.694(n86) 2022:0.836(n12) 2023:0.750(n4) 2024:0.722(n8)
compare c_r6a vs c_r4b dev: M1 0.7114 vs 0.7050, diff +0.006 [0.000, 0.013], p(a > b) 0.983
compare (info) c_r6a vs c_r5a dev: diff -0.001, p 0.372
```
Dev run 341 s (8 workers).

## Assumes / fixed
As c_r4b and c_r4a (grids, usable share, min clear dates, floors, targets).
