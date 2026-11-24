# c_r3a: c_r1a EVI2 ridge + multiple-instance sigmoid on ln NIR retention (round 3 improver of c_r1a)

## Idea
Grows from champion c_r1a. Two pixel maps on the same calendar grid (doy 100..324, 16 d, linear interpolation
between each pixel's clear dates on usable scenes with >= 50 % of the outline clear), averaged with equal weight:
- A (c_r1a unchanged): pixel EVI2 profile, pixel loss A = clip(b + w . profile, 0, 1); ridge on field-mean profiles.
- B (new): pixel ln NIR retention curve r(t) = ln B08(t) - pixel max over the grid (signal from c_r2b);
  pixel loss B = sigmoid(v . r + c). v, c are fitted as a multiple-instance model: the field mean of the pixel
  sigmoids (= what the scorer averages) is fitted to the field label by least squares with an L2 penalty on v,
  using all outline pixels of every training field.
Pixel loss = 0.5 A + 0.5 B. No weather, no damage-free set.

## Learned from labels (fit, development folds only)
- A: 15 weights + intercept, ridge alpha from [10..10000] by inner leave-one-year-out C. All-dev fit: alpha 3000.
- B: 15 weights + intercept, lambda from [30, 100, 300] by inner leave-one-year-out C. All-dev fit: lambda 300,
  inner C 0.680; v > 0 for doy 148-196, < 0 for doy 212-308.
- Targets as c_r1a: paid = relative loss, paid with unknown loss = 0.85, no claim = 0.15. Inner C counts paid with
  unknown loss only in (paid, none) pairs (as the scorer; c_r1a ranked them at 0.85 among paid).

## Settings chosen by looking at dev scores (own LOYO script, field-mean features, scorer's C; compare p vs c_r1a)
1. Despiking (running median of 3 over each pixel's clear dates) on EVI2: 0.605 (p 0.000); ln NIR retention 0.674
   (no despike) / 0.674 (despike). Not used.
2. Combination of A with ln NIR retention: ridge on concatenated EVI2 + retention 0.633; average of two ridges
   0.670 (p 0.953); 25 %-quantile pixel curves instead of mean: EVI2 0.658, retention 0.672.
3. Component B as multiple-instance sigmoid (400 pixels/field sample), fixed lambda:
   retention 1 / 10 / 100 / 300 / 1000 / 3000 -> 0.649 / 0.667 / 0.679 / 0.680 / 0.586 / 0.520;
   EVI2 1 / 10 / 100 / 300 / 1000 / 3000 -> 0.620 / 0.636 / 0.645 / 0.648 / 0.557 / 0.515;
   EVI2 + retention 300 / 1000 / 3000 -> 0.670 / 0.652 / 0.535. Lambda grid [30, 100, 300] set after this.
4. With lambda by inner LOYO: MIL retention alone 0.680 (p 0.967), MIL EVI2+retention 0.662, MIL EVI2 0.656;
   averages: ridge EVI2 + MIL retention 0.673 (p 0.992, chosen), MIL EVI2 + MIL retention 0.673 (p 0.986).
5. In the method: 400-pixel random sample per field -> dev M1 0.671, compare p 0.955; then all outline pixels
   (kept) -> results below. Weight 0.5 fixed a priori, not varied.

## Assumes / fixed
Grid, usable share, min clear dates, targets 0.15 / 0.85 and alpha grid from c_r1a; nir floor 0.001 from c_r2b.
`cache/` = label-free per-field features (field-mean EVI2 profile, pixel retention curves), speed only.

## Dev results (leave one year out, competition.score)
```
c_r3a dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.672 [0.634, 0.712] | M2 within-year C 0.676 [0.630, 0.719] | M4 AUC 0.693 [0.622, 0.762]
  M3 median none 0.479 mild 0.480 medium 0.497 heavy 0.544 | JT z 7.44 p 0.0000
  M5 MAE paid 0.131 | none >= 0.35 0.985 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:1.000(n4) 2017:0.750(n20) 2018:0.250(n7) 2019:0.673(n79) 2020:0.626(n55) 2021:0.698(n86) 2022:0.800(n12) 2023:0.250(n4) 2024:0.389(n8)
compare c_r3a vs c_r1a dev: M1 0.672 vs 0.658, diff +0.014 [-0.001, 0.030], p(a > b) 0.9675
```
