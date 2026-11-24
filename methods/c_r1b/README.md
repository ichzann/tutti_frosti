# c_r1b: crop-calendar position (pixel season centre from the red-edge chlorophyll curve)

## Idea
Each outline pixel gets its season centre (day of year): the midpoint between its green-up and senescence
half-amplitude crossings of the CIre curve (B07 / B05 - 1, clear values on usable dates). A pixel whose season
sits earlier in the calendar gets a higher loss value, through a decreasing sigmoid learned from labels.
No weather, no frost nights, no event date; year is used only to turn dates into day of year.

## Pixel map
- per pixel: peak = season maximum of CIre; green-up = interpolated crossing of lo_before + 0.5 (peak - lo_before)
  before the peak; senescence = crossing of lo_after + 0.5 (peak - lo_after) after the peak; centre = mean of both.
- loss(pixel) = hi / (1 + exp((centre - m) / s)); NaN where a crossing is missing or < 3 clear dates.
- field score (scorer) = mean of the map.

## Learned from labels
(m, s, hi) by Nelder-Mead on development fields: field mean of the pixel map vs label, censored squared error
(paid with loss: (score - loss)^2; paid without loss value: min(score - 0.35, 0)^2; no-claim: max(score - 0.35, 0)^2).
Fit on all 275 dev fields: m 241.6, s 30.5, hi 0.671.

## Settings chosen on development (not label-fitted)
Field-level screen (field-median series, censored C on all dev, no fit): season centre at 0.5 amplitude,
C / within-year C: CIre 0.643 / 0.705, NDRE 0.637 / 0.688, NDMI 0.637 / 0.688, EVI2 0.616 / 0.669.
Peak day of year of field-median EVI2: 0.613 / 0.635; senescence 0.5 crossing EVI2: 0.617 / 0.663.

LOYO ablations (own script, same metric code; rows marked * used the start curve m 230, s 15, hi 0.8 instead of the
fit, because of a fit bug fixed afterwards; ranking-only comparison):

| variant | M1 C | M2 |
|---|---|---|
| CIre, 3x3 smoothing, min_obs 5 * | 0.656 | 0.711 |
| CIre, no smoothing, min_obs 5 * | 0.656 | 0.711 |
| EVI2 * | 0.640 | 0.690 |
| NDRE * | 0.653 | 0.698 |
| CIre frac 0.3 (fitted) | 0.648 | 0.702 |
| CIre frac 0.7 (fitted) | 0.640 | 0.675 |
| CIre, no smoothing, min_obs 3 / 5 / 8 (fitted) | 0.647 / 0.647 / 0.646 | 0.699 / 0.699 / 0.699 |

Smoothing removed (no M1 difference). min_obs = 3 is the structural minimum.

## Assumptions
Season fits inside 15 Mar - 15 Nov with one CIre peak; the half-amplitude crossings exist on both sides of the peak.

## Results (dev, leave-one-year-out, competition.score)
```
c_r1b dev: n 275 (paid 209, none 66), errors 0, no-value fields 0
  M1 C 0.647 [0.605, 0.692] | M2 within-year C 0.699 [0.653, 0.746] | M4 AUC 0.651 [0.563, 0.735]
  M3 median none 0.492 mild 0.522 medium 0.546 heavy 0.580 | JT z 6.30 p 0.0000
  M5 MAE paid 0.128 | none >= 0.35 0.803 | M6 dmg share none 1.000 mild 1.000 medium 1.000 heavy 1.000
  per year C 2016:1.000(n4) 2017:0.429(n20) 2018:0.312(n7) 2019:0.661(n79) 2020:0.673(n55) 2021:0.755(n86) 2022:0.673(n12) 2023:0.000(n4) 2024:0.278(n8)
```
compare c_r1b vs low_peak_evi2 dev: M1 0.647 vs 0.480, diff 0.167 [0.109, 0.224], p(a > b) 1.000.

`_cache/` holds per-field centre maps (imagery only, no labels); delete to recompute.
