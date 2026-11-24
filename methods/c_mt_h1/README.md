# c_mt_h1 (max-theory test, H1; outside the champion loop)
c_r6a (0.7 DTW season analogues + 0.3 c_r4a ridge map) unchanged except the time axis: DOY' = DOY - delta_y in the
analogue curves and the ridge EVI2 / ln NIR grids. delta_y from the Module A clock (methods/c_mt/README.md).
**Clock: dev-split imagery only, user decision 2026-11-09.** delta used: 2016 0, 2017 +9, 2018 0, 2019 +4, 2020 +24,
2021 -2, 2022 +6, 2023 0, 2024 0. Learned from labels: as c_r6a. One scorer run.

## Dev (competition.score dev c_mt_h1; log methods/c_mt/report/score_c_mt_h1.log)
```
M1 C 0.694 [0.654, 0.729] | M2 within-year C 0.691 [0.647, 0.732] | M4 AUC 0.720 [0.649, 0.785]
M3 median none 0.461 mild 0.466 medium 0.498 heavy 0.539 | JT z 8.39
M5 MAE paid 0.130 | none >= 0.35 1.000
per year C 2016:0.500(n4) 2017:0.893(n20) 2018:0.188(n7) 2019:0.731(n79) 2020:0.606(n55) 2021:0.684(n86) 2022:0.836(n12) 2023:0.500(n4) 2024:0.722(n8)
compare vs c_r6a: diff -0.018 [-0.029, -0.007], p(a > b) 0.0005
compare vs c_mt_perm: diff -0.003 [-0.015, 0.008], p 0.280
```
