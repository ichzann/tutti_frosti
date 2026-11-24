# c_mt_perm (max-theory test, falsification F6; outside the champion loop)
As c_mt_h1, with the per-year delta_y permuted across years (numpy default_rng(20261109).permutation over the 9
clock years). **Clock: dev-split imagery only, user decision 2026-11-09.** delta used: 2016 +24, 2017 0, 2018 0,
2019 +9, 2020 +4, 2021 -2, 2022 0, 2023 +6, 2024 0. One scorer run.

## Dev (log methods/c_mt/report/score_c_mt_perm.log)
```
M1 C 0.697 [0.659, 0.733] | M2 within-year C 0.699 [0.654, 0.740] | M4 AUC 0.724 [0.654, 0.790]
M3 median none 0.457 mild 0.474 medium 0.495 heavy 0.548 | JT z 8.41
M5 MAE paid 0.134 | none >= 0.35 1.000
per year C 2016:0.000(n4) 2017:0.884(n20) 2018:0.312(n7) 2019:0.725(n79) 2020:0.629(n55) 2021:0.699(n86) 2022:0.818(n12) 2023:0.750(n4) 2024:0.667(n8)
compare vs c_r6a: diff -0.014 [-0.025, -0.004], p 0.0015
```
