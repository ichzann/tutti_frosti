# c_mt_core (max-theory test, evidence alone; outside the champion loop)
Pixel loss = clip(b0 + b1 E_p + b2 X~_k, 0, 1). E_p = standardised ln(1 + D_p / kappa), D_p = sum over the pixel's
canopy-present intervals of omega_bar (leave-self-out clock) x S(stage) x max(0, -z_p - 1) x persistence (Module C,
methods/c_mt/evidence.py). X_k = stage sensitivity at the clock's frost times. kappa, mean / sd from training fields
(label-free). **Clock: dev-split imagery only, user decision 2026-11-09.**
Learned from labels: b0, b1, b2 (least squares of field means on targets loss / 0.85 / 0.15; negative weight -> 0,
refit). Per fold (held-out year: b0, b1, b2): 2016 0.498 0.075 0; 2017 0.491 0.081 0; 2018 0.499 0.076 0;
2019 0.482 0.031 0; 2020 0.510 0.095 0; 2021 0.494 0.075 0; 2022 0.498 0.076 0; 2023 0.496 0.079 0; 2024 0.497 0.075 0.
One scorer run.

## Dev (log methods/c_mt/report/score_c_mt_core.log)
```
M1 C 0.585 [0.539, 0.627] | M2 within-year C 0.612 [0.554, 0.661] | M4 AUC 0.624 [0.545, 0.698]
M3 median none 0.451 mild 0.480 medium 0.494 heavy 0.504 | JT z 3.63 p 0.0001
M5 MAE paid 0.153 | none >= 0.35 1.000
per year C 2016:0.250(n4) 2017:0.607(n20) 2018:0.438(n7) 2019:0.691(n79) 2020:0.408(n55) 2021:0.624(n86) 2022:0.600(n12) 2023:0.000(n4) 2024:0.833(n8)
compare vs c_r6a: diff -0.127 [-0.181, -0.075], p 0.000
```
