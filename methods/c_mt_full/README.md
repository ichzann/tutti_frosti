# c_mt_full (max-theory candidate vs c_r6a; outside the champion loop)
L_p = sigmoid(logit(clip(A_p, 0.02, 0.98)) + beta1 E_p + beta2 X~_k); A_p = Module B prior on the **calendar** axis
(rule R1: c_mt_h1 dev M1 0.694 < 0.7114), i.e. c_r6a recomputed from the copied code; E_p, X~_k as c_mt_core.
Where E_p is NaN, L_p = A_p. **Clock: dev-split imagery only, user decision 2026-11-09.**
Learned from labels: prior as c_r6a; beta by grid (beta1 {0,.1,.2,.3,.5,.75,1} x beta2 {0,.1,.2,.3,.5}) on the
censored C of the field-level out-of-fold prior (analogue: own-year references dropped; ridge: R4._loyo), smallest
beta1 + beta2 within 0.002 of the best (R2, no override). Fallback not used.
Per fold (held-out year: beta1, beta2, OOF C at chosen beta / at beta 0): 2016 0.1 0 0.7109 / 0.7019; 2017 0.1 0
0.6990 / 0.6855; 2018 0.1 0 0.7135 / 0.7062; 2019 0.1 0 0.6955 / 0.6851; 2020 0.1 0 0.7232 / 0.7006; 2021 0.1 0
0.7097 / 0.7008; 2022 0.1 0 0.7122 / 0.6990; 2023 0.1 0 0.7120 / 0.7023; 2024 0.1 0 0.7063 / 0.6992
(methods/c_mt/report/fit_c_mt_full.jsonl). One scorer run.

## Dev (log methods/c_mt/report/score_c_mt_full.log)
```
M1 C 0.712 [0.675, 0.745] | M2 within-year C 0.715 [0.670, 0.755] | M4 AUC 0.759 [0.696, 0.817]
M3 median none 0.443 mild 0.468 medium 0.499 heavy 0.541 | JT z 9.08
M5 MAE paid 0.133 | none >= 0.35 0.985
per year C 2016:0.250(n4) 2017:0.946(n20) 2018:0.312(n7) 2019:0.754(n79) 2020:0.606(n55) 2021:0.717(n86) 2022:0.800(n12) 2023:0.500(n4) 2024:0.778(n8)
compare vs c_r6a: diff +0.0006 [-0.016, +0.017], p(a > b) 0.4985
```
