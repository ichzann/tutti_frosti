# Second lockbox (batch 4) — pre-registration

Written 2026-11-11, before any method has been run on a batch-4 field. Committed to git before scoring; the commit
time is the proof. User request: "make sure that the lockbox is actually a lockbox" (no overfitting, no leaks).

## Why a second lockbox

The first lockbox (116 fields, scored 2026-11-08) has been seen: its numbers were shown, round 7 and the max-theory
test ran after it. Batch 4 is new data, downloaded 2026-11-09/10 after all methods were frozen
(validation/brazil_psr/field_full.py run_b4, same automatic outline rules as the competition set).

## Fields

- 99 fields, selection_b4.parquet (wids F5284-F6783 range): heavy 22, medium 20, mild 27, none 30;
  years 2017 2, 2018 1, 2019 32, 2020 32, 2021 29, 2022 3.
- All 99 are lockbox. No field of batch 4 is used for anything but this one scoring.
- Package: competition_b4/ (COMP_DIR=competition_b4). Training split = the same 275 dev fields with the same ids
  (C0001-C0391 range) and the same labels as the competition; batch-4 fields get ids C2001-C2099 (random order),
  so no method cache keyed by field id can collide. Old lockbox fields are not in the package.
- Preprocessing frozen: same S2 10-band download, same outline rule, QA clear-sky line fitted on the full
  competition set's rows (not on batch 4).

## Methods (fixed now, chosen by dev M1 only, before any batch-4 result)

Top 5 distinct methods by dev leave-one-year-out M1 (c_r5a left out as near-copy of c_r6a, as for lockbox 1):

| method | dev M1 | seen a lockbox before |
|---|---|---|
| c_r7a | 0.714 | no |
| c_mt_full | 0.712 | no |
| c_r6a | 0.711 | yes (lockbox 1) |
| c_r7b | 0.707 | no |
| c_r4b | 0.705 | yes (lockbox 1) |

Baselines: low_peak_evi2, ndmi_drop; weather reference era5_coldest.

frost_recovery family (user's method; self-trained without labels, per year store):
- frost_recovery_v2 (whole season, field number = share_all). Output has no weather input; its training and
  calibration use ERA5 to pick frost-free intervals. Reported in the table, marked as such.
- frost_recovery (original, field number = largest damaged share over the ERA5 event nights): weather-based,
  reported next to era5_coldest, not ranked with the satellite-only methods.
- Frozen: model and calibration trained on the full competition stores (wheat<year>full, imagery only, no
  labels), copied unchanged to the batch-4 stores. Batch-4 imagery is not used for their training.
  Batch-4 years without a full-store model (training failure) are reported as missing, not re-tuned.

## Rules

1. No code, setting, threshold or seed of any listed method changes after this file is committed.
2. ML methods: fit on all 275 dev fields and dev labels, predict the 99 batch-4 fields once
   (python -m competition.score lockbox <m> with COMP_DIR=competition_b4; ledger competition_b4/_harness/lockbox_runs.json
   refuses a second run).
3. Labels of batch 4 are read only by the scorer; nobody looks at per-field scores before all methods have run.
4. A crashed run is fixed only if the fix does not touch method logic (paths, memory), and the fix is logged.
5. No method, method choice or threshold is changed after the results; any later method is a new name and needs
   a third lockbox.

## Metrics (same scorer, competition/score.py)

M1 censored concordance C (main), M2 within-year C, M3 group medians + JT z, M4 AUC paid vs none, M5, M6,
per-year C. Paired bootstrap of M1 differences (2000 resamples, seed 7).

Overfitting / leak checks, all fixed now:
- Generalisation gap per method: dev LOYO M1 vs lockbox-1 M1 vs lockbox-2 M1.
- Label-permutation null on batch 4: 2000 permutations of loss/claim within year; share of permuted M1 >= observed.
- Duplicate check: no shared PSR proposal id / point with the package (done 2026-11-11: 0 shared ids, 0 shared
  points); outline pixel overlap batch-4 vs dev fields (any year); sensitivity: M1 without batch-4 fields whose
  point is within 1 km of a dev field (4 fields) or whose outline overlaps a dev outline.
- Group mix differs from dev (batch 4 drawn damage-first, none/mild near frost-paid policies): reported, not
  corrected.
