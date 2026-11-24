# Method competition: pixel-level crop damage from Sentinel-2, scored against insurance loss

Status: draft (2026-11-07). Fixed before any agent starts; not changed after the first lockbox score.

## 1. Goal

Each agent builds a **method**: an algorithm that turns one field's Sentinel-2 season into a **pixel loss map**.
The agent does not judge damage itself (no looking at images and labelling) and does not rank other agents.
One fixed scoring script runs every method on the same fields and compares the result with the insurance loss.

What we want to show: the method detects damage, and the more harvest a field lost, the more damage the method finds.

## 2. Data

Fields: the automatic outline selection (`cache/brazil_psr/handcheck_wheat/selection.parquet`), southern Brazil
wheat, PSR policies 2016-2024. Outlines pass the automatic shape rules only; no hand verdict decides in or out.
Target size ~400 fields (heavy 110 / medium 90 / mild 100 / none 100). The data set is frozen before the split.

Per field the method gets:

| Input | Content |
|---|---|
| S2 series | all Sentinel-2 L2A dates of the season (15 Mar - 15 Nov), 10 bands, 10 m grid, window = outline bbox + 300 m |
| QA | per date: usable flag, per-pixel clear mask |
| Outline | pixel mask of the field |
| Year | season year (needed for the calendar) |

Not given: weather, frost nights, claim dates, insured yield, municipality, policy id.

Labels (development fields only):

| Label | Meaning |
|---|---|
| `loss` | paid frost claim: relative loss = (1 - coverage) + coverage x indemnity / insured value (always >= 0.35) |
| `none` | no claim: loss is **below ~0.35**, exact value unknown (censored, not zero) |
| `group` | none / mild (< 0.50) / medium (0.50-0.70) / heavy (>= 0.70) |

## 3. Split

- **Lockbox**: 30 % of fields, stratified by year x group, drawn once with a fixed seed by the harness.
  Agents never get its labels. Each method is scored on it **once**, at the end.
- **Development**: the other 70 %. Agents may use the labels here freely.
- Development cross-validation: **leave one year out** (train on the other years, score the held-out year).
  The scoring script reports this; agents use it for their own choices.

## 4. Hard rules

1. Output is a **pixel loss map** per field: one value 0..1 per outline pixel = estimated share of harvest lost
   at that pixel. NaN allowed where no estimate is possible (e.g. never clear).
2. **No weather or frost-night input.** The method finds when and where the canopy changed from the satellite only.
3. **No assumed damage-free training set.** No-claim fields enter only as "loss < 0.35". Any parameter that is not
   learned from the labels must be shown to matter on development labels (or be removed).
4. **No lockbox access**: no reading its labels, no tuning on it, no repeated lockbox runs.
5. **Same method for every field**: no per-field or per-year hand settings. Year may be used only as a calendar.

Everything else is free: indices, physics, mixture models, time-series models, gradient boosting, deep learning,
how growth stage is handled, how pixels are combined.

## 5. What a method delivers

```
methods/<name>/
  method.py       predict(field) -> float32 array (H, W), loss 0..1, NaN outside outline / no estimate
                  fit(dev_fields, dev_labels) -> None   (optional; only development data)
  settings.yaml   every parameter, with where its value comes from
  README.md       the idea in plain words, what it learns from labels, what it assumes
```

`field` = the inputs of section 2 for one field. Runtime limit: the whole lockbox in under 1 hour on this machine.

## 6. Scoring

**Field score = mean of the pixel loss map inside the outline**, computed by the scoring script, not the method.
Methods cannot choose a different field number; damage has to be put into pixels.

Metrics, on the lockbox, each with a 95 % bootstrap interval (2,000 resamples over fields):

| # | Metric | Question |
|---|---|---|
| **M1 (main)** | **Concordance C** with censoring: pairs (paid, paid) ordered by loss; every (paid, none) pair counts paid > none; (none, none) not comparable | Do fields that lost more get higher scores? |
| M2 | Concordance C within year only (pairs from the same year) | Is it real damage, not "bad year"? |
| M3 | Median field score per group none / mild / medium / heavy, and trend test (Jonckheere-Terpstra) | More damage, stronger signal? |
| M4 | AUC paid vs none | Can it separate claims from no-claims? |
| M5 | Calibration: median absolute error of field score vs loss (paid fields); share of none fields with score >= 0.35 | Are the numbers usable as loss, not only as a ranking? |
| M6 | Damaged-pixel share (pixels >= 0.35) per group | Extent of damage, for the figures |

Reported also, not ranked: the same metrics on development leave-one-year-out, per year, and on the subset with
outline area within 5 % of the insured area.

## 7. Baselines

Scored by the same script (their field number is used directly; they have no pixel map):

- NDMI drop: largest drop of the field-median NDMI between consecutive clear dates in the season
- Low peak EVI2: 1 - maximum of the field-median EVI2
- frost_recovery_v2 (whole season, no frost night): field damaged share
- ERA5 coldest night (weather, reference only; breaks rule 2 and is not a competitor)

## 8. Ranking

- Ranked by M1 on the lockbox.
- A method **beats a baseline** only if the paired bootstrap interval of the M1 difference excludes 0.
- Two methods whose paired M1 interval includes 0 are reported as **tied**.
- M2-M5 are shown next to M1; a method with M2 near 0.5 is reported as not detecting within-year damage,
  whatever its M1.

## 9. Not tested here

- **Where** in the field the damage is: the insurance data has no location truth. The maps are checked only for
  the total. A later step uses neighbouring insured fields with different outcomes and hail / damage data sets
  with mapped damage.
- Cause of damage: claims are frost (GEADA); no-claim fields may still carry other damage below the threshold.

## 10. Procedure

1. Harness: freeze the field set, draw the lockbox, build the per-field data package, write the scoring script,
   score the baselines on development and lockbox.
2. Agents: get this file, the development package and the scoring script; each builds one method in its own
   folder. Different starting directions are suggested, not enforced.
3. Agents hand in; the harness runs each method once on the lockbox and publishes the table.
4. Results recorded as numbers only.
