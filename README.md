# Frost damage map — Phase 1 baseline

Per-pixel radiational frost damage maps for potato fields at Sentinel-2 resolution,
for one county (Bingham County, ID) and one season (2024).
Step 0 history: Fremont/Teton 2022 dropped (all trigger nights outside the growth phase);
2024 chosen for the widespread Jun 18-19 frost, Bingham over Jefferson/Bonneville.

## Run

```
conda env create -f environment.yml && conda activate frostmap
python run.py                       # prepare inputs (cached) -> method -> runs/<id>/ -> report.html
python run.py --run-id test --fields 5   # quick debug run on 5 fields
python -m report.build runs/<id>    # rebuild only the report from existing run outputs
```

Open `runs/<id>/report.html` directly in a browser (no server; Leaflet is loaded from unpkg).

## Layout

| path | role |
|---|---|
| `config.yaml` | every threshold and setting (initial proposals, not tuned in Phase 1) |
| `data/` | download + preprocessing into `cache/`; `data/store.py` = read-only access for methods |
| `methods/<name>/` | damage methods (plugin); `methods/evi2_gpr/` = baseline |
| `validation/` | method-independent test harness (Tier 1), `config.yaml` = fixed test settings and pass criteria |
| `report/` | `build.py` + `template.html`: turns run outputs into `report.html` |
| `run.py` | end-to-end command |
| `runs/<id>/` | outputs of one run |

## Data sources

| data | source | kept locally |
|---|---|---|
| Weather | NOAA RTMA 2.5 km hourly (TMP, DPT, WIND, GUST, TCDC), byte-range reads of the needed GRIB messages from the NOAA AWS bucket, stream-and-discard | `cache/nights.parquet` (cell × night), `cache/rtma_cells.gpkg`, `cache/rtma_hours/` (county-window subsets per hour, resume cache only) |
| Vegetation | Sentinel-2 L2A, Microsoft Planetary Computer; SCL 4/5 = clear, everything else masked; windows around field clusters only | `cache/s2/<field_id>.npz` (EVI2, RGB quicklook per date), `cache/s2_qa_<fips>.parquet`, `cache/s2_aot_<fips>.parquet` |
| Vegetation, all bands | same items / windows / SCL rule; B02–B07, B08, B8A, B11, B12 (20 m bands nearest-resampled to 10 m), `python -m data.s2bands` (resumes per batch) | `cache/s2bands/<field_id>.npz` (refl × 10000, aligned with `cache/s2` dates) |
| Crop | USDA NASS CDL: class from CropScape (by FIPS, 30 m). Confidence: 30 m years from the national zip (GeoTIFF entry streamed, clipped, deleted); 2024 exists only as a 10 m national layer (68 GB zip), which is stream-inflated twice without being stored (pass 1 reads the TIFF directory at the end, pass 2 re-inflates only the county's span from an in-memory checkpoint) | `cache/cdl/` |
| Fields | USDA NASS Crop Sequence Boundaries 2018–2025 (national archive downloaded once, candidate counties cut out, archive deleted) | `cache/csb_candidates.gpkg` |
| Terrain | USGS 3DEP 10 m (Planetary Computer `3dep-seamless`) | `cache/dem/` |

Field pixels = buffered (10 m inward) CSB potato polygon ≥ 4 ha AND CDL potato AND the confidence rule
(`cdl_min_confidence`, brief: 75 %; 40 % for 2024, whose random-forest confidence has no Bingham potato pixel ≥ 75),
on a 10 m grid aligned with Sentinel-2 (UTM 12N). `cdl_confidence_rule`:

- `pixel` (brief): each pixel needs confidence ≥ threshold. With the speckled 2024 10 m confidence this
  breaks fields into pieces (Bingham: median 3, max 47 connected pieces per field; 75 % of the polygon kept).
- `field` (current): a field is kept when the median confidence of its CDL potato pixels is ≥ threshold, and
  then all of its CDL potato pixels are used (Bingham: 561 of 728 fields, median 1 piece, 94 % of the polygon).

Acquisition QA (`data/s2qa.py`, `config.yaml: s2_qa`): a field-date is used whole or not at all.
It is dropped when < 95 % of the field's pixels are SCL-clear (`cloud`); on closed canopy (field-median
EVI2 ≥ 0.5) when the Haze Optimized Transform (HOT, blue lifted above the county clear-sky blue/red line)
exceeds the field's own clear-date median by > 0.01 (`haze`); on sparser cover, where soil makes HOT
unreliable, when the Sen2Cor AOT > 0.3 (`aot`). Haze lowers EVI2 like damage does (2024: HOT anomaly
> 0.02 → median EVI2 −0.13 vs neighbouring dates), but lifts blue rather than red, so HOT tells them apart.
AOT alone does not drop canopy dates: L2A is already corrected for it (2024-07-23, AOT 0.45 smoke, clean).
Dropped dates are hidden by `data/store.py` (EVI2 set to NaN for the whole field) and marked in the report.

The report's NDMI quicklooks (`data/ndmi.py`, (B8A − B11)/(B8A + B11), 20 m) are fetched after the run only
for the dates it displays; they are not a method input.

`python -m data.step0` prints the step 0 county comparison (potato area, field terrain, trigger-night counts).

## Method interface

A method is a package `methods/<name>/` exposing `NAME`, `REASONS` (valid-mask reason codes) and

```python
m = Method(cfg, store)            # store = data.store.FieldStore(cfg)
r = m.damage(field_id, night)     # any field, any night (night = evening date, 'YYYY-MM-DD')
r["score"]   # (H, W) float, damage score, NaN where not valid
r["valid"]   # (H, W) bool
r["meta"]    # pre_date, post_date, next_date, gap_days_before, gap_days_after, growth_stage,
             # emergence, peak, dose, tmin_c, trigger, near_trigger, cell_id
r["curve"]   # optional field-level prediction band for the report (or None)
r["layers"]  # method-specific rasters (evi2_gpr: z, predicted, pred_std, observed, reason, ...)
```

`Method.score_threshold`: a pixel is labelled damaged if `score >= score_threshold` (evi2_gpr: 2, i.e. z ≤ −2).

Run with another method: `python run.py --method <name>`. Methods never download; they only read the store.

### Baseline `evi2_gpr`

1. Night flags per RTMA cell (`data/rtma.py`, parameters and sources in `config.yaml: night`).
   Regime from the 3 hours up to the night's Tmin: radiative if mean 10 m wind ≤ 2.7 m/s and
   cloud ≤ 30 % both in the cell and as the mean of its 3×3 neighbourhood (~7.5 km: a clear 2.5 km hole
   under a cloud deck still gets back-radiation from it), else advective. The evening cooling rate
   (first 3 h after sunset) is reported as an independent check of the cloud input. Trigger if RTMA hourly 2 m Tmin ≤ threshold of the regime:
   advective 0 °C = leaf injury −2 °C + 1 °C RTMA/terrain margin + 1 °C hourly-sampling miss
   (p90, RTMA-RU 15-min check); radiative +2 °C (leaf ~2 °C colder than 2 m air on clear calm
   nights). Near-trigger (control): radiative and 6 ≤ Tmin ≤ 12 °C. Windy / cloudy nights are no
   longer excluded (Phase 1 rule: Tmin ≤ 1 °C and calm, clear, dry as a hard gate).
2. Growth phase per field: emergence = first date the smoothed field-median EVI2 > 0.20
   (smoothing: centred running median over 3 clear observations); end = date of the smoothed peak.
3. Per pixel: GPR (constant × RBF + white noise, hyperparameters by marginal likelihood) on all clear
   observations before the night, predicted at the first clear observation after it. Valid only with
   ≥ 4 clear obs (bare soil included) in the 45 d before, ≥ 1 of them since emergence, last ≤ 20 d before,
   first after ≤ 10 d after. (Brief: ≥ 4 since emergence; relaxed because the 5-day revisit never gives
   4 post-emergence obs by the Jun 18 2024 frost.)
   Fits are cached per (pixel, last obs before, first obs after): nights between the same two
   acquisitions give identical results.
   Variants (`config.yaml: gpr.hyper`, `gpr.mean`; the description above = `pixel` + `constant`):
   - `hyper: field`: one hyperparameter set (amplitude, length scale, noise on raw EVI2, pixel series
     centred on their own mean) per field and set of acquisitions before the night, by the joint marginal
     likelihood of a random sample of `hyper_sample_px` pixels; then exact GP conditioning per pixel.
   - `mean: reference`: the GP gives the pixel's smoothed EVI2 at its last obs before the night (s, var);
     predicted = s + Δ, variance = var + var(Δ) + noise, where Δ = mean EVI2 change of the
     `reference.k_neighbors` nearest frost-free intervals (field-median EVI2 of other fields; features:
     EVI2 before = s, field slope before, gap, day of year). Intervals come from non-lockbox fields,
     with no trigger night and every night's RTMA Tmin ≥ `reference.min_tmin_c` (6 °C) in the field's
     cell; a field is predicted only from other field folds.
4. z = (observed − predicted) / predictive std; score = max(0, −z); damaged if z ≤ −2.
5. Dose = degree-hours of the estimated leaf temperature (2 m TMP − 2 °C on radiative nights, TMP on
   advective) below −1 °C, from the field's RTMA cell (cell containing the field's representative point).

### Prototype `frost_evidence`

Satellite-first: the score uses only Sentinel-2 EVI2; air temperature and terrain never gate it and only
label how plausible frost is as the cause. Settings in `config.yaml: frost_evidence`.

1. Expected EVI2 without frost = `evi2_gpr` with `gpr.hyper: field`, `gpr.mean: reference` (forced).
2. Persistence: residuals at up to 2 clear obs after the night within 20 d, combined by their GLS mean
   (both share the error of the level before the night).
3. Spatial support: z averaged over a 3 × 3 window of valid pixels.
4. Empirical calibration: z = Φ⁻¹ of the statistic's rank among frost-free pixel-intervals of 150 other
   non-lockbox fields (leave-fold-out; every night of the 20 d window ≥ `reference.min_tmin_c`), per stage
   and number of post obs. Built once per config, cached as `cache/frost_evidence_cal_<fips>_<hash>.parquet`.
   Damaged if z ≤ −2.
5. Attribution (layers / meta only): `p_frost` = P(leaf ≤ −2 °C) from RTMA Tmin − 0.3 °C (median hourly
   miss) − on radiative nights 2 °C leaf offset and 0.1 °C per m below the 300 m mean elevation (cold-air
   pooling, max 4 °C), error sd 2 °C; `terrain_rho` = rank correlation of z with that pooling index;
   `attribution` = no damage / damage, frost plausible (max `p_frost` ≥ 0.1) / damage, frost implausible.
   Nights between the same two acquisitions get the same damage evidence; only the attribution differs.

### Prototype `frost_signature`

Multi-index and recovery-shaped; weather and terrain only for attribution (as `frost_evidence`).
Settings in `config.yaml: frost_signature`; needs `cache/s2bands`.

1. Indices EVI2 (canopy), NDMI (B8A/B11, leaf water), NDRE (B8A/B05, chlorophyll), PSRI ((B04−B02)/B06,
   browning). Each: expected value at the post obs = field-hyperparameter GP level at the last clear obs before
   + a `GrowthReference` change built on that index; standardised residuals u.
2. Frost fingerprint in index space per pixel: linear mixing of the pixel's green fraction into killed material
   (75 % dead haulm, 25 % soil; method-owned endmembers from non-lockbox fields). Haze fingerprint: median band
   change of QA haze-dropped field-dates vs. neighbouring usable dates.
3. Per post obs (up to 3 within 30 d): matched filter of the frost fingerprint in the residual space whitened
   by the indices' residual correlation per stage, haze direction projected out (orthogonal subspace projection).
4. Recovery: loss A·0.5^(t/h), h ∈ {7, 14, 28} d; statistic = max over h. Layers `loss`, `half_days`.
5. 3 × 3 spatial mean, empirical leave-fold-out calibration and attribution as `frost_evidence`.
   Caches `cache/frost_signature_{spectra,sigma,cal}_<fips>_<hash>.*`.

## Run outputs (`runs/<id>/`)

| file | content |
|---|---|
| `nights.parquet` | RTMA cell × night for the county, with flags and dose |
| `field_nights.parquet` | field × night for every night in the field's growth phase and every flagged night (trigger/near-trigger in any potato cell): mean/median z, share damaged, valid share, dates, gaps, stage, flags |
| `rasters/<field>/<night>.npz` | flagged nights in growth phase: score, valid, reason, z, observed, predicted, pred_std, RGB/EVI2 before/after/next |
| `img/<field>/` | pre-rendered PNGs for the report |
| `fields.geojson`, `rtma_cells.geojson`, `hillshade.png`, `fields_detail.json`, `s2_acquisitions.parquet` | report inputs |
| `run_meta.json`, `config_used.yaml` | git commit, run info, the config used |
| `report.html` | the viewer |

## Validation (`validation/`)

There is no ground truth, so methods are tested against what they must do if they work.
Tier 1 (no ground truth needed), identical for every method:

```
python -m validation.run --method <name>                  # dev split, 150 fields (evi2_gpr: hours with
                                                          # gpr.hyper pixel, ~30 s with field)
python -m validation.run --method <name> --split holdout  # lockbox: final comparison only
python -m validation.run --rescore validation/results/<id>
```

- **Cases**: harness-defined (not method-defined) field-intervals between consecutive usable acquisitions
  inside the field's growth phase, with a random night in the interval and no trigger night in it.
  Frost-free is conservative (`domain.null_min_tmin_c`, 6 °C): every night of the interval needs RTMA
  Tmin ≥ 6 °C, because the trigger uses 2.5 km 2 m air temperature and leaves / low spots can be colder.
  The filter is applied after the random draws, so the remaining cases keep their `case_id`; runs made
  before it were rescored on those cases as `<run>_cons6` (`run_meta.json: derived_from`).
  Stage = field-median EVI2 at the acquisition before (early < 0.35 ≤ mid < 0.60 ≤ closed).
- **Null calibration**: on these frost-free cases, share of valid pixels at or above the method's threshold
  (pixel false-positive rate), share of field-cases flagged (≥ 10 % of valid pixels), coverage. If the method
  returns `predicted` / `pred_std` / `observed` layers: z distribution, 95 % interval coverage, PIT, CRPS.
- **Synthetic injection**: each case is re-run with a frost signature added to the real EVI2 series after
  the night (`evi2' = 0.1 + (evi2 − 0.1)(1 − f·w·D(t))`, canopy loss f 0.05–0.5, pattern whole field or the
  field's lowest 35 % by elevation, loss held 7 d then recovering to half by 21 d). The method sees it only
  through a wrapped store. Pixel AUC vs null scores, detection rate at the method's threshold and at the
  null 95th percentile, detection limit in absolute EVI2 drop, field-level detection, localisation,
  and whether the damage makes the method go silent (validity).
- **Spectral mode** (`injection.mode: spectral`, from runs `*_spec` / `frost_signature_dev_v1`): the damage
  is injected into all 10 bands by linear mixing, `refl' = refl + loss·g·(E_k − E_green)`, E_k = dead_mix·E_dead
  + (1 − dead_mix)·E_soil (harness-owned endmembers, `cache/validation_endmembers_<fips>.json`), loss held 3 d
  then halving every 7, 14 or 28 d (one per case); EVI2 recomputed from the bands. A third run per case adds
  haze only (blue +0.01/0.02/0.03, Ångström 1.5) to the first post obs; criterion N5 = haze field-cases flagged.
  Spectral draws use a separate random stream, so cases stay identical to earlier runs.
- **Rules**: settings and criteria in `validation/config.yaml` are fixed across methods and change only by
  user decision; criteria are provisional until approved. 30 % of fields (by hash of `field_id`) are the
  lockbox, never used while developing a method. The injection signature is known, so a method can be
  tuned to it: injection results show sensitivity to this signature, not to real frost.
- **Outputs** (`validation/results/<id>/`): `scorecard.json` (all metrics, 95 % cluster-bootstrap CIs over
  fields, criteria), `cases.parquet`, `case_results.parquet`, `pixels.parquet`;
  one row per run in `validation/results/leaderboard.csv`.
