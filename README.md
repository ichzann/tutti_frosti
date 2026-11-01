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
| `report/` | `build.py` + `template.html`: turns run outputs into `report.html` |
| `run.py` | end-to-end command |
| `runs/<id>/` | outputs of one run |

## Data sources

| data | source | kept locally |
|---|---|---|
| Weather | NOAA RTMA 2.5 km hourly (TMP, DPT, WIND, GUST, TCDC), byte-range reads of the needed GRIB messages from the NOAA AWS bucket, stream-and-discard | `cache/nights.parquet` (cell × night), `cache/rtma_cells.gpkg`, `cache/rtma_hours/` (county-window subsets per hour, resume cache only) |
| Vegetation | Sentinel-2 L2A, Microsoft Planetary Computer; SCL 4/5 = clear, everything else masked; windows around field clusters only | `cache/s2/<field_id>.npz` (EVI2, RGB quicklook per date) |
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

Run with another method: `python run.py --method <name>`. Methods never download; they only read the store.

### Baseline `evi2_gpr`

1. Night flags per RTMA cell (`data/rtma.py`, parameters and sources in `config.yaml: night`).
   Regime from the 3 hours up to the night's Tmin: radiative if mean 10 m wind ≤ 2.7 m/s and
   cloud ≤ 30 %, else advective. Trigger if RTMA hourly 2 m Tmin ≤ threshold of the regime:
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
4. z = (observed − predicted) / predictive std; score = max(0, −z); damaged if z ≤ −2.
5. Dose = degree-hours of the estimated leaf temperature (2 m TMP − 2 °C on radiative nights, TMP on
   advective) below −1 °C, from the field's RTMA cell (cell containing the field's representative point).

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
