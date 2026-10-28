# Kickoff prompt — Phase 1: baseline damage map + visualization

## What this project is
I am building **per-pixel radiational frost damage maps** for **potato fields in the US**
at Sentinel-2 resolution (10–20 m). Potato is frost-tender, so its canopy acts as a physical
sensor of where radiational frost happened and how strong it was. Long term these maps
become labels for a frost early-warning model, but that is NOT part of this work.

## What this phase is (and is not)
This phase builds exactly one thing: the **baseline damage-mapping pipeline as I designed it**,
plus a **visualization** so I can inspect the results by eye.

- Implement the method below faithfully. Do not improve, tune or replace it.
- Do NOT build validation tests yet (placebo, dose–response, physical consistency).
  They come in a later phase, in a separate test harness. But the code must be structured
  so that harness can call the method (see "Method interface").
- All thresholds below are **initial proposals**. They live in one config file and are
  NOT changed in this phase, not even if the maps "look wrong". They will be revisited
  later against validation results.
- Out of scope: other indices, other crops, other countries, forecasting models.

## Study area and period
- **County: Fremont County, Idaho** (Ashton seed-potato area, ~1,600 m elevation,
  short season with frost possible in summer). Chosen for: large potato area, terrain
  varying from river bottoms to benches and foothills, and frequent radiational frost.
  Fallback: Teton County, Idaho (neighboring, higher basin, used in earlier work).
- **Season: 2022**, one season end-to-end first. Window: 1 May – 30 September;
  damage is only evaluated inside each field's growth phase (see step 2).

**Step 0 — sanity check before building anything.** For Fremont and Teton, 2022, report:
potato area (CSB polygons with potato in 2022), elevation range and relative relief of
those fields from the DEM, and the number of trigger and near-trigger nights from step 1.
Then stop and wait for my confirmation of the county.

## Data
1. **Weather — RTMA**, 2.5 km grid, hourly. Variables used: TMP, DPT, WIND, GUST, TCDC.
   - Process stream-and-discard: never keep the raw GRIB archive.
   - Keep only a small **nights table**: one row per RTMA cell × night with min temperature,
     hours below the frost threshold, degree-hours below the frost threshold, mean wind,
     mean cloud cover, dew point at sunset, and the flags from step 1.
   - Also keep the RTMA cell geometries for the county (needed for the map).
2. **Vegetation — Sentinel-2 L2A.** Red and NIR for EVI2, blue and green for an RGB view.
   A pixel observation is clear only if its scene classification is vegetation or bare soil;
   everything else (cloud, shadow, cirrus, snow, water, no data) is masked.
3. **Crop location — USDA Cropland Data Layer (CDL)**, potato class, 2022.
   Only pixels with CDL confidence **≥ 75 %** are used.
4. **Field boundaries — USDA NASS Crop Sequence Boundaries (CSB)**, the 2018–2025 release.
   A field = a CSB polygon whose 2022 crop attribute is potato. Inside it, keep only
   pixels that also pass the CDL confidence rule. Buffer the polygon 10 m inward (one pixel)
   to drop mixed edge pixels. Drop fields smaller than **4 ha** after buffering.
5. **Terrain — USGS 3DEP DEM** (10 m): elevation, relative elevation within each field
   (pixel elevation minus field median), and a hillshade for the county view.

Before any large download, tell me the expected volume and wait for my OK.

## Method (baseline — implement as described)
**Step 1 — Night flagging (per RTMA cell × night).** Night = local sunset to sunrise.
- `trigger`: all of
  - min 2 m temperature **≤ 1.0 °C** (2 m air is used as a proxy; on radiative nights the
    canopy is typically a few degrees colder than 2 m air),
  - mean wind speed over the night **≤ 2.5 m/s** and no hourly gust **> 6 m/s**,
  - mean total cloud cover over the night **≤ 30 %**,
  - dew point at sunset **≤ 5 °C** (dry air → stronger radiative cooling).
- `near_trigger`: the same wind, cloud and dew point conditions, but min temperature
  **between 4 °C and 10 °C**. The gap between 1 °C and 4 °C is deliberate: borderline nights
  are neither frost nor clean controls. These nights are computed now for the later
  placebo test and are shown in the visualization.

**Step 2 — Growth phase (per field).** From emergence (first date the field's median EVI2,
smoothed, rises above 0.20) up to the date of the seasonal EVI2 peak. Nights outside this
window are not evaluated (decline, senescence, vine kill, harvest are excluded).

**Step 3 — Expected trajectory (per pixel).** Fit a Gaussian Process Regression to the
pixel's EVI2 time series using **only clear observations before the night**, and predict
EVI2 at the first clear acquisition after the night.
A pixel-night is valid only if:
- at least **4** clear observations exist since emergence, the last one ≤ **20 days** before the night,
- the first clear observation after the night is ≤ **10 days** after it.

**Step 4 — Anomaly and damage score (per pixel).**
- `z = (observed − predicted) / predictive std`
- continuous damage score = `max(0, −z)` (higher = more damage)
- binary label: damaged if **z ≤ −2.0**

**Step 5 — Dose.** Degree-hours below **1.0 °C** for that night, from the RTMA cell
(sum over night hours of max(0, 1.0 − TMP)).

## Method interface (important for later phases)
The method must be callable as a plugin, for ANY field and ANY night, not only trigger nights:

```
damage(field_id, night) -> {
  score: per-pixel damage score,
  valid: per-pixel mask (cloud, edge, outside growth phase, too few observations, ...),
  meta:  pre_date, post_date, gap_days_before, gap_days_after,
         growth_stage, dose, trigger flag, near_trigger flag
}
```

Keep it separate from the data layer and from the visualization, e.g.
`data/` (download + preprocessing), `methods/evi2_gpr/` (this method),
`report/` (visualization), `runs/<id>/` (outputs), `config.yaml` (all thresholds above).
A future method must be addable as a new folder under `methods/` without touching
anything else.

## Outputs (per run, in `runs/<id>/`)
- nights table (cell × night, with flags and dose)
- field-night table (one row per field × evaluated night: mean z, share of damaged pixels,
  valid pixel share, dates, gaps, stage, flags)
- per field-night pixel rasters: score, valid mask, observed / predicted EVI2, RGB
- run metadata: config used and git commit

## Visualization
One static HTML file `runs/<id>/report.html`, opened in a browser without a server
(Leaflet or similar). It **only reads run outputs**: it computes nothing and downloads nothing.
Build it once and keep it simple; do not keep polishing it.

1. **Season timeline (navigation).** A horizontal strip over the season showing trigger
   nights (shaded by dose), near-trigger nights, clear Sentinel-2 acquisitions (cloudy
   ones faded) and growth stages as background bands. Clicking a night selects it.
2. **County view — "where did frost happen?"** Hillshade background; potato field polygons
   colored by one field-level value for the selected night (share of damaged pixels or
   mean z); RTMA 2.5 km grid outlined, showing which cells triggered that night.
   Fields in non-triggered cells are shown the same way, so triggered and non-triggered
   neighbors can be compared side by side.
3. **Field view — "is this damage real?"** (click a field) Toggleable pixel layers: RGB,
   EVI2 before, EVI2 after, predicted EVI2, z-score, valid mask, relative elevation.
   Every image is labeled with its day offset from the night (e.g. "before: −4 d,
   after: +6 d"); show the next clear acquisition too if available, to see persistence
   or recovery. Below: the field's EVI2 time series with the prediction band and the
   night marked as a vertical line.

Do not show only damage. Near-trigger nights, non-triggered fields and trigger nights
without detected damage must be just as visible as the positive cases.

Practical limits: no 10 m rasters over the whole county. County level = polygons +
field values + a downsampled hillshade; 10 m pixels only in the field view, as small
pre-rendered images per field-night. Keep the report reasonably small.

## Working rules
- Do not create notes, idea or summary files. Documentation lives in one `README.md`
  (how to run, data sources, config, method interface).
- Work in small steps; commit after each working step.
- If something in this brief is impossible or contradictory, stop and ask; do not
  silently change the method or a threshold.

## Done when
For one county and one season the pipeline runs end to end with one command, produces the
outputs above, and `report.html` lets me navigate the season, the county and individual fields.
Then stop and report: what ran, data volume, known issues. No conclusions about whether
the damage is real; that is the next phase.
