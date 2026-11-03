# Validating damage detection without ground truth — literature summary

Scan date: 2026-11-03. Purpose: find how others confirm satellite-detected crop damage/stress when no field truth exists, to test `methods/frost_recovery`.

Read in full: Cogato et al. 2020 (`papers/remotesensing-12-01896.pdf`). Everything else is from abstracts/search summaries only.

## 1. Affected vs. control comparison (closest to frost_recovery)

**Cogato et al. 2020, *Remote Sensing* 12:1896** — late frost, 19–20 Apr 2017, Garganega vineyards, N Italy, Sentinel-2.
- Design: 3 frost-affected plots (F, 4.96 ha) vs 3 unaffected control plots (NF, 11.43 ha), within 20 km², ~10 km apart. F/NF known from grower observation, no damage scoring.
- Normalisation: each 2017 pixel divided by the mean of the same date in 2018–2019 (non-frost years), then scaled to the first 2017 date = 1. This removes interannual and site effects.
- Test: unpaired two-sample t-test F vs NF per image date, pixels as samples.
- Damage signal: EVI −16.6%, CARI −5.3%, MTVI1 −5.8%, NIR −16.5%, Red Edge 7 −16.7% vs NF. NDVI showed no significant difference at damage time.
- Recovery: significant recovery after 24 May; full recovery ~40 days after frost (CARI, EVI, MTVI1, NIR, RE7, NDVI, MSR).
- Independent check: must quality at harvest (total acidity +5.98%, pH −2.47% in F) — a downstream crop outcome used as corroboration.
- Limits: 6 plots, pixels treated as independent samples (inflates df), a single event.

## 2. Insurance / claims data

- **GEEadas (Chaves, Ademi et al., Nov 2025)** — Sentinel-2 + Random Forest frost detection on corn, Brazil. Checked against State Dept. of Agriculture records and insurer farmer-loss data. *Full text not yet read.*
- **Hail studies** — Lombardy/Brescia maize (insurer adjuster surveys, ~125 fields), Canada canola/wheat/lentil (Sentinel-2 vs ground/insurance estimates). Field-level agreement with adjuster damage %.
- **UNL drought report** — USDA RMA cause-of-loss claims (county × month) cross-tabbed with US Drought Monitor category. Simple contingency design, transferable to RMA "Freeze" claims.
- RMA Cause of Loss data: county × commodity × cause × month since 1989; AgRisk viewer (NMSU) for browsing.

## 3. Documented events + official statistics / news

- **Shandong winter wheat, Apr 2018 (*Plants* 2023)** — MODIS frost map (89.7% reported accuracy), checked against county yield statistics and news accounts.
- **Chuvashia drought 2010** — satellite-mapped affected area matched official crop-death statistics.
- Local equivalents: NASS Idaho Crop Progress (week ending 23 Jun 2024 mentions frost), extension/trade press; a second Sentinel-2-era E Idaho frost (June 2026 reported) as an out-of-sample event.

## 4. Independent sensors / measurements

- Station temperature (AgriMet, e.g. Aberdeen) as independent exposure evidence — check whether RTMA assimilates them.
- Second sensor (Landsat, Sentinel-1 SAR) confirming the same anomaly. NASS validated Sentinel-1 flood maps against manually derived reference (Boryan 2018).

## 5. No-truth statistics

- **Triple collocation (Pierdicca et al. 2018, IEEE TGRS)** — error variance of three independent damage maps estimated without truth. *Full text not yet read.*
- Synthetic injection + null tests (already in `validation/`).

## 6. Yield linkage

- Grower yield maps / county yields vs damage maps (China wheat, GRDC Australia frost service). For potatoes this would need grower or processor yields.

## Notes

- No study found validates frost detection against insurance data at field level; insurance/statistics checks are aggregate (county or claim).
- Field-level evidence without truth comes mainly from affected-vs-control designs (Cogato) and independent station data.

## Sources

- Cogato et al. 2020 — https://doi.org/10.3390/rs12111896
- Shandong MODIS frost — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10708044/
- GRDC frost note — https://grdc.com.au/resources-and-publications/grdc-update-papers/tab-content/grdc-update-papers/2021/08/using-satellite-data-to-assess-frost-damage
- Hail, Sentinel-2 (Canada) — https://science-ouverte.canada.ca/items/90de773d-fe5a-4dc6-a3e9-3aadc758a7e1
- Hail/insurer (Padua) — https://research.unipd.it/handle/11577/3508508
- UNL RMA vs Drought Monitor — https://droughtcenter.unl.edu/Outgoing/RMACropInsurance/Year%201%20CauseOfLoss%20HotColdSpots%20report.pdf
- AgRisk viewer — https://agrisk.jornada.nmsu.edu/
- Chuvashia drought — https://agris.fao.org/search/en/records/65df8d6663b8185d9cad47f5
- NASS Sentinel-1 flood — https://www.nass.usda.gov/Research_and_Science/Cropland/docs/Boryan_IGARSS2018%20_Flood.pdf
