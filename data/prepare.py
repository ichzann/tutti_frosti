"""Build every cached input for the study county (each step is skipped when its cache exists)."""
from __future__ import annotations

from . import cdl, dem, rtma, s2, s2qa
from .common import cache_dir


def prepare(cfg, log=print):
    fips = cfg["study"]["county_fips"]
    c = cache_dir(cfg)
    if not (c / "nights.parquet").exists():
        cells, nights = rtma.run(cfg, log=log)
        cells.to_file(c / "rtma_cells.gpkg", driver="GPKG")
        nights.to_parquet(c / "nights.parquet")
    dem.county_dem(cfg, fips)
    cdl.county_cdl(cfg, fips)
    cdl.cleanup(cfg)
    s2.build(cfg, fips, log=log)
    s2qa.build(cfg, fips, log=log)
