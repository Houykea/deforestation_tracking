"""Download Sentinel-2 and build one dry-season composite per year.

Usage:
  python scripts/download_composites.py                       # all COMPOSITE_YEARS, whole Kratie
  python scripts/download_composites.py --years 2019 2021
  python scripts/download_composites.py --test                # small 5x5 km area, quick check

Years whose file already exists are skipped, so an interrupted run can be restarted.
"""
from __future__ import annotations

import argparse
import time

from deforestation import config
from deforestation.data.aoi import BBox, kratie
from deforestation.data.sentinel2 import build_composite

TEST_BBOX = BBox(106.05, 12.55, 106.10, 12.60)  # ~5 x 5 km inside Kratie


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="+", default=list(config.COMPOSITE_YEARS))
    p.add_argument("--test", action="store_true", help="small area, written to data/composites/test/")
    p.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    args = p.parse_args()

    bbox = TEST_BBOX.validate() if args.test else kratie()
    out_dir = config.COMPOSITES_DIR / "test" if args.test else config.COMPOSITES_DIR

    for year in args.years:
        dest = out_dir / f"s2_{year}.tif"
        start, end = config.season_dates(year)
        print(f"[{year}] season {start} to {end}")
        if dest.exists() and not args.force:
            print(f"  exists  {dest.relative_to(config.ROOT)}")
            continue
        t0 = time.time()
        info = build_composite(bbox, year, dest)
        print(f"  saved -> {dest.relative_to(config.ROOT)}  ({(time.time() - t0) / 60:.1f} min)")
        print(f"  pixels with data: {info['valid_pct']:.1f}%   "
              f"median clear observations: {info['median_clear_obs']:.0f}")


if __name__ == "__main__":
    main()
