"""Build per-year label rasters (Hansen + WorldCover) and the training table.

Needs: scripts/download_labels.py and scripts/download_composites.py done first.

Usage:
  python scripts/build_training_data.py          # whole Kratie
  python scripts/build_training_data.py --test   # the small test area

Outputs:
  data/labels/labels_<year>.tif          1 forest, 0 non-forest, 255 ignore; same grid as s2_<year>.tif
  data/processed/training_samples.parquet  one row per sampled pixel
"""
from __future__ import annotations

import argparse

import rasterio

from deforestation import config
from deforestation.features.labels import build_label_raster
from deforestation.features.sampling import build_training_table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="+", default=list(config.COMPOSITE_YEARS))
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    comp_dir = config.COMPOSITES_DIR / "test" if args.test else config.COMPOSITES_DIR
    lab_dir = config.LABELS_DIR / "test" if args.test else config.LABELS_DIR
    table_path = (config.PROCESSED_DIR / "test" / config.TRAINING_TABLE.name) if args.test \
        else config.TRAINING_TABLE

    for year in args.years:
        comp = comp_dir / f"s2_{year}.tif"
        if not comp.exists():
            raise SystemExit(f"Missing {comp}. Run scripts/download_composites.py first.")
        with rasterio.open(comp) as src:
            grid = {"height": src.height, "width": src.width, "crs": src.crs, "transform": src.transform}
        counts = build_label_raster(year, grid, lab_dir / f"labels_{year}.tif")
        total = sum(counts.values())
        print(f"[{year}] labels: forest {counts['forest']:,} ({counts['forest'] / total:.1%})  "
              f"non-forest {counts['non_forest']:,} ({counts['non_forest'] / total:.1%})  "
              f"ignored {counts['ignored'] / total:.1%}")

    table = build_training_table(args.years, comp_dir, lab_dir, table_path)
    print(f"\nTraining table -> {table_path.relative_to(config.ROOT)}  ({len(table):,} rows)")
    summary = table.groupby(["year", "split", "label"]).size().unstack("label", fill_value=0)
    summary.columns = ["non-forest" if c == 0 else "forest" for c in summary.columns]
    print(summary.to_string())
    print("\nMean NDVI by class:",
          table.groupby("label")["NDVI"].mean().rename({0: "non-forest", 1: "forest"}).round(3).to_dict())


if __name__ == "__main__":
    main()
