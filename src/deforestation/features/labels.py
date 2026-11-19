"""Per-year forest / non-forest labels from Hansen GFC + ESA WorldCover.

Label codes:  1 = forest, 0 = non-forest, 255 = not used (uncertain or outside Kratie).

Image year Y shows the land at the end of Y (dry season Nov Y - Feb Y+1).

  forest      Hansen tree cover 2000 >= 60%
              AND WorldCover 2021 = tree
              AND no Hansen loss up to and including Y+1
  non-forest  (a) never forest: tree cover 2000 < 10%, WorldCover not tree, no loss
              (b) recently cleared: Hansen loss in Y-3 .. Y-1
  skipped     loss in Y or Y+1 (may fall before or after the image dates),
              loss older than Y-3 (may have regrown or become plantation),
              tree cover 10-60%, or the two products disagree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject, transform_geom

from deforestation import config

FOREST, NONFOREST, IGNORE = 1, 0, 255


def label_rule(treecover2000: np.ndarray, lossyear: np.ndarray, worldcover: np.ndarray,
               year: int) -> np.ndarray:
    """Apply the label rules to aligned arrays. lossyear uses Hansen coding (0 = none, 1 = 2001)."""
    loss = np.where(lossyear > 0, 2000 + lossyear.astype("int16"), 0)
    no_loss = loss == 0
    wc_tree = worldcover == config.WORLDCOVER_TREE
    wc_valid = worldcover != config.WORLDCOVER_NODATA

    forest = (
        (treecover2000 >= config.FOREST_MIN_TREECOVER)
        & wc_tree
        & (no_loss | (loss > year + 1))
    )
    never_forest = (treecover2000 < config.NONFOREST_MAX_TREECOVER) & wc_valid & ~wc_tree & no_loss
    recently_cleared = (loss >= year - config.RECENT_LOSS_YEARS) & (loss <= year - 1)

    out = np.full(treecover2000.shape, IGNORE, dtype="uint8")
    out[never_forest | recently_cleared] = NONFOREST
    out[forest] = FOREST
    return out


def _on_grid(path: Path, profile: dict, resampling: Resampling) -> np.ndarray:
    dst = np.zeros((profile["height"], profile["width"]), dtype="uint8")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=resampling,
        )
    return dst


def inside_kratie(profile: dict) -> np.ndarray:
    """Boolean mask of grid pixels inside the official Kratie polygon."""
    from deforestation.data.labels_download import load_boundary

    geom = transform_geom("EPSG:4326", profile["crs"], load_boundary())
    return ~geometry_mask([geom], (profile["height"], profile["width"]), profile["transform"])


def build_label_raster(year: int, grid_profile: dict, dest: Path) -> dict[str, int]:
    """Write the label raster for one year on the composite grid. Returns class counts."""
    tc = _on_grid(config.LABELS_DIR / "hansen_treecover2000_kratie.tif", grid_profile, Resampling.nearest)
    ly = _on_grid(config.LABELS_DIR / "hansen_lossyear_kratie.tif", grid_profile, Resampling.nearest)
    wc = _on_grid(config.LABELS_DIR / f"worldcover_{config.WORLDCOVER_YEAR}_kratie.tif",
                  grid_profile, Resampling.mode)

    labels = label_rule(tc, ly, wc, year)
    labels[~inside_kratie(grid_profile)] = IGNORE

    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1, "nodata": IGNORE,
        "height": grid_profile["height"], "width": grid_profile["width"],
        "crs": grid_profile["crs"], "transform": grid_profile["transform"],
        "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(labels, 1)
        dst.set_band_description(1, "1=forest 0=non-forest 255=ignore")
        dst.update_tags(year=year, rule=__doc__.strip().splitlines()[0])

    return {
        "forest": int((labels == FOREST).sum()),
        "non_forest": int((labels == NONFOREST).sum()),
        "ignored": int((labels == IGNORE).sum()),
    }
