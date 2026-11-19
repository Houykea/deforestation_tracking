"""Training table: sample labelled pixels from each year's composite.

One row per pixel: year, coordinates, spatial block, split, band values, indices, label.
Split is by 5 km block (same blocks held out in every year), not by random pixel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from deforestation import config
from deforestation.features.indices import nbr, ndmi, ndvi
from deforestation.features.labels import FOREST, NONFOREST

BAND_NAMES = list(config.S2_BANDS.values())


def block_ids(rows: np.ndarray, cols: np.ndarray, width: int) -> np.ndarray:
    size = max(1, config.BLOCK_SIZE_M // config.GRID_RES)
    n_block_cols = -(-width // size)
    return (rows // size) * n_block_cols + (cols // size)


def test_blocks(n_blocks: int, seed: int = config.RANDOM_SEED) -> set[int]:
    rng = np.random.default_rng(seed)
    k = int(round(n_blocks * config.TEST_BLOCK_FRACTION))
    return set(rng.choice(n_blocks, size=k, replace=False).tolist())


def sample_year(year: int, composite: Path, labels: Path, n_per_class: int,
                rng: np.random.Generator) -> pd.DataFrame:
    with rasterio.open(labels) as src:
        lab = src.read(1)
    with rasterio.open(composite) as src:
        n_clear = src.read(src.count)
        transform, width, height = src.transform, src.width, src.height

        frames = []
        for cls in (FOREST, NONFOREST):
            rr, cc = np.nonzero((lab == cls) & (n_clear > 0))
            take = min(n_per_class, rr.size)
            idx = rng.choice(rr.size, size=take, replace=False)
            frames.append((cls, rr[idx], cc[idx]))

        rows = np.concatenate([f[1] for f in frames])
        cols = np.concatenate([f[2] for f in frames])
        label = np.concatenate([np.full(f[1].size, f[0], dtype="uint8") for f in frames])
        values = {}
        for b, name in enumerate(BAND_NAMES, 1):
            values[name] = src.read(b)[rows, cols].astype("float32") / 10_000

    xs, ys = rasterio.transform.xy(transform, rows, cols)
    df = pd.DataFrame({"year": year, "row": rows, "col": cols, "x": xs, "y": ys, **values})
    df["NDVI"] = ndvi(df["B8"], df["B4"])
    df["NBR"] = nbr(df["B8"], df["B12"])
    df["NDMI"] = ndmi(df["B8"], df["B11"])
    df["n_clear"] = n_clear[rows, cols]
    df["block_id"] = block_ids(rows, cols, width)
    df["label"] = label
    # Drop pixels where any band was missing (0 in the composite).
    df = df[(df[BAND_NAMES] > 0).all(axis=1)].reset_index(drop=True)
    df.attrs["n_blocks"] = int(block_ids(np.array([height - 1]), np.array([width - 1]), width)[0] + 1)
    return df


def build_training_table(years, composites_dir: Path, labels_dir: Path, dest: Path,
                         n_per_class: int = config.SAMPLES_PER_CLASS_PER_YEAR) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    parts = []
    n_blocks = 0
    for year in years:
        df = sample_year(year, composites_dir / f"s2_{year}.tif", labels_dir / f"labels_{year}.tif",
                         n_per_class, rng)
        n_blocks = max(n_blocks, df.attrs["n_blocks"])
        parts.append(df)
    table = pd.concat(parts, ignore_index=True)
    held_out = test_blocks(n_blocks)
    table["split"] = np.where(table["block_id"].isin(held_out), "test", "train")
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(dest, index=False)
    return table
