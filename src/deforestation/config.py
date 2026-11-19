"""Central settings. Change values here, not inside modules."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Bounding box of Kratie Province (min_lon, min_lat, max_lon, max_lat), WGS84.
# Official extent (geoBoundaries ADM1, OpenStreetMap): lon 105.587-106.802, lat 11.949-13.407.
# Padded by ~0.05 deg so edge pixels are not cut off. The exact polygon is in KRATIE_BOUNDARY.
KRATIE_BBOX = (105.54, 11.90, 106.85, 13.46)
BOUNDARY_API = "https://www.geoboundaries.org/api/current/gbOpen/KHM/ADM1/"
PROVINCE_NAME = "Kratie"

# Project period. Fixed (not "today") so results are reproducible.
START_YEAR = 2019
END_YEAR = 2025

# Years with a Sentinel-2 composite (every 2 years keeps the download manageable).
# Image "Y" = dry season from 1 Nov of Y to end of Feb of Y+1, so it shows the land at the
# END of year Y. Comparing images Y1 and Y2 then matches Hansen loss years Y1+1 .. Y2.
#   2019 -> 2021 : loss in 2020-2021
#   2021 -> 2023 : loss in 2022-2023
#   2023 -> 2025 : loss in 2024-2025
COMPOSITE_YEARS = (2019, 2021, 2023, 2025)
SEASON_START_MONTH_DAY = (11, 1)  # in year Y
SEASON_END_MONTH_DAY = (2, 28)    # in year Y+1 (inclusive)


def season_dates(year: int) -> tuple[str, str]:
    (sm, sd), (em, ed) = SEASON_START_MONTH_DAY, SEASON_END_MONTH_DAY
    return f"{year}-{sm:02d}-{sd:02d}", f"{year + 1}-{em:02d}-{ed:02d}"


# Sentinel-2 imagery: public STAC catalog, no account needed.
STAC_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
# Earth Search asset name -> short band name used in files and features.
S2_BANDS = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8", "swir16": "B11", "swir22": "B12"}
S2_MAX_SCENE_CLOUD = 40      # % cloud per scene for the catalog search
S2_SCENES_PER_TILE = 5       # keep only the clearest N scenes per Sentinel-2 tile per year
# Scene Classification (SCL) classes kept as clear: 4 vegetation, 5 bare soil, 6 water.
# Dropped: 0 nodata, 1 saturated, 2 dark/shadow, 3 cloud shadow, 7 unclassified,
#          8-10 clouds and cirrus, 11 snow.
S2_SCL_CLEAR = (4, 5, 6)

# Output grid shared by composites, labels and predictions.
GRID_CRS = "EPSG:32648"  # UTM zone 48N, metres
GRID_RES = 20            # metres
GRID_WINDOW = 1024       # pixels per processing window (memory control)

# Spatial blocks for train/test splitting (avoid neighbouring pixels in both sets).
BLOCK_SIZE_M = 5_000
TEST_BLOCK_FRACTION = 0.2

# Label rules (see features/labels.py).
FOREST_MIN_TREECOVER = 60     # Hansen 2000 tree cover %, to count as forest
NONFOREST_MAX_TREECOVER = 10  # below this in 2000 = never forest
RECENT_LOSS_YEARS = 3         # loss within this many years before Y = still cleared in Y
WORLDCOVER_TREE = 10
WORLDCOVER_NODATA = 0

# Sampling for the training table.
SAMPLES_PER_CLASS_PER_YEAR = 5_000

# Label sources: direct public GeoTIFF downloads, no account needed.
# Kratie spans two WorldCover 3x3-degree tiles (south edge is below 12N) and one Hansen tile.
WORLDCOVER_YEAR = 2021
WORLDCOVER_TILES = ("N09E105", "N12E105")


def worldcover_url(tile: str) -> str:
    return (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    )


HANSEN_VERSION = "GFC-2025-v1.13"  # loss recorded through 2025
HANSEN_TILE = "20N_100E"
HANSEN_LAYERS = ("treecover2000", "lossyear")


def hansen_url(layer: str) -> str:
    return (
        f"https://storage.googleapis.com/earthenginepartners-hansen/{HANSEN_VERSION}/"
        f"Hansen_{HANSEN_VERSION}_{layer}_{HANSEN_TILE}.tif"
    )


RANDOM_SEED = 42

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABELS_DIR = DATA_DIR / "labels"
KRATIE_BOUNDARY = RAW_DIR / "kratie_boundary.geojson"
COMPOSITES_DIR = DATA_DIR / "composites"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_TABLE = PROCESSED_DIR / "training_samples.parquet"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
