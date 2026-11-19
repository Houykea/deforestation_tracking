# Deforestation Tracking: Kratie, Cambodia

Pixel-based Random Forest on Sentinel-2 dry-season composites (2019, 2021, 2023, 2025) to map
forest and forest loss in Kratie. All data is public and needs no account.

| Data | Source |
|---|---|
| Sentinel-2 L2A imagery | Earth Search STAC (AWS) |
| Forest labels | ESA WorldCover 2021 (10 m) |
| Loss check / label filter | Hansen Global Forest Change (30 m) |

## Layout

```
src/deforestation/
  config.py     settings (region, years, sources, paths)
  data/         AOI, label download, Sentinel-2 composites
  features/     spectral indices, labels, sampling, splits
  models/       train, predict, evaluate, tune
  change/       forest mask, change detection, area, report
app/            Streamlit UI
scripts/        CLI entry points
data/           raw, composites, labels, ... (git-ignored)
models/         trained models
notebooks/      exploration
tests/
```

The previous NestJS/Next.js prototype is on the `legacy-nestjs-nextjs` branch.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m pytest
```

## Phase 1: data

```powershell
python scripts/download_labels.py        # Hansen + WorldCover + Kratie boundary (~790 MB once)
python scripts/download_composites.py    # Sentinel-2 dry-season composites 2019, 2021, 2023, 2025
python scripts/build_training_data.py    # per-year label rasters + training table
```

Add `--test` to the last two to run on a 5 x 5 km area first (about 2 minutes).

| Output | Content |
|---|---|
| `data/composites/s2_<year>.tif` | 20 m, UTM 48N. Bands B2 B3 B4 B8 B11 B12 (reflectance x 10000) + `n_clear` |
| `data/labels/labels_<year>.tif` | Same grid. 1 forest, 0 non-forest, 255 not used |
| `data/processed/training_samples.parquet` | Sampled pixels: bands, NDVI/NBR/NDMI, label, 5 km `block_id`, train/test `split` |

Image year Y = 1 Nov Y to 28 Feb Y+1, i.e. the land at the end of Y.
Label rules are documented in `src/deforestation/features/labels.py`.
