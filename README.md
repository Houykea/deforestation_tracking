# Deforestation Tracking — Prey Lang (Cambodia)

A machine-learning project that labels Sentinel-2 sample points as **forest / non-forest**
and tracks forest cover decline across years in the **Prey Lang** landscape (Cambodia).

Everything runs in one Colab notebook: **`notebooks/RF_prey_lang_deforest_test_training.ipynb`**.

## The idea in plain words

- The model answers **one question per point**: *is this spot forest or not?* (a binary
  Random Forest over 6 Sentinel-2 bands).
- **Training** = 10,000 fixed points sampled in the same locations every year, labelled with
  the Hansen Global Forest Change v1.13 dataset.
- **Forest loss** is not predicted directly: we run the model on each yearly composite and
  compare years — a point that was forest earlier and is not forest later is a loss.

## Data — shared Google Drive

The dataset lives in a shared Google Drive folder (we no longer commit imagery):

**[PreyLang folder on Google Drive](https://drive.google.com/drive/folders/1kvfNXBqCKLmKLxW5Yipdf9tU5iWlXsfq?usp=sharing)** — path inside Colab: `/content/drive/MyDrive/PreyLang/`

| File | What it is |
|---|---|
| `PreyLang_Samples_{2019,2021,2023,2025}.csv` | 10,000 fixed sample points per year (~2,617 valid after cloud masking); columns `B2 B3 B4 B8 B11 B12 label year lon lat` |
| `PreyLang_Composite_{year}_tile_{i}_{j}.tif` | 4 tiled GeoTIFFs per year (16 total), `Float32`, 7 bands: `B2 B3 B4 B8 B11 B12 label` |
| `rf_preylang.pkl` | the trained Random Forest (written by the notebook) |
| `results_summary.json`, `confusion_matrix.png`, `feature_importance.png` | results written by the notebook |

Label values: `0.0` non-forest, `1.0` forest, `NaN` masked / outside the Prey Lang boundary.

### How the data was made

Imagery is a **Sentinel-2 L2A** dry-season median composite (Dec–Mar), cloud/shadow masked
(`SCL` classes 3, 9, 10). Labels come from **Hansen GFC v1.13** — a point is *forest* if
`treecover2000 >= 30` **and** it has not been lost by that year.

To regenerate the data yourself, run **`scripts/preylang_export.js`** in the
[Google Earth Engine Code Editor](https://code.earthengine.google.com): paste it, click
**Run**, then **Run** each export task. Output lands in a Drive folder named `PreyLang`.
The 10,000 points are fixed (`seed: 42`), so everyone gets the same locations.

## Quick start

1. Open the Google Drive folder above and make a copy (or open the notebook directly).
2. Add `notebooks/RF_prey_lang_deforest_test_training.ipynb` to
   [Google Colab](https://colab.research.google.com) and either:
   - run it in the shared Drive (Cell 1 mounts Drive and reads `/content/drive/MyDrive/PreyLang/`), or
   - copy the `PreyLang/` folder into your own Drive.
3. **Runtime → Run all.** The notebook trains the model, prints the metrics and the
   per-year forest-fraction predictions, and saves the `.pkl` + result files back to Drive.

## Results (Random Forest baseline)

**5-fold spatial cross-validation** (2019, 2021, 2023 — grouped by 0.01° blocks):

```
Mean F1 (forest): 0.649 ± 0.034
Folds:            0.650, 0.676, 0.691, 0.593, 0.636
Accuracy:         85.7% – 90.0%
Feature importance: B11 (0.323) > B3 (0.202) > B12 (0.179) > B8 (0.115) > B2 (0.091) > B4 (0.090)
```

**Leave-one-year-out generalization:** 2019 `F1 0.724`, 2021 `F1 0.633`, 2023 `F1 0.470`
(mean `0.609`).

**Full-tile forest cover inside the Prey Lang mask** (11,945,268 valid pixels/year):

| Year | Ground-truth sample forest % | Predicted full-tile forest % |
|:--|:--:|:--:|
| 2019 | 27.10% | **27.23%** |
| 2021 | 16.50% | **20.40%** |
| 2023 | 6.30% | **9.84%** |
| 2025 | *(labels unusable)* | **6.11%** |

## Caveats (state these honestly)

- Labels are **auto-generated** (Hansen GFC), not hand-corrected — label quality is the ceiling.
- **2025 labels are unusable** (Hansen's preliminary layer over-reports loss: `0/2617` forest) —
  2025 is used only as an unlabeled inference target.
- Classes are **imbalanced** and the forest fraction shrinks each year (27.1% → 6.3%), so
  per-class F1 matters more than accuracy.
- `class_weight='balanced'` gives a small **positive over-prediction bias** (~+3.5%) in the
  rare-forest years.
- One study area (Prey Lang); generalization elsewhere is untested.

See **`REPORT.md`** for a full write-up of what changed versus the earlier approach and what
we learned, and `NOTES.md` for working notes.

## Layout

```
notebooks/RF_prey_lang_deforest_test_training.ipynb   Colab: data -> model -> per-year predictions
scripts/preylang_export.js                            Google Earth Engine export script
REPORT.md                                             improvements + learnings write-up
NOTES.md                                              working notes
requirements.txt                                      local-run dependencies (most use Colab)
legacy/                                               retired code (not tracked)
```
