# Project notes — Prey Lang (working notes, committed for the team)

Plain-language notes for us and the team. The polished write-up lives in `README.md`,
the improvements/learnings story in `REPORT.md`.

## What the project does, in plain words

- The model answers **one question per sample point**: *is this spot forest or not?*
- **Training** = 10,000 fixed points, labelled every year with Hansen GFC v1.13.
- **"Loss" is not predicted by the model.** We run the model on each yearly composite and
  compare years: a point that was forest in an earlier year and is not forest later is a loss.

## What the model learns (and what it doesn't)

A **binary Random Forest** over 6 Sentinel-2 bands (`B2 B3 B4 B8 B11 B12`). Output =
forest / non-forest **per point**. It does **not** predict loss or hectares directly.

## The data

- `PreyLang_Samples_{2019,2021,2023,2025}.csv` — 10,000 fixed points/year, ~2,617 valid
  (cloud-free) each; **same locations every year** so the comparison is apples-to-apples.
- `PreyLang_Composite_{year}_tile_{i}_{j}.tif` — 4 tiles/year (16 total), `Float32`,
  bands `[B2 B3 B4 B8 B11 B12 label]`; `label` is `NaN` outside the Prey Lang polygon.
- Labels: **Hansen GFC v1.13** — forest if `treecover2000 >= 30` and not lost by that year.
- Imagery: Sentinel-2 L2A dry-season median (Dec–Mar), `SCL` 3/9/10 masked.
- Everything lives in the shared Google Drive `PreyLang/` folder; regenerate with
  `scripts/preylang_export.js` in GEE.

## Method

- **Spatial GroupKFold** (5 folds), grouping points into 0.01° lon/lat blocks (1,239 blocks).
  Never use a random `train_test_split` — spatial autocorrelation leaks and inflates F1 (~0.15).
- **`class_weight='balanced'`** — the minority (forest) class shrinks every year.
- **Leave-one-year-out** to test whether the model generalizes across time.
- At **full-tile inference**, mask with `~np.isnan(band7)` from a reference year **and**
  `~np.isnan(B2:B12)`, so we only score pixels inside the polygon.

## Results (Random Forest baseline)

```
Spatial CV:  mean F1(forest) 0.649 ± 0.034  (folds 0.650, 0.676, 0.691, 0.593, 0.636)
Cross-year:  2019 0.724 | 2021 0.633 | 2023 0.470  (mean 0.609)
Importance:  B11 0.323 > B3 0.202 > B12 0.179 > B8 0.115 > B2 0.091 > B4 0.090

Full-tile forest % (inside polygon):
  2019 27.23%  |  2021 20.40%  |  2023 9.84%  |  2025 6.11%
```

## Caveats (state these honestly)

- Labels are auto-generated (Hansen GFC), not hand-corrected — the ceiling is label quality.
- **2025 labels are unusable** (Hansen preliminary layer → `0/2617` forest); 2025 is inference only.
- Severe class imbalance, and it worsens over time (27.1% → 16.5% → 6.3%).
- `class_weight='balanced'` over-predicts forest by ~+3.5% in rare-forest years.
- One study area; generalization elsewhere untested.

## How to answer the professor

- **The ML:** a supervised **binary** point classifier (Random Forest) over 6 bands; evaluated
  with per-class F1 and spatial cross-validation (not a random split).
- **Why binary:** "deforested" is defined by Hansen over 2019–2025, so a single year's image
  can't carry that label — loss comes from comparing years.
- **The loss number:** predict each yearly composite → forest map → compare years → count
  forest→non-forest points.
- **Rigor:** spatial blocks (no leakage) + leave-one-year-out (does it generalize in time?).
- **Next step:** a U-Net on the tiles (same binary task, uses spatial context) and compare.

## Team handoff

- Open `notebooks/RF_prey_lang_deforest_test_training.ipynb` in Colab.
- Put the shared `PreyLang/` Drive folder in your Drive (or run the notebook on the shared copy).
- Optional: run `scripts/preylang_export.js` in GEE to regenerate the data from scratch.

## Future improvements

| Improvement | Effort | Benefit |
|---|---|---|
| U-Net on the tiles (spatial context) | Medium | Better edges / context than point RF |
| Hand-fix a few hundred labels | High | Raises the accuracy ceiling |
| Add features (NDVI, NBR, NDWI, texture) | Low | Small RF boost |
| Validate loss map against Hansen lossyear | Low | Independent check |
| Probability threshold tuning / calibration | Low | Reduce the balanced-weight bias |

## Known limitations (verified 2026-11)

1. Labels are auto-generated (Hansen GFC), not hand-corrected.
2. Only sample points are labelled; tiles are predicted, not validated pixel-by-pixel.
3. Class imbalance is severe and worsens over time.
4. One area (Prey Lang); cross-region generalization untested.
5. 2025 labels excluded because Hansen's preliminary layer is unusable.
