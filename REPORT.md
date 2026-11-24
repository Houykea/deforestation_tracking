# REPORT — Improvements over the previous approach & what we learned

This report compares the new **Prey Lang** pipeline (`notebooks/RF_prey_lang_deforest_test_training.ipynb`,
Hansen GFC labels, shared Drive data) with the earlier **Cambodia / ESA WorldCover** pipeline
(`notebooks/deforestation.ipynb`, since retired to `legacy/`) and records what we learned along the way.

---

## 1. What the previous approach did

- **Imagery:** Sentinel-2 chips (256×256, 9 channels incl. NDVI/NBR/NDWI), one labelled year (2021).
- **Labels:** auto-generated from **ESA WorldCover 2021** + Hansen GFC — a single year only.
- **Model:** binary (forest / non-forest) Random Forest over per-pixel features.
- **Validation:** one random train/test split; reported per-class IoU/F1.
- **Loss:** compare the model's forest maps between years.

It worked (forest IoU ≈ 0.98) but had two structural weaknesses: labels existed for **one year
only**, and the validation split was **random** across a spatially correlated landscape.

---

## 2. Improvements in the new pipeline

### 2.1 Per-year labels instead of a single-year label
Hansen GFC v1.13 defines forest **per year** (`treecover2000 >= 30` AND not lost by that year).
We now have labels for **2019, 2021, 2023** (2025 excluded — see §3.3), so we can:

- train on multiple years, and
- test whether the model **generalizes across time**, not just across space.

The old approach could only ever validate 2021.

### 2.2 Spatial cross-validation (no leakage)
We replaced the random split with **`GroupKFold` over 0.01° spatial blocks** (1,239 blocks).
Random splits let near-identical neighbouring points fall into both train and test, inflating
F1 by roughly **~0.15**. Spatial blocks give an honest estimate (Mean F1(forest) `0.649 ± 0.034`).

### 2.3 Leave-one-year-out cross-year evaluation
Training on some years and testing on a held-out year (`0.724 / 0.633 / 0.470`) surfaces a
failure mode the old single-year setup could not see: the model degrades as the minority class
becomes rarer in later years.

### 2.4 Honest imbalance handling
The forest fraction collapses over time (**27.1% → 16.5% → 6.3%**). We use
`class_weight='balanced'` and report **per-class F1**, and we explicitly flag the resulting
**~+3.5% positive over-prediction bias** rather than hiding it behind overall accuracy.

### 2.5 Correct full-tile masking
During inference we restrict scoring to the Prey Lang polygon via `~np.isnan(band7)` (the label
band is `NaN` outside the boundary) in addition to `~np.isnan(B2:B12)`. Without this, the
rectangular tile bounding box would silently include out-of-polygon pixels.

### 2.6 Reproducible, shared data
The dataset moved to a **shared Google Drive** with a documented GEE export script
(`scripts/preylang_export.js`, `seed: 42` → identical 10,000 points for everyone), replacing the
ad-hoc local chips.

### 2.7 A physically sensible result
Feature importance is **SWIR-1 (`B11`) dominant**, as expected: intact canopy holds moisture and
absorbs SWIR, while cleared land reflects it. This is a sanity check that the model learned
vegetation structure, not an artifact.

---

## 3. What we learned

### 3.1 Spatial autocorrelation is the first thing to fix
The single biggest gain in *trustworthiness* was blocking spatial leakage. A random split looks
better but is misleading; spatial GroupKFold is now mandatory for this kind of data.

### 3.2 A single year's image cannot carry the "deforested" label
Loss is defined over a time range, so a pixel cleared *after* the image year still looks like
forest, and one cleared *before* looks like non-forest. The label is not visible in the input,
so it cannot be learned directly — it must come from comparing years. (Consistent with the
earlier finding that the 3-class "deforested" IoU was ≈0.07 vs 0.98 for binary forest.)

### 3.3 Preliminary label years are dangerous
Hansen's **preliminary 2025 layer over-reports loss**: it yields `0/2,617` forest points. Training
on it would teach the model that essentially everything is non-forest. We exclude 2025 from
training and use it only as an unlabeled inference target — a reminder to always inspect label
distributions per year before trusting them.

### 3.4 Temporal generalization is bounded by class rarity, not just noise
Cross-year F1 falls `0.724 → 0.633 → 0.470` as the forest class shrinks to ~6%. With few minority
samples and small inter-year radiometric shifts, late-year performance is inherently hard. Report
this honestly instead of a single headline number.

### 3.5 Balanced weighting buys recall at the cost of precision
`class_weight='balanced'` is necessary for the rare class but introduces a real
**~+3.5% over-prediction** of forest in 2021–2023. A balanced number needs a calibration/
threshold step, or the bias must be stated as a caveat.

### 3.6 The label ceiling dominates everything
Because labels are auto-generated (Hansen GFC), no modelling change can exceed label quality.
The most valuable next investment is **hand-fixing a few hundred labels**, not a fancier model.

### 3.7 Keep it reproducible
Pinned points (`seed: 42`), a committed GEE script, and shared Drive storage mean any teammate
reproduces the exact same dataset — the thing the older local-chip workflow got wrong.

---

## 4. Result summary (new pipeline)

```
Spatial CV:  mean F1(forest) 0.649 ± 0.034  (folds 0.650, 0.676, 0.691, 0.593, 0.636)
Cross-year:  2019 0.724 | 2021 0.633 | 2023 0.470  (mean 0.609)
Importance:  B11 0.323 > B3 0.202 > B12 0.179 > B8 0.115 > B2 0.091 > B4 0.090

Full-tile forest % (inside polygon):
  2019 27.23%  |  2021 20.40%  |  2023 9.84%  |  2025 6.11%
```

## 5. Recommended next steps

1. **U-Net** on the tiles (same binary task, adds spatial context) and compare F1/IoU.
2. **Calibrate / threshold** the Random Forest to reduce the balanced-weight bias.
3. **Hand-fix labels** for a subset to raise the accuracy ceiling.
4. **Independent validation** against Hansen `lossyear` on the loss map.
