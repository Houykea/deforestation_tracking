# Deforestation Tracking

A machine-learning project that marks each pixel of a Sentinel-2 satellite image as
**forest / non-forest**, then compares the maps from different years to find where
forest was **lost** (in hectares).

The whole pipeline lives in one readable notebook: **`notebooks/deforestation.ipynb`**.

## The idea in plain words

- The model answers **one question per pixel**: *is this dot forest or not?*
- **Training** = showing it 484 little labelled pictures so it learns the pattern.
- **"Deforested" is not predicted by the model.** We get it by comparing two years:
  a pixel that was forest before and is not forest now is loss. The hectares are
  just counting those pixels.

## Data (in `data/`, not committed - download with `scripts/get_data.ps1`)

| Folder | What it is |
|---|---|
| `data/tiles/` | 484 labelled training chips (256x256, 9 channels). Split 340 / 72 / 72. |
| `data/hf/composites/` | 4 yearly images: `s2_2019/2021/2023/2025.tif` |

Label values: `0` non-forest, `1` forest, `2` deforested. Channels:
`B02 B03 B04 B08 B11 B12 NDVI NBR NDWI`.

Imagery comes from the Hugging Face dataset `umeshuu/cambodia-deforestation`
(pinned revision), so every teammate gets identical files.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

.\scripts\get_data.ps1          # download the data from Hugging Face

jupyter notebook notebooks/deforestation.ipynb
```

Run all cells top to bottom. The notebook trains the model, prints the metrics,
and shows the forest-loss chart and maps.

## Optional dashboard

The notebook saves its results to `outputs/`. To view them as a web page:

```powershell
streamlit run app/streamlit_app.py
```

## How to answer the professor

- **The ML:** a supervised **binary pixel classifier** (Random Forest). Input = a
  pixel's 9 channels; output = forest / non-forest. Trained on 484 labelled chips;
  evaluated with per-class **IoU / F1** (not just accuracy).
- **Why binary, not 3-class:** the "deforested" label is defined using Hansen loss
  over 2019-2025, but the input is the 2021 image, so a pixel cleared later still
  looks like forest and a pixel cleared earlier looks like non-forest. The label
  isn't visible in the input, so it can't be learned - it comes from the
  year-to-year comparison instead.
- **The deforestation number:** run the model on each yearly image -> forest map ->
  compare years -> count the pixels that went forest -> non-forest -> hectares.
- **Caveats:** labels are auto-generated (WorldCover 2021 + Hansen), only 2021 is
  labelled, and classes are imbalanced.
- **Next step:** swap the Random Forest for a **U-Net** (same binary task, but it
  uses spatial context) and compare per-class IoU / F1.

## Layout

```
notebooks/deforestation.ipynb   the whole pipeline (data -> model -> hectares -> chart)
app/streamlit_app.py            optional dashboard over outputs/
scripts/get_data.ps1            pinned Hugging Face download
data/                           chips + yearly images (git-ignored)
outputs/                        results written by the notebook (git-ignored)
models/                         saved models (git-ignored)
```
