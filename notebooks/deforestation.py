"""Deforestation detection + visualization pipeline.

Input GeoTIFF band order (Sentinel-2 L2A, 6 bands, same for every scene):
    1=B2 (blue)  2=B3 (green)  3=B4 (red)  4=B8 (NIR)  5=B11 (SWIR1)  6=B12 (SWIR2)
Values may be DN reflectance (0-10000) or already 0-1.

Model input = those 6 bands + NDVI + NBR + NDWI = 9 channels.
Model output convention: class 0 = non-forest, class 1 = forest.
"Deforested" is NOT a per-image class; it is derived by comparing two dates
(post-classification comparison): forest before AND non-forest after.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_geom
from scipy import ndimage as ndi

BAND_ORDER = ("B2", "B3", "B4", "B8", "B11", "B12")
NUM_FEATURES = 9
EPS = 1e-6

# change-map codes
NO_CHANGE, STABLE_FOREST, LOSS, GAIN, INVALID = 0, 1, 2, 3, 255


# --------------------------------------------------------------------------- #
# Reading + features
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    bands: np.ndarray   # (6, H, W) float32 reflectance in [0, 1]
    valid: np.ndarray   # (H, W) bool, False = nodata
    crs: object
    transform: object
    path: str

    @property
    def shape(self):
        return self.bands.shape[1:]


def read_scene(path: str, reference: Scene | None = None) -> Scene:
    """Read a 6-band GeoTIFF. If `reference` is given, warp onto its grid."""
    with rasterio.open(path) as src:
        if src.count < len(BAND_ORDER):
            raise ValueError(f"{path}: need {len(BAND_ORDER)} bands "
                             f"({', '.join(BAND_ORDER)}), found {src.count}")
        idx = list(range(1, len(BAND_ORDER) + 1))
        if reference is None:
            arr = src.read(idx, masked=True)
            crs, transform = src.crs, src.transform
        else:
            h, w = reference.shape
            same = (src.crs == reference.crs and src.width == w and src.height == h
                    and src.transform.almost_equals(reference.transform))
            if same:
                arr = src.read(idx, masked=True)
            else:
                with WarpedVRT(src, crs=reference.crs, transform=reference.transform,
                               width=w, height=h,
                               resampling=Resampling.bilinear) as vrt:
                    arr = vrt.read(idx, masked=True)
            crs, transform = reference.crs, reference.transform

    mask = np.ma.getmaskarray(arr).any(axis=0)
    data = np.ma.filled(arr.astype(np.float32), 0.0)
    bad = ~np.isfinite(data).all(axis=0)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.max() > 2.0:          # DN reflectance (0-10000)
        data = data / 10000.0
    data = np.clip(data, 0.0, 1.0)
    valid = ~(mask | bad) & (data.sum(axis=0) > 0)   # all-zero = outside footprint
    return Scene(data, valid, crs, transform, path)


def _nd(a, b):
    return (a - b) / (a + b + EPS)


def compute_features(bands: np.ndarray) -> np.ndarray:
    """(6,H,W) -> (9,H,W): bands + NDVI, NBR, NDWI."""
    b2, b3, b4, b8, b11, b12 = bands
    ndvi = _nd(b8, b4)      # vegetation greenness
    nbr = _nd(b8, b12)      # burn / damage
    ndwi = _nd(b3, b8)      # water
    return np.concatenate([bands, np.stack([ndvi, nbr, ndwi])]).astype(np.float32)


def simple_cloud_mask(bands: np.ndarray, dilate_px: int = 3) -> np.ndarray:
    """Rough heuristic (bright, whitish, non-vegetated). Prefer the Sentinel-2
    SCL band or s2cloudless when you have them."""
    b2, b3, b4, b8, b11, b12 = bands
    vis = (b2 + b3 + b4) / 3.0
    whiteness = (np.maximum(np.maximum(b2, b3), b4) -
                 np.minimum(np.minimum(b2, b3), b4)) / (vis + EPS)
    ndvi = _nd(b8, b4)
    cloud = (vis > 0.22) & (whiteness < 0.35) & (ndvi < 0.35)
    if cloud.any() and dilate_px > 0:
        cloud = ndi.binary_dilation(cloud, iterations=dilate_px)
    return cloud


def pixel_area_ha(scene: Scene) -> float:
    """Area of one pixel in hectares (approximate if the CRS is geographic)."""
    t = scene.transform
    dx, dy = abs(t.a), abs(t.e)
    crs = scene.crs
    if crs is not None and crs.is_geographic:
        h, w = scene.shape
        lat = t.f + t.e * h / 2.0
        m_x = dx * 111_320.0 * math.cos(math.radians(lat))
        m_y = dy * 110_574.0
        return m_x * m_y / 10_000.0
    factor = 1.0
    try:
        factor = float(crs.linear_units_factor[1])
    except Exception:
        pass
    return (dx * factor) * (dy * factor) / 10_000.0


# --------------------------------------------------------------------------- #
# Predictors: all expose predict(features (9,H,W)) -> forest probability (H,W)
# --------------------------------------------------------------------------- #
class NDVIPredictor:
    """Training-free baseline: soft threshold on NDVI. Use it to test the
    pipeline before you have trained weights."""

    def __init__(self, threshold: float = 0.5, softness: float = 0.08):
        self.threshold, self.softness = threshold, softness

    def predict(self, feats: np.ndarray) -> np.ndarray:
        ndvi = feats[6]
        return (1.0 / (1.0 + np.exp(-(ndvi - self.threshold) / self.softness))
                ).astype(np.float32)


class RandomForestPredictor:
    """Pixel-wise sklearn RandomForest saved with joblib.
    Trained on (n_pixels, 9) features with labels 0/1."""

    def __init__(self, path: str, chunk: int = 500_000):
        import joblib
        self.model = joblib.load(path)
        self.chunk = chunk

    def predict(self, feats: np.ndarray) -> np.ndarray:
        c, h, w = feats.shape
        x = feats.reshape(c, -1).T
        out = np.empty(x.shape[0], dtype=np.float32)
        classes = list(self.model.classes_)
        pos = classes.index(1)
        for i in range(0, x.shape[0], self.chunk):
            out[i:i + self.chunk] = self.model.predict_proba(x[i:i + self.chunk])[:, pos]
        return out.reshape(h, w)


class UNetPredictor:
    """Sliding-window U-Net inference with overlap and Hann-window blending
    (avoids visible seams at tile borders)."""

    def __init__(self, weights: str, tile: int = 256, overlap: float = 0.5,
                 batch_size: int = 8, device: str | None = None):
        import torch
        from unet import UNet
        self.torch = torch
        self.tile = tile
        self.stride = max(1, int(tile * (1 - overlap)))
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt = torch.load(weights, map_location=self.device)
        cfg = {"in_channels": NUM_FEATURES, "num_classes": 2, "base": 32, "depth": 4}
        state = ckpt
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
            cfg.update({k: ckpt[k] for k in cfg if k in ckpt})
        self.model = UNet(**cfg).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        if tile % (2 ** cfg["depth"]):
            raise ValueError(f"tile must be divisible by {2 ** cfg['depth']}")

    def predict(self, feats: np.ndarray) -> np.ndarray:
        torch = self.torch
        c, h, w = feats.shape
        t, s = self.tile, self.stride
        ph = (t - h) if h < t else (-(h - t)) % s
        pw = (t - w) if w < t else (-(w - t)) % s
        padded = np.pad(feats, ((0, 0), (0, ph), (0, pw)), mode="reflect")
        H, W = padded.shape[1:]

        win1d = np.hanning(t + 2)[1:-1].astype(np.float32)
        win = np.outer(win1d, win1d)
        acc = np.zeros((H, W), np.float32)
        wsum = np.zeros((H, W), np.float32)

        coords = [(y, x) for y in range(0, H - t + 1, s) for x in range(0, W - t + 1, s)]
        with torch.no_grad():
            for i in range(0, len(coords), self.batch_size):
                chunk = coords[i:i + self.batch_size]
                batch = np.stack([padded[:, y:y + t, x:x + t] for y, x in chunk])
                logits = self.model(torch.from_numpy(batch).to(self.device))
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                for (y, x), p in zip(chunk, prob):
                    acc[y:y + t, x:x + t] += p * win
                    wsum[y:y + t, x:x + t] += win
        return (acc / np.maximum(wsum, 1e-8))[:h, :w]


# --------------------------------------------------------------------------- #
# Change detection + post-processing
# --------------------------------------------------------------------------- #
def remove_small_patches(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1 or not mask.any():
        return mask
    labels, _ = ndi.label(mask, structure=np.ones((3, 3)))
    keep = np.bincount(labels.ravel()) >= min_pixels
    keep[0] = False
    return keep[labels]


@dataclass
class ChangeResult:
    change_map: np.ndarray      # uint8 codes (see top of file)
    loss: np.ndarray            # bool
    gain: np.ndarray            # bool
    forest_before: np.ndarray   # bool
    forest_after: np.ndarray    # bool
    stats: dict


def detect_change(prob_before, prob_after, valid, px_ha,
                  threshold=0.5, margin=0.1, min_area_ha=0.5, smooth=True) -> ChangeResult:
    """Post-classification comparison with a hysteresis margin.

    loss = P(forest, before) >= threshold+margin  AND  P(forest, after) <= threshold-margin
    Pixels in the uncertain band are not counted as change, which cuts
    false positives caused by seasonal / atmospheric differences.
    """
    hi, lo = threshold + margin, threshold - margin
    loss = (prob_before >= hi) & (prob_after <= lo) & valid
    gain = (prob_before <= lo) & (prob_after >= hi) & valid
    if smooth:
        st = np.ones((3, 3), bool)
        loss = ndi.binary_opening(loss, structure=st)
        gain = ndi.binary_opening(gain, structure=st)
    min_px = max(1, math.ceil(min_area_ha / px_ha))
    loss = remove_small_patches(loss, min_px)
    gain = remove_small_patches(gain, min_px)

    forest_b = (prob_before >= threshold) & valid
    forest_a = (prob_after >= threshold) & valid

    cmap = np.full(valid.shape, NO_CHANGE, np.uint8)
    cmap[forest_b & forest_a] = STABLE_FOREST
    cmap[gain] = GAIN
    cmap[loss] = LOSS
    cmap[~valid] = INVALID

    fb = float(forest_b.sum() * px_ha)
    lo_ha = float(loss.sum() * px_ha)
    n_patches = int(ndi.label(loss, structure=np.ones((3, 3)))[1])
    stats = {
        "pixel_area_ha": px_ha,
        "valid_area_ha": float(valid.sum() * px_ha),
        "forest_before_ha": fb,
        "forest_after_ha": float(forest_a.sum() * px_ha),
        "forest_loss_ha": lo_ha,
        "forest_gain_ha": float(gain.sum() * px_ha),
        "loss_pct_of_forest_before": (100.0 * lo_ha / fb) if fb > 0 else 0.0,
        "loss_patches": n_patches,
        "threshold": threshold, "margin": margin, "min_area_ha": min_area_ha,
    }
    return ChangeResult(cmap, loss, gain, forest_b, forest_a, stats)


# --------------------------------------------------------------------------- #
# Output files
# --------------------------------------------------------------------------- #
def write_raster(path, array, scene: Scene, dtype="uint8", nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                       width=array.shape[1], count=1, dtype=dtype, crs=scene.crs,
                       transform=scene.transform, nodata=nodata, compress="lzw") as dst:
        dst.write(array.astype(dtype), 1)


def loss_to_geojson(loss: np.ndarray, scene: Scene, px_ha: float, path) -> int:
    """Vectorise loss patches (WGS84 GeoJSON, area_ha per polygon)."""
    labels, n = ndi.label(loss, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    feats = []
    if n:
        for geom, val in shapes(labels.astype(np.int32), mask=labels > 0,
                                transform=scene.transform, connectivity=8):
            if scene.crs is not None:
                geom = transform_geom(scene.crs, "EPSG:4326", geom)
            feats.append({"type": "Feature", "geometry": geom,
                          "properties": {"id": int(val),
                                         "area_ha": round(float(sizes[int(val)] * px_ha), 3)}})
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return len(feats)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _stretch(*band_stacks):
    """One shared 2-98 percentile stretch so dates are visually comparable."""
    vals = np.concatenate([b[[2, 1, 0]].reshape(3, -1)[:, ::50].ravel() for b in band_stacks])
    vals = vals[vals > 0]
    return tuple(np.percentile(vals, (2, 98))) if vals.size else (0.0, 1.0)


def _rgb(bands: np.ndarray, step: int, stretch=None) -> np.ndarray:
    rgb = bands[[2, 1, 0]][:, ::step, ::step].transpose(1, 2, 0)
    lo, hi = stretch if stretch else _stretch(bands)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def _overlay(rgb, mask, color, alpha):
    out = rgb.copy()
    for i, c in enumerate(color):
        out[..., i] = np.where(mask, (1 - alpha) * out[..., i] + alpha * c, out[..., i])
    return out


def plot_change(before: Scene, after: Scene, res: ChangeResult, out_png, max_dim=2000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    h, w = before.shape
    step = max(1, max(h, w) // max_dim)
    sl = (slice(None, None, step), slice(None, None, step))

    st = _stretch(before.bands, after.bands)
    rgb_b, rgb_a = _rgb(before.bands, step, st), _rgb(after.bands, step, st)
    stable = (res.change_map == STABLE_FOREST)[sl]
    loss, gain = res.loss[sl], res.gain[sl]

    ov = _overlay(rgb_a, stable, (0.1, 0.6, 0.2), 0.35)
    ov = _overlay(ov, gain, (0.1, 0.3, 0.9), 0.8)
    ov = _overlay(ov, loss, (0.9, 0.1, 0.1), 0.85)

    s = res.stats
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, img, title in zip(axes, (rgb_b, rgb_a, ov),
                              ("Before (true colour)", "After (true colour)",
                               "Detected forest change")):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    axes[2].legend(handles=[Patch(color=(0.9, 0.1, 0.1), label="Forest loss"),
                            Patch(color=(0.1, 0.3, 0.9), label="Forest gain"),
                            Patch(color=(0.1, 0.6, 0.2), alpha=0.5, label="Stable forest")],
                   loc="lower right", framealpha=0.9)
    fig.suptitle(
        f"Forest loss: {s['forest_loss_ha']:,.1f} ha "
        f"({s['loss_pct_of_forest_before']:.2f}% of forest before) | "
        f"{s['loss_patches']} patches | forest before {s['forest_before_ha']:,.0f} ha "
        f"-> after {s['forest_after_ha']:,.0f} ha",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_single(scene: Scene, forest: np.ndarray, px_ha: float, out_png, max_dim=2000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = scene.shape
    step = max(1, max(h, w) // max_dim)
    sl = (slice(None, None, step), slice(None, None, step))
    rgb = _rgb(scene.bands, step)
    ov = _overlay(rgb, forest[sl], (0.1, 0.6, 0.2), 0.45)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    axes[0].imshow(rgb, interpolation="nearest"); axes[0].set_title("True colour")
    axes[1].imshow(ov, interpolation="nearest")
    axes[1].set_title(f"Forest mask ({forest.sum() * px_ha:,.0f} ha)")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def run_change_pipeline(before_path, after_path, out_dir, predictor,
                        threshold=0.5, margin=0.1, min_area_ha=0.5,
                        mask_clouds=True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    before = read_scene(before_path)
    after = read_scene(after_path, reference=before)   # warped onto `before` grid
    px_ha = pixel_area_ha(before)

    valid = before.valid & after.valid
    if mask_clouds:
        valid &= ~simple_cloud_mask(before.bands) & ~simple_cloud_mask(after.bands)

    p_before = predictor.predict(compute_features(before.bands))
    p_after = predictor.predict(compute_features(after.bands))

    res = detect_change(p_before, p_after, valid, px_ha, threshold, margin, min_area_ha)

    write_raster(out / "change_map.tif", res.change_map, before, nodata=INVALID)
    write_raster(out / "forest_prob_before.tif", p_before, before, "float32")
    write_raster(out / "forest_prob_after.tif", p_after, after, "float32")
    n_poly = loss_to_geojson(res.loss, before, px_ha, out / "forest_loss.geojson")
    plot_change(before, after, res, out / "change_map.png")

    res.stats["polygons"] = n_poly
    (out / "stats.json").write_text(json.dumps(res.stats, indent=2))
    return res.stats


def run_single_pipeline(path, out_dir, predictor, threshold=0.5) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = read_scene(path)
    px_ha = pixel_area_ha(scene)
    prob = predictor.predict(compute_features(scene.bands))
    forest = (prob >= threshold) & scene.valid
    write_raster(out / "forest_mask.tif", forest.astype(np.uint8), scene, nodata=255)
    plot_single(scene, forest, px_ha, out / "forest_mask.png")
    stats = {"forest_ha": float(forest.sum() * px_ha), "pixel_area_ha": px_ha}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats