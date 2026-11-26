"""Create a synthetic before/after Sentinel-2-like pair with known clearings
(80 ha + 36 ha = 116 ha) so you can sanity-check the pipeline.

    python make_demo_data.py && python run_detection.py --before demo_before.tif --after demo_after.tif --model ndvi
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin

rng = np.random.default_rng(0)
H = W = 512
FOREST = [0.03, 0.05, 0.03, 0.35, 0.15, 0.07]   # B2 B3 B4 B8 B11 B12
CLEARED = [0.08, 0.11, 0.14, 0.22, 0.30, 0.25]
WATER = [0.04, 0.05, 0.03, 0.02, 0.01, 0.01]


def scene(cleared_boxes):
    img = np.empty((6, H, W), np.float32)
    for i in range(6):
        img[i] = FOREST[i]
    img[:, 400:480, 50:150] = np.array(WATER)[:, None, None]       # lake
    for (y0, y1, x0, x1) in cleared_boxes:
        img[:, y0:y1, x0:x1] = np.array(CLEARED)[:, None, None]
    img *= rng.normal(1.0, 0.04, img.shape).astype(np.float32)
    return (np.clip(img, 0, 1) * 10000).astype(np.uint16)


def save(path, arr):
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=6,
                       dtype="uint16", crs="EPSG:32648",
                       transform=from_origin(500000, 1300000, 10, 10)) as dst:
        dst.write(arr)


save("demo_before.tif", scene([]))
save("demo_after.tif", scene([(100, 200, 100, 180), (300, 360, 300, 360)]))
print("wrote demo_before.tif, demo_after.tif (expected loss = 116 ha)")