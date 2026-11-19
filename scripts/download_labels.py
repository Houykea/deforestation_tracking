"""Download WorldCover + Hansen label rasters and the Kratie boundary, clip to Kratie.

Usage:  python scripts/download_labels.py
Statistics are computed only for pixels inside the official province polygon.
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.features import geometry_mask

from deforestation import config
from deforestation.data.aoi import kratie
from deforestation.data.labels_download import download_labels, load_boundary

WORLDCOVER_CLASSES = {0: "no data", 10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
                      60: "bare", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}


def summarize(name: str, path, boundary: dict) -> None:
    with rasterio.open(path) as src:
        a = src.read(1)
        inside = ~geometry_mask([boundary], a.shape, src.transform)
        bounds = src.bounds
        transform = src.transform
    v = a[inside]
    print(f"\n{name}: {a.shape[1]}x{a.shape[0]} px, bounds lon {bounds.left:.3f}-{bounds.right:.3f}, "
          f"lat {bounds.bottom:.3f}-{bounds.top:.3f}; {v.size:,} px inside Kratie")
    if name.startswith("worldcover"):
        print(f"  coverage inside Kratie: {np.mean(v != 0):.2%} of pixels have data")
        vals, counts = np.unique(v, return_counts=True)
        for val, c in zip(vals, counts):
            print(f"  {WORLDCOVER_CLASSES.get(int(val), val)!s:<9} {c / v.size:6.1%}")
    elif name.endswith("lossyear"):
        print(f"  share of Kratie with loss 2001-{2000 + int(v.max())}: {np.mean(v > 0):.1%}")
        # Pixel area from latitude: 1 degree ~ 111,320 m; longitude shrinks with cos(lat).
        rows = np.arange(a.shape[0])
        lat = transform.f + (rows + 0.5) * transform.e
        px_ha = (abs(transform.e) * 111_320) * (transform.a * 111_320 * np.cos(np.radians(lat))) / 10_000
        area = np.broadcast_to(px_ha[:, None], a.shape)[inside]
        print(f"  Kratie area: {area.sum():,.0f} ha")
        for y in range(config.START_YEAR - 2000, int(v.max()) + 1):
            hit = v == y
            print(f"  {2000 + y}: {int(hit.sum()):>8,} px  ~{area[hit].sum():>8,.0f} ha")
    else:
        print(f"  mean tree cover in 2000: {v.mean():.1f}%")


def main() -> None:
    paths = download_labels(kratie())
    boundary = load_boundary()
    for name, path in paths.items():
        summarize(name, path, boundary)


if __name__ == "__main__":
    main()
