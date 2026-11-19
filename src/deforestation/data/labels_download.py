"""Download public label rasters (ESA WorldCover, Hansen GFC) and clip them to the AOI.

Full tiles go to data/raw/ (kept so re-runs skip the download).
Clipped rasters go to data/labels/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import rasterio
import requests
from rasterio.merge import merge

from deforestation import config
from deforestation.data.aoi import BBox


def download(url: str, dest: Path, chunk: int = 1 << 20, retries: int = 8) -> Path:
    """Stream a file to disk, resuming from a .part file after dropped connections."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = int(requests.head(url, timeout=60).headers.get("Content-Length", 0))
    if dest.exists() and size and dest.stat().st_size == size:
        print(f"  exists  {dest.name}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        if size and done >= size:
            break
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                r.raise_for_status()
                if done and r.status_code != 206:  # server ignored Range: restart
                    done = 0
                with open(tmp, "ab" if done else "wb") as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
                        done += len(block)
                        if size:
                            print(f"\r  {dest.name}: {done / size:6.1%}", end="", flush=True)
            print()
            break
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError,
                requests.Timeout) as exc:
            print(f"\n  connection dropped (attempt {attempt}/{retries}), resuming: {exc.__class__.__name__}")
            time.sleep(min(2 ** attempt, 30))
    else:
        raise IOError(f"Download failed after {retries} attempts: {url}")

    if size and tmp.stat().st_size != size:
        raise IOError(f"Incomplete download for {url}")
    tmp.replace(dest)
    return dest


def clip(src_paths: list[Path], dest: Path, bbox: BBox) -> Path:
    """Mosaic one or more WGS84 rasters, crop to the bbox, save as compressed GeoTIFF."""
    srcs = [rasterio.open(p) for p in src_paths]
    try:
        # Keep each source's own nodata. Hansen has none: 0 means "no loss" / "0% cover".
        nodata = srcs[0].nodata
        data, transform = merge(srcs, bounds=bbox.bounds(), nodata=nodata)
        profile = srcs[0].profile | {
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
            "nodata": nodata,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }
    finally:
        for s in srcs:
            s.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(data)
    return dest


def download_boundary(dest: Path = config.KRATIE_BOUNDARY) -> Path:
    """Save the official province polygon (geoBoundaries ADM1) as GeoJSON."""
    if dest.exists():
        return dest
    meta = requests.get(config.BOUNDARY_API, timeout=60).json()
    gj = requests.get(meta["gjDownloadURL"], timeout=300).json()
    feats = [f for f in gj["features"] if f["properties"]["shapeName"] == config.PROVINCE_NAME]
    if len(feats) != 1:
        raise ValueError(f"Expected one '{config.PROVINCE_NAME}' feature, found {len(feats)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return dest


def load_boundary(path: Path = config.KRATIE_BOUNDARY) -> dict:
    return json.loads(path.read_text())["features"][0]["geometry"]


def download_labels(bbox: BBox) -> dict[str, Path]:
    sources = {
        f"worldcover_{config.WORLDCOVER_YEAR}": [config.worldcover_url(t) for t in config.WORLDCOVER_TILES]
    }
    sources |= {f"hansen_{layer}": [config.hansen_url(layer)] for layer in config.HANSEN_LAYERS}

    print("[boundary]")
    print(f"  saved -> {download_boundary().relative_to(config.ROOT)}")

    out: dict[str, Path] = {}
    for name, urls in sources.items():
        print(f"[{name}]")
        raws = [download(u, config.RAW_DIR / Path(u).name) for u in urls]
        out[name] = clip(raws, config.LABELS_DIR / f"{name}_kratie.tif", bbox)
        print(f"  clipped -> {out[name].relative_to(config.ROOT)}")
    return out
