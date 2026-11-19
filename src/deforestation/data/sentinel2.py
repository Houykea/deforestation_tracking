"""Cloud-free dry-season Sentinel-2 composites from the public Earth Search catalog.

Steps per year:
  1. Search scenes over the AOI in the season window with scene cloud < S2_MAX_SCENE_CLOUD.
  2. Keep the clearest S2_SCENES_PER_TILE scenes per Sentinel-2 tile.
  3. Read only the AOI part of each scene at GRID_RES on the shared UTM grid.
  4. Mask clouds/shadows with the SCL band, convert to reflectance (0-1),
     correcting the +1000 offset of processing baseline >= 04.00 per scene.
  5. Per-pixel median over the clear observations.

Output: uint16 GeoTIFF, reflectance x 10000, nodata 0, one band per S2_BANDS entry,
plus a last band "n_clear" = number of clear observations used.
Processed window by window so memory stays at a few hundred MB.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox
from rasterio.windows import Window

from deforestation import config
from deforestation.data.aoi import BBox


def grid_for(bbox: BBox) -> GeoBox:
    """The shared output grid: AOI bbox in UTM 48N at GRID_RES metres."""
    bb = BoundingBox(*bbox.bounds(), crs="EPSG:4326").to_crs(config.GRID_CRS)
    return GeoBox.from_bbox(bb, crs=config.GRID_CRS, resolution=config.GRID_RES)


def search_scenes(bbox: BBox, year: int) -> list:
    from pystac_client import Client

    start, end = config.season_dates(year)
    items = Client.open(config.STAC_URL).search(
        collections=[config.S2_COLLECTION],
        bbox=bbox.bounds(),
        datetime=f"{start}T00:00:00Z/{end}T23:59:59Z",
        query={"eo:cloud_cover": {"lt": config.S2_MAX_SCENE_CLOUD}},
    ).item_collection()

    by_tile = defaultdict(list)
    for it in items:
        by_tile[it.properties.get("grid:code", it.id.split("_")[1])].append(it)
    picked = []
    for tile_items in by_tile.values():
        tile_items.sort(key=lambda it: it.properties["eo:cloud_cover"])
        picked += tile_items[: config.S2_SCENES_PER_TILE]
    return picked


def scene_offset(item) -> float:
    """Reflectance offset to add after scaling by 1e-4 (baseline 04.00+ stores DN + 1000)."""
    baseline = float(item.properties.get("s2:processing_baseline", "0") or 0)
    applied = bool(item.properties.get("earthsearch:boa_offset_applied", False))
    return -0.1 if baseline >= 4.0 and not applied else 0.0


def _configure_gdal() -> None:
    import os

    from odc.stac import configure_rio

    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})


def _load(items: list, gbox: GeoBox):
    from odc.stac import load

    return load(
        items,
        bands=[*config.S2_BANDS, "scl"],
        geobox=gbox,
        groupby="id",  # one time slice per scene, so the offset can be applied per scene
        chunks={"x": config.GRID_WINDOW, "y": config.GRID_WINDOW},
        resampling={"scl": "nearest", "*": "average"},
        fail_on_error=False,  # skip a broken scene rather than abort the year
    )


def build_composite(bbox: BBox, year: int, dest: Path, log=print) -> dict:
    """Build and write the composite for one year. Returns a small summary."""
    _configure_gdal()
    items = search_scenes(bbox, year)
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {year}.")
    log(f"  {len(items)} scenes selected "
        f"({len({i.properties.get('grid:code') for i in items})} tiles)")

    gbox = grid_for(bbox)
    # Load scenes in groups that share the same offset, so no per-slice bookkeeping is needed.
    groups = defaultdict(list)
    for it in items:
        groups[scene_offset(it)].append(it)
    loaded = [(off, _load(group, gbox)) for off, group in groups.items()]
    log("  offsets: " + ", ".join(f"{len(g)} scenes {off:+.1f}" for off, g in groups.items()))

    bands = list(config.S2_BANDS.values())
    h, w = gbox.shape
    profile = {
        "driver": "GTiff", "dtype": "uint16", "count": len(bands) + 1, "nodata": 0,
        "height": h, "width": w, "crs": config.GRID_CRS, "transform": gbox.affine,
        "compress": "deflate", "predictor": 2, "tiled": True, "blockxsize": 512, "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part.tif")
    step = config.GRID_WINDOW
    windows = [(r, c) for r in range(0, h, step) for c in range(0, w, step)]

    with rasterio.open(tmp, "w", **profile) as dst:
        for k, (r, c) in enumerate(windows, 1):
            win = dict(y=slice(r, r + step), x=slice(c, c + step))
            refl_stack = {asset: [] for asset in config.S2_BANDS}
            clear_stack = []
            for off, ds in loaded:
                sub = ds.isel(**win).compute()
                clear = np.isin(sub["scl"].values, config.S2_SCL_CLEAR)
                clear_stack.append(clear)
                for asset in config.S2_BANDS:
                    dn = sub[asset].values.astype("float32")
                    refl = dn * 1e-4 + off
                    refl[~clear | (dn == 0)] = np.nan
                    refl_stack[asset].append(refl)
            n_clear = np.concatenate(clear_stack).sum(axis=0)
            out = np.zeros((len(bands) + 1, *n_clear.shape), dtype="uint16")
            for b, asset in enumerate(config.S2_BANDS):
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
                    med = np.nanmedian(np.concatenate(refl_stack[asset]), axis=0)
                out[b] = np.where(np.isnan(med), 0, np.clip(med, 0.0001, 6.5) * 10_000).astype("uint16")
            out[-1] = np.minimum(n_clear, 65535)
            dst.write(out, window=Window(c, r, out.shape[2], out.shape[1]))
            log(f"\r  window {k}/{len(windows)}", end="", flush=True)
        for i, name in enumerate([*bands, "n_clear"], 1):
            dst.set_band_description(i, name)
        start, end = config.season_dates(year)
        dst.update_tags(year=year, season=f"{start}/{end}", scenes=len(items),
                        scale="reflectance x 10000", source=config.STAC_URL)
    log("")
    tmp.replace(dest)

    with rasterio.open(dest) as src:
        n = src.read(src.count)
    return {"scenes": len(items), "valid_pct": float((n > 0).mean() * 100),
            "median_clear_obs": float(np.median(n[n > 0])) if (n > 0).any() else 0.0}
