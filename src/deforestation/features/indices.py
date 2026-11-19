"""Spectral indices from Sentinel-2 reflectance arrays (any shape, float)."""
from __future__ import annotations

import numpy as np


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype("float32")
    b = b.astype("float32")
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom == 0, np.nan, (a - b) / denom)


def ndvi(nir, red):
    return _nd(nir, red)


def nbr(nir, swir2):
    return _nd(nir, swir2)


def ndmi(nir, swir1):
    return _nd(nir, swir1)
