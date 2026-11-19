"""Area of interest and year validation. Pure Python, no Earth Engine needed to test."""
from __future__ import annotations

from dataclasses import dataclass

from deforestation import config

# Rough limits of Cambodia, to catch swapped lon/lat or wrong-country boxes.
_CAMBODIA = (102.3, 10.4, 107.7, 14.7)


@dataclass(frozen=True)
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def validate(self) -> "BBox":
        if self.min_lon >= self.max_lon or self.min_lat >= self.max_lat:
            raise ValueError("BBox min values must be smaller than max values (order: lon, lat).")
        w, s, e, n = _CAMBODIA
        if not (w <= self.min_lon and self.max_lon <= e and s <= self.min_lat and self.max_lat <= n):
            raise ValueError("BBox lies outside Cambodia. Check lon/lat order.")
        return self

    def bounds(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


def kratie() -> BBox:
    return BBox(*config.KRATIE_BBOX).validate()


def validate_years(start: int, end: int) -> tuple[int, int]:
    if start > end:
        raise ValueError("start year must be <= end year.")
    if start < config.START_YEAR:
        raise ValueError(f"Project period starts in {config.START_YEAR}.")
    if end > config.END_YEAR:
        raise ValueError(f"end year cannot be later than {config.END_YEAR}.")
    return start, end
