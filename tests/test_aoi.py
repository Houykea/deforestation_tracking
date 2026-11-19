import pytest

from deforestation.data.aoi import BBox, kratie, validate_years


def test_kratie_bbox_is_valid():
    assert kratie().min_lon < kratie().max_lon


def test_swapped_lon_lat_rejected():
    with pytest.raises(ValueError):
        BBox(12.0, 105.7, 13.2, 106.9).validate()


def test_inverted_bbox_rejected():
    with pytest.raises(ValueError):
        BBox(106.9, 12.0, 105.7, 13.2).validate()


def test_years():
    assert validate_years(2019, 2025) == (2019, 2025)
    with pytest.raises(ValueError):
        validate_years(2018, 2020)
    with pytest.raises(ValueError):
        validate_years(2019, 2026)
    with pytest.raises(ValueError):
        validate_years(2022, 2020)


def test_indices():
    import numpy as np

    from deforestation.features.indices import ndvi

    out = ndvi(np.array([0.4, 0.0]), np.array([0.1, 0.0]))
    assert out[0] == pytest.approx(0.6)
    assert np.isnan(out[1])
