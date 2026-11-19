import numpy as np

from deforestation import config
from deforestation.features.labels import FOREST, IGNORE, NONFOREST, label_rule
from deforestation.features.sampling import block_ids

TREE, CROP = 10, 40


def _one(tc, loss_year, wc, year):
    ly = 0 if loss_year == 0 else loss_year - 2000
    return int(label_rule(np.array([tc]), np.array([ly]), np.array([wc]), year)[0])


def test_intact_forest_is_forest_every_year():
    for y in config.COMPOSITE_YEARS:
        assert _one(90, 0, TREE, y) == FOREST


def test_pixel_cut_in_2022():
    assert _one(90, 2022, TREE, 2019) == FOREST     # still standing at end of 2019
    assert _one(90, 2022, TREE, 2021) == IGNORE     # cut in Y+1, may be inside the image window
    assert _one(90, 2022, TREE, 2023) == NONFOREST  # cleared one year before
    assert _one(90, 2022, TREE, 2025) == NONFOREST  # cleared three years before


def test_old_loss_is_ignored():
    assert _one(90, 2015, CROP, 2021) == IGNORE      # may have regrown / become plantation


def test_never_forest():
    assert _one(3, 0, CROP, 2023) == NONFOREST


def test_disagreement_is_ignored():
    assert _one(90, 0, CROP, 2021) == IGNORE         # Hansen forest, WorldCover crop
    assert _one(40, 0, TREE, 2021) == IGNORE         # medium tree cover


def test_season_window():
    assert config.season_dates(2023) == ("2023-11-01", "2024-02-28")


def test_block_ids_group_neighbours():
    size = config.BLOCK_SIZE_M // config.GRID_RES
    ids = block_ids(np.array([0, size - 1, size]), np.array([0, 0, 0]), width=10 * size)
    assert ids[0] == ids[1] != ids[2]
