# tests/test_seed.py
import random

import numpy as np
import pytest

from river_water_quality.utils.seed import set_seed


def test_seed_reproducibility():
    set_seed(42)
    a = random.random()

    set_seed(42)
    b = random.random()

    assert a == b


def test_different_seeds_differ():
    set_seed(1)
    a = random.random()

    set_seed(2)
    b = random.random()

    assert a != b


def test_numpy_reproducibility():
    set_seed(42)
    first = np.random.random(5)
    set_seed(42)
    np.testing.assert_array_equal(first, np.random.random(5))


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        set_seed(seed)
