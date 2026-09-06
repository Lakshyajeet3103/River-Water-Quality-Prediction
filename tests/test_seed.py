# tests/test_seed.py
import random

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