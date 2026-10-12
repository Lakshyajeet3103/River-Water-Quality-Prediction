"""Reproducibility helpers."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy's legacy RNG; pass seed explicitly to default_rng.

    This does not guarantee deterministic future frameworks or hardware kernels.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    random.seed(seed)
    np.random.seed(seed)
