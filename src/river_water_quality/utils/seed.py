"""Reproducibility helpers."""

import random


def set_seed(seed: int) -> None:
    """Seed Python's standard random generator.

    Framework-specific seeding will be added when model dependencies are introduced.
    """
    random.seed(seed)
