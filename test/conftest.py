"""Shared pytest fixtures for the test suite."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed NumPy's global RNG before every test so random-data tests are deterministic."""
    np.random.seed(0)
