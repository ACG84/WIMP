"""Shared fixtures for the WIMP test suite."""

import numpy as np
import pytest

from wimp.io import WIMPDataset


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def tau_ramsey():
    """Typical Ramsey tau array (0 to 5 us)."""
    return np.linspace(0, 5e-6, 200)


@pytest.fixture
def tau_echo():
    """Typical echo tau array (0 to 300 us)."""
    return np.linspace(0, 300e-6, 150)


@pytest.fixture
def tau_t1():
    """Typical T1 tau array (0 to 25 ms)."""
    return np.linspace(0, 25e-3, 150)


@pytest.fixture
def nd_positions():
    """Simple ND positions along a worm body."""
    n = 8
    x = np.linspace(0.1e-3, 0.9e-3, n)
    return np.column_stack([x, np.zeros(n), np.full(n, 5e-6)])


@pytest.fixture
def neuron_positions():
    """Simple neuron positions along a worm body."""
    n = 4
    x = np.linspace(0.15e-3, 0.85e-3, n)
    return np.column_stack([x, np.zeros(n), np.zeros(n)])


@pytest.fixture
def sample_dataset(tau_ramsey, rng):
    """A minimal WIMPDataset for testing."""
    n_nds = 3
    signal = rng.normal(0.5, 0.1, size=(n_nds, len(tau_ramsey)))
    return WIMPDataset(
        protocol="ramsey",
        tau_array=tau_ramsey,
        signal=signal,
        metadata={"test": True},
    )
