"""Physical constants for NV-centre diamond magnetometry.

All values are in SI units unless noted otherwise.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# NV-centre parameters
# ---------------------------------------------------------------------------

GAMMA_NV: float = 28.024e9
"""NV gyromagnetic ratio (Hz / T)."""

D0: float = 2.870e9
"""Zero-field splitting at room temperature (Hz)."""

D0_TEMP_COEFF: float = -74e-6 * 1e9
"""Temperature coefficient of D0 (Hz / K)."""

G_NV: float = 2.0028
"""Electron g-factor for the NV ground state."""

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------

MU0: float = 4.0 * math.pi * 1e-7
"""Vacuum permeability (T m / A)."""

HBAR: float = 1.054571817e-34
"""Reduced Planck constant (J s)."""

MU_B: float = 9.2740100783e-24
"""Bohr magneton (J / T)."""

K_B: float = 1.380649e-23
"""Boltzmann constant (J / K)."""

# ---------------------------------------------------------------------------
# C. elegans morphology defaults
# ---------------------------------------------------------------------------

CANONICAL_BODY_LENGTH: float = 1.0e-3
"""Canonical adult hermaphrodite body length (m)."""

LONGITUDINAL_VARIABILITY: float = 0.04
"""Fractional longitudinal variability in neuron position."""

TRANSVERSE_VARIABILITY: float = 3.0e-6
"""Transverse positional variability (m)."""

# ---------------------------------------------------------------------------
# Unit-conversion helpers
# ---------------------------------------------------------------------------


def tesla_to_mt(val: float) -> float:
    """Convert Tesla to millitesla."""
    return val * 1e3


def mt_to_tesla(val: float) -> float:
    """Convert millitesla to Tesla."""
    return val * 1e-3


def hz_to_mhz(val: float) -> float:
    """Convert Hz to MHz."""
    return val * 1e-6


def mhz_to_hz(val: float) -> float:
    """Convert MHz to Hz."""
    return val * 1e6


def hz_to_ghz(val: float) -> float:
    """Convert Hz to GHz."""
    return val * 1e-9


def ghz_to_hz(val: float) -> float:
    """Convert GHz to Hz."""
    return val * 1e9


def meters_to_um(val: float) -> float:
    """Convert metres to micrometres."""
    return val * 1e6


def um_to_meters(val: float) -> float:
    """Convert micrometres to metres."""
    return val * 1e-6


def seconds_to_us(val: float) -> float:
    """Convert seconds to microseconds."""
    return val * 1e6


def us_to_seconds(val: float) -> float:
    """Convert microseconds to seconds."""
    return val * 1e-6
