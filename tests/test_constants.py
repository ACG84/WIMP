"""Tests for wimp.constants."""

import math

from wimp.constants import (
    GAMMA_NV, D0, D0_TEMP_COEFF, G_NV,
    MU0, HBAR, MU_B, K_B,
    CANONICAL_BODY_LENGTH, LONGITUDINAL_VARIABILITY, TRANSVERSE_VARIABILITY,
    tesla_to_mt, mt_to_tesla,
    hz_to_mhz, mhz_to_hz,
    hz_to_ghz, ghz_to_hz,
    meters_to_um, um_to_meters,
    seconds_to_us, us_to_seconds,
)


class TestPhysicalConstants:
    def test_gamma_nv_positive(self):
        assert GAMMA_NV > 0

    def test_gamma_nv_order_of_magnitude(self):
        assert 1e9 < GAMMA_NV < 1e11

    def test_d0_zero_field_splitting(self):
        assert 2.8e9 < D0 < 2.9e9

    def test_d0_temp_coeff_negative(self):
        assert D0_TEMP_COEFF < 0

    def test_g_factor(self):
        assert abs(G_NV - 2.0028) < 0.001

    def test_mu0(self):
        expected = 4.0 * math.pi * 1e-7
        assert abs(MU0 - expected) < 1e-20

    def test_hbar_positive(self):
        assert HBAR > 0
        assert 1e-35 < HBAR < 1e-33

    def test_mu_b_positive(self):
        assert MU_B > 0

    def test_k_b_positive(self):
        assert K_B > 0


class TestBiologyConstants:
    def test_body_length(self):
        assert CANONICAL_BODY_LENGTH == 1.0e-3

    def test_longitudinal_variability(self):
        assert 0 < LONGITUDINAL_VARIABILITY < 1

    def test_transverse_variability_positive(self):
        assert TRANSVERSE_VARIABILITY > 0


class TestUnitConversions:
    def test_tesla_mt_roundtrip(self):
        assert mt_to_tesla(tesla_to_mt(1.5)) == pytest.approx(1.5)

    def test_hz_mhz_roundtrip(self):
        assert mhz_to_hz(hz_to_mhz(1e7)) == pytest.approx(1e7)

    def test_hz_ghz_roundtrip(self):
        assert ghz_to_hz(hz_to_ghz(3e9)) == pytest.approx(3e9)

    def test_meters_um_roundtrip(self):
        assert um_to_meters(meters_to_um(1e-4)) == pytest.approx(1e-4)

    def test_seconds_us_roundtrip(self):
        assert us_to_seconds(seconds_to_us(1e-6)) == pytest.approx(1e-6)

    def test_tesla_to_mt_value(self):
        assert tesla_to_mt(1.0) == pytest.approx(1000.0)

    def test_hz_to_mhz_value(self):
        assert hz_to_mhz(1e6) == pytest.approx(1.0)

    def test_meters_to_um_value(self):
        assert meters_to_um(1e-6) == pytest.approx(1.0)

    def test_seconds_to_us_value(self):
        assert seconds_to_us(1e-6) == pytest.approx(1.0)


import pytest
