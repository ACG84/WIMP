"""Tests for wimp.calibration."""

import numpy as np
import pytest

from wimp.constants import D0, D0_TEMP_COEFF, GAMMA_NV
from wimp.calibration import (
    characterize_nv,
    characterize_nv_cw,
    calibrate_with_known_field,
    temperature_correction,
    correct_field_for_temperature,
    cross_validate_hall,
    noise_floor_estimate,
    linearity_check,
)


class TestCharacterizeNV:
    def test_basic(self):
        from wimp.synthetic import generate_ramsey_data, generate_echo_data, generate_t1_data
        tau_r = np.linspace(0, 5e-6, 200)
        tau_e = np.linspace(0, 300e-6, 150)
        tau_t = np.linspace(0, 25e-3, 150)

        r_data = generate_ramsey_data(tau_r, 50e-6, 1e-6, snr=100, seed=1)
        e_data = generate_echo_data(tau_e, 100e-6, snr=100, seed=2)
        t_data = generate_t1_data(tau_t, 5e-3, snr=100, seed=3)

        result = characterize_nv(
            tau_r, r_data["signal"],
            tau_e, e_data["signal"],
            tau_t, t_data["signal"],
        )
        assert "t2star" in result
        assert "t2" in result
        assert "t1" in result
        assert "dc_sensitivity" in result
        assert "ac_sensitivity" in result
        assert result["dc_sensitivity"] > 0
        assert result["ac_sensitivity"] > 0


class TestCalibrateWithKnownField:
    def test_perfect_calibration(self):
        applied = np.array([0, 1e-6, 2e-6, 3e-6, 4e-6])
        measured = applied * 1.0 + 0.0  # perfect scale=1, offset=0
        cal = calibrate_with_known_field(measured, applied)
        assert cal["scale"] == pytest.approx(1.0, abs=1e-10)
        assert cal["r_squared"] == pytest.approx(1.0, abs=1e-10)

    def test_with_offset(self):
        applied = np.array([0, 1e-6, 2e-6, 3e-6, 4e-6])
        measured = applied * 1.05 + 1e-7
        cal = calibrate_with_known_field(measured, applied)
        assert cal["scale"] == pytest.approx(1.05, rel=0.01)
        assert cal["offset"] == pytest.approx(1e-7, rel=0.01)


class TestTemperatureCorrection:
    def test_no_shift(self):
        delta_T = temperature_correction(D0, D0)
        assert delta_T == pytest.approx(0.0)

    def test_positive_shift(self):
        # D0 decreases with temperature (D0_TEMP_COEFF < 0)
        # So lower D0 → higher temperature → positive delta_T
        d0_measured = D0 + D0_TEMP_COEFF * 10  # 10 K warmer
        delta_T = temperature_correction(d0_measured)
        assert delta_T == pytest.approx(10.0, rel=0.01)


class TestCorrectFieldForTemperature:
    def test_no_correction_at_ref(self):
        field = np.array([1e-6, 2e-6])
        corrected = correct_field_for_temperature(field, D0)
        np.testing.assert_allclose(corrected, field)

    def test_correction_applied(self):
        field = np.array([1e-6])
        d0_shifted = D0 + 1e6  # 1 MHz shift
        corrected = correct_field_for_temperature(field, d0_shifted)
        expected_spurious = 1e6 / GAMMA_NV
        np.testing.assert_allclose(corrected, field - expected_spurious)


class TestCrossValidateHall:
    def test_perfect_agreement(self):
        rng = np.random.default_rng(0)
        hall = np.linspace(0, 1e-3, 50)
        nv = hall.copy()
        result = cross_validate_hall(nv, hall)
        assert result["slope"] == pytest.approx(1.0, abs=1e-10)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-10)
        assert result["rms_residual"] == pytest.approx(0.0, abs=1e-10)

    def test_output_keys(self):
        hall = np.array([1e-6, 2e-6, 3e-6])
        nv = np.array([1.1e-6, 2.1e-6, 3.1e-6])
        result = cross_validate_hall(nv, hall)
        assert "slope" in result
        assert "offset" in result
        assert "r_squared" in result
        assert "residuals" in result
        assert "rms_residual" in result


class TestNoiseFloorEstimate:
    def test_basic(self):
        rng = np.random.default_rng(0)
        bg = rng.normal(0, 1e-9, size=10000)
        result = noise_floor_estimate(bg, fs=1000.0)
        assert "noise_rms" in result
        assert "noise_psd" in result
        assert "freq" in result
        assert result["noise_rms"] > 0


class TestLinearityCheck:
    def test_perfect_linearity(self):
        fields = np.linspace(0, 1e-3, 20)
        responses = 2.0 * fields + 0.1
        result = linearity_check(fields, responses)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-10)
        assert result["slope"] == pytest.approx(2.0, rel=0.01)

    def test_output_keys(self):
        fields = np.array([0, 1, 2, 3, 4.0])
        responses = np.array([0.1, 1.1, 2.0, 3.1, 4.0])
        result = linearity_check(fields, responses)
        assert "slope" in result
        assert "r_squared" in result
        assert "max_deviation" in result
        assert "linear_range" in result
        assert len(result["linear_range"]) == 2


# ---------------------------------------------------------------------------
# Time-varying temperature correction
# ---------------------------------------------------------------------------

from wimp.calibration import temperature_series_correction, estimate_temperature_from_ramsey


class TestTemperatureSeriesCorrection:
    def test_output_keys(self):
        field = np.ones(100) * 1e-6
        d0_ts = np.ones(100) * D0
        result = temperature_series_correction(field, d0_ts)
        assert "corrected_field" in result
        assert "temperature_shifts" in result
        assert "spurious_field" in result

    def test_no_shift_at_ref(self):
        field = np.ones(50) * 1e-6
        d0_ts = np.ones(50) * D0
        result = temperature_series_correction(field, d0_ts)
        np.testing.assert_allclose(result["corrected_field"], field)
        np.testing.assert_allclose(result["temperature_shifts"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["spurious_field"], 0.0, atol=1e-10)

    def test_correction_applied(self):
        field = np.ones(50) * 1e-6
        delta_d0 = 1e6  # 1 MHz shift
        d0_ts = np.ones(50) * (D0 + delta_d0)
        result = temperature_series_correction(field, d0_ts)
        expected_spurious = delta_d0 / GAMMA_NV
        np.testing.assert_allclose(
            result["corrected_field"], field - expected_spurious
        )

    def test_temperature_shift_sign(self):
        field = np.ones(50) * 1e-6
        # D0_TEMP_COEFF is negative, so +delta_d0 -> negative dT
        delta_d0 = 1e6
        d0_ts = np.ones(50) * (D0 + delta_d0)
        result = temperature_series_correction(field, d0_ts)
        # delta_d0 > 0 and D0_TEMP_COEFF < 0 -> temp_shifts < 0
        assert np.all(result["temperature_shifts"] < 0)


class TestEstimateTemperatureFromRamsey:
    def test_no_shift_at_d0(self):
        # If oscillation frequency == 0 and B == 0, delta_T should be 0
        delta_T = estimate_temperature_from_ramsey(0.0, b_known=0.0)
        assert delta_T == pytest.approx(0.0, abs=1e-6)

    def test_known_shift(self):
        # Apply a 10 K shift: oscillation freq = D0_TEMP_COEFF * 10
        freq = D0_TEMP_COEFF * 10
        delta_T = estimate_temperature_from_ramsey(freq, b_known=0.0)
        assert delta_T == pytest.approx(10.0, rel=0.01)

    def test_with_known_field(self):
        b = 1e-6  # 1 uT field
        freq = GAMMA_NV * b  # no temp shift, just Zeeman contribution
        delta_T = estimate_temperature_from_ramsey(freq, b_known=b)
        assert delta_T == pytest.approx(0.0, abs=1e-6)


class TestCharacterizeNVCW:
    def test_returns_expected_keys(self):
        from wimp.relaxation import odmr_model

        b = 0.5e-3
        f_minus = D0 - GAMMA_NV * b
        f_plus = D0 + GAMMA_NV * b
        freq = np.linspace(2.8e9, 2.94e9, 500)
        signal = odmr_model(freq, 1.0, f_minus, f_plus, 0.03, 0.03, 5e6, 5e6)
        rng = np.random.default_rng(55)
        signal += rng.normal(0, 0.001, signal.shape)

        result = characterize_nv_cw(freq, signal)
        for key in ("fit", "linewidth", "contrast", "t2star_estimated",
                     "b_field", "cw_sensitivity", "center_freq", "temperature_shift"):
            assert key in result

    def test_sensitivity_positive(self):
        from wimp.relaxation import odmr_model

        b = 0.5e-3
        f_minus = D0 - GAMMA_NV * b
        f_plus = D0 + GAMMA_NV * b
        freq = np.linspace(2.8e9, 2.94e9, 500)
        signal = odmr_model(freq, 1.0, f_minus, f_plus, 0.03, 0.03, 5e6, 5e6)
        rng = np.random.default_rng(56)
        signal += rng.normal(0, 0.001, signal.shape)

        result = characterize_nv_cw(freq, signal)
        assert result["cw_sensitivity"] > 0
        assert result["linewidth"] > 0
        assert result["contrast"] > 0

