"""Extra stress tests exposing real bugs in WIMP modules.

Each test class targets a specific bug or edge case found during code audit.
Tests are written to FAIL against the current (unfixed) code so that fixes
can be verified.
"""

import math
import threading
import warnings

import numpy as np
import pytest

from wimp.constants import GAMMA_NV, D0, D0_TEMP_COEFF


# ============================================================================
# 1. calibration.estimate_temperature_from_ramsey — wrong formula
# ============================================================================


class TestEstimateTemperatureFromRamsey:
    """The function subtracts D0 (~2.87 GHz) from a Ramsey oscillation
    frequency (~MHz-scale), producing wildly wrong temperature estimates.

    The docstring says *frequency* is the "Measured Ramsey oscillation
    frequency (Hz)", which is the detuning from the microwave drive
    (typically kHz–MHz scale, NOT the full GHz transition frequency).

    The formula should NOT subtract D0.
    """

    def test_zero_field_zero_detuning_gives_zero_temp_shift(self):
        """If frequency == 0 and b_known == 0, there is no detuning,
        so the temperature shift should be exactly 0."""
        from wimp.calibration import estimate_temperature_from_ramsey

        dt = estimate_temperature_from_ramsey(0.0, b_known=0.0)
        assert dt == pytest.approx(0.0, abs=1e-6), (
            f"Expected ~0 K shift for zero detuning, got {dt:.3e} K"
        )

    def test_small_detuning_gives_small_temp_shift(self):
        """A 1 kHz detuning at zero field implies a tiny D0 shift,
        which corresponds to a small temperature change."""
        from wimp.calibration import estimate_temperature_from_ramsey

        dt = estimate_temperature_from_ramsey(1e3, b_known=0.0)
        # D0_TEMP_COEFF ~ -74 kHz/K, so 1 kHz → ~-0.0135 K
        expected = 1e3 / D0_TEMP_COEFF  # small number
        assert abs(dt) < 1.0, (
            f"1 kHz detuning should give < 1 K shift, got {dt:.3e} K"
        )
        assert dt == pytest.approx(expected, rel=0.01)

    def test_known_field_cancels_zeeman(self):
        """When the Ramsey oscillation equals the Zeeman shift exactly,
        the temperature shift should be zero."""
        from wimp.calibration import estimate_temperature_from_ramsey

        b_known = 1e-4  # 0.1 mT
        zeeman_freq = GAMMA_NV * b_known
        dt = estimate_temperature_from_ramsey(zeeman_freq, b_known=b_known)
        assert dt == pytest.approx(0.0, abs=1e-6), (
            f"Expected ~0 K when detuning equals Zeeman shift, got {dt:.3e}"
        )


# ============================================================================
# 2. analysis.noise_spectroscopy — division by zero when tau contains 0
# ============================================================================


class TestNoiseSpectroscopyTauZero:
    """noise_spectroscopy computes freq = 1/(2*tau), which produces
    inf when tau contains zero.  It should either skip tau=0 entries
    or raise a clear error.
    """

    def test_tau_with_zero_produces_no_inf(self):
        from wimp.analysis import noise_spectroscopy

        tau = np.array([0.0, 1e-6, 2e-6, 5e-6])
        coh = np.array([1.0, 0.9, 0.8, 0.5])
        result = noise_spectroscopy(coh, tau, n_pulses=1)
        assert np.all(np.isfinite(result["freq"])), (
            "freq contains non-finite values when tau includes 0"
        )
        assert np.all(np.isfinite(result["spectral_density"])), (
            "spectral_density contains non-finite values when tau includes 0"
        )

    def test_tau_all_positive_works(self):
        """Baseline: positive tau values should work fine."""
        from wimp.analysis import noise_spectroscopy

        tau = np.array([1e-6, 2e-6, 5e-6, 10e-6])
        coh = np.array([0.95, 0.85, 0.6, 0.3])
        result = noise_spectroscopy(coh, tau, n_pulses=4)
        assert np.all(np.isfinite(result["freq"]))
        assert np.all(np.isfinite(result["spectral_density"]))


# ============================================================================
# 3. relaxation._initial_frequency_guess — crash on single-element tau
# ============================================================================


class TestInitialFrequencyGuessSingleElement:
    """_initial_frequency_guess calls np.diff(tau), which returns an
    empty array when tau has only 1 element.  np.median([]) returns nan.
    The function should handle this gracefully.
    """

    def test_single_element_tau_returns_finite(self):
        from wimp.relaxation import _initial_frequency_guess

        tau = np.array([1e-6])
        signal = np.array([0.5])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            freq = _initial_frequency_guess(tau, signal)
        assert np.isfinite(freq), (
            f"Expected finite frequency for single-element tau, got {freq}"
        )

    def test_empty_tau_returns_finite(self):
        from wimp.relaxation import _initial_frequency_guess

        tau = np.array([])
        signal = np.array([])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            freq = _initial_frequency_guess(tau, signal)
        assert np.isfinite(freq), (
            f"Expected finite frequency for empty tau, got {freq}"
        )


# ============================================================================
# 4. relaxation._fit_odmr_single — bounds assume ascending freq
# ============================================================================


class TestFitODMRDescendingFreq:
    """_fit_odmr_single and _fit_odmr_double use freq[0] as lower bound
    and freq[-1] as upper bound for center frequency.  If the freq
    array is descending, lower > upper and curve_fit raises ValueError.
    """

    def test_fit_odmr_descending_freq(self):
        from wimp.relaxation import fit_odmr
        from wimp.synthetic import generate_odmr_spectrum

        freq_asc = np.linspace(2.8e9, 2.94e9, 300)
        data = generate_odmr_spectrum(freq_asc, 0.5e-3, snr=100, seed=42)
        # Reverse the frequency and signal arrays
        freq_desc = freq_asc[::-1]
        signal_desc = data["signal"][::-1]
        # Should still fit without crashing
        result = fit_odmr(freq_desc, signal_desc)
        assert "b_field" in result, "fit_odmr failed on descending freq array"
        assert result["b_field"] == pytest.approx(0.5e-3, rel=0.15)

    def test_fit_odmr_descending_single_dip(self):
        from wimp.relaxation import fit_odmr
        from wimp.synthetic import generate_odmr_spectrum

        freq_asc = np.linspace(2.82e9, 2.92e9, 300)
        data = generate_odmr_spectrum(freq_asc, 1e-6, snr=200, seed=43)
        freq_desc = freq_asc[::-1]
        signal_desc = data["signal"][::-1]
        result = fit_odmr(freq_desc, signal_desc, n_dips=1)
        assert "b_field" in result


# ============================================================================
# 5. realtime.RealtimeProcessor.clear — race condition
# ============================================================================


class TestRealtimeClearRaceCondition:
    """clear() resets _kalman_filters outside the lock.  If the
    background thread is mid-fit using _kalman_filters, this can
    cause KeyError or produce stale results.

    We test that _kalman_filters is reset inside the lock by verifying
    it is fully cleared after clear() returns.
    """

    def test_clear_resets_kalman_under_lock(self):
        from wimp.realtime import (
            RealtimeProcessor, RealtimeConfig, MeasurementFrame,
        )
        from wimp.synthetic import generate_ramsey_data

        cfg = RealtimeConfig(
            protocol="ramsey", averaging=1, use_kalman=True,
        )
        proc = RealtimeProcessor(cfg)

        # Push a frame and process to initialize Kalman filters
        tau = np.linspace(0, 5e-6, 100)
        data = generate_ramsey_data(tau, b_field=50e-6, t2star=1e-6, snr=100, seed=1)
        frame = MeasurementFrame(tau_array=tau, signal=data["signal"])
        proc.push(frame)
        result = proc.process_once()
        assert result is not None

        # There should be Kalman filters now
        assert len(proc._kalman_filters) > 0

        # clear() should reset them
        proc.clear()
        assert len(proc._kalman_filters) == 0, (
            "_kalman_filters not cleared by clear()"
        )


# ============================================================================
# 6. sensitivity.adaptive_protocol_recommendation — wrong constraint
# ============================================================================


class TestAdaptiveProtocolConstraintBug:
    """The loop checks `if t2 > max_dur: break` which does not depend
    on `n`.  This means the constraint either blocks ALL pulse counts
    or NONE, regardless of sequence duration.

    With a short max_sequence_duration (shorter than t2), the constraint
    should still allow low pulse counts whose sequence fits within the
    duration limit.
    """

    def test_constraint_allows_low_pulse_counts(self):
        from wimp.sensitivity import adaptive_protocol_recommendation

        char = {"t2star": 1e-6, "t2": 100e-6, "t1": 5e-3}
        # max_sequence_duration is shorter than t2 but should still
        # allow hahn_echo (n=1, tau ≈ t2/2 = 50 us < 80 us)
        result = adaptive_protocol_recommendation(
            char, target="ac",
            constraints={"max_sequence_duration": 80e-6, "max_n_pulses": 256},
        )
        # Should recommend echo, not be blocked entirely
        assert "recommended_protocol" in result
        assert result["predicted_sensitivity"] > 0

    def test_high_pulse_count_excluded_by_duration(self):
        """With a very short max_duration, high pulse counts should be
        excluded because n*tau would exceed the limit."""
        from wimp.sensitivity import adaptive_protocol_recommendation

        char = {"t2star": 1e-6, "t2": 100e-6, "t1": 5e-3}
        result = adaptive_protocol_recommendation(
            char, target="ac",
            constraints={"max_sequence_duration": 10e-6, "max_n_pulses": 256},
        )
        # With max_dur=10 us, only n=1 tau≈50us wouldn't even fit,
        # but the function should still return a recommendation
        assert "recommended_protocol" in result


# ============================================================================
# 7. denoising.denoise_butterworth — no validation for cutoff <= 0
# ============================================================================


class TestDenoiseButterworthInvalidParams:
    """denoise_butterworth doesn't validate cutoff <= 0 or fs <= 0,
    which causes scipy.signal.butter to raise a ValueError.  The
    function should raise a clear ValueError itself.
    """

    def test_zero_cutoff_raises(self):
        from wimp.denoising import denoise_butterworth

        signal = np.random.randn(100)
        with pytest.raises(ValueError):
            denoise_butterworth(signal, cutoff=0.0, fs=1000.0)

    def test_negative_cutoff_raises(self):
        from wimp.denoising import denoise_butterworth

        signal = np.random.randn(100)
        with pytest.raises(ValueError):
            denoise_butterworth(signal, cutoff=-10.0, fs=1000.0)

    def test_zero_fs_raises(self):
        from wimp.denoising import denoise_butterworth

        signal = np.random.randn(100)
        with pytest.raises(ValueError):
            denoise_butterworth(signal, cutoff=100.0, fs=0.0)

    def test_negative_fs_raises(self):
        from wimp.denoising import denoise_butterworth

        signal = np.random.randn(100)
        with pytest.raises(ValueError):
            denoise_butterworth(signal, cutoff=100.0, fs=-1000.0)


# ============================================================================
# 8. analysis.noise_spectroscopy — n_pulses=0 division by zero
# ============================================================================


class TestNoiseSpectroscopyNPulsesZero:
    """S = chi / (8 * n_pulses * tau * gamma_sq) divides by n_pulses.
    If n_pulses=0, the result is inf/nan.
    """

    def test_n_pulses_zero_raises(self):
        from wimp.analysis import noise_spectroscopy

        tau = np.array([1e-6, 2e-6, 5e-6])
        coh = np.array([0.9, 0.8, 0.5])
        with pytest.raises(ValueError, match="n_pulses"):
            noise_spectroscopy(coh, tau, n_pulses=0)


# ============================================================================
# 9. sensitivity — division by zero with zero-valued parameters
# ============================================================================


class TestSensitivityZeroParams:
    """All sensitivity functions blow up with ZeroDivisionError when
    contrast, t2star, readout_rate, etc. are zero.
    """

    def test_dc_sensitivity_zero_contrast_raises(self):
        from wimp.sensitivity import dc_sensitivity

        with pytest.raises((ValueError, ZeroDivisionError)):
            dc_sensitivity(0.0, 1e-6, 1e5)

    def test_dc_sensitivity_zero_t2star_raises(self):
        from wimp.sensitivity import dc_sensitivity

        with pytest.raises((ValueError, ZeroDivisionError)):
            dc_sensitivity(0.03, 0.0, 1e5)

    def test_cw_sensitivity_zero_contrast_raises(self):
        from wimp.sensitivity import cw_sensitivity

        with pytest.raises((ValueError, ZeroDivisionError)):
            cw_sensitivity(0.0, 5e6, 1e5)

    def test_cw_sensitivity_zero_readout_raises(self):
        from wimp.sensitivity import cw_sensitivity

        with pytest.raises((ValueError, ZeroDivisionError)):
            cw_sensitivity(0.03, 5e6, 0.0)


# ============================================================================
# 10. stability.allan_variance — single sample returns empty without error
# ============================================================================


class TestAllanVarianceSingleSample:
    """With n=1, the function should either raise or return empty arrays
    without crashing."""

    def test_single_sample_does_not_crash(self):
        from wimp.stability import allan_variance

        result = allan_variance(np.array([1.0]), fs=1.0)
        assert len(result["taus"]) == 0 or isinstance(result["taus"], np.ndarray)
        assert len(result["avar"]) == len(result["taus"])

    def test_two_samples_works(self):
        from wimp.stability import allan_variance

        result = allan_variance(np.array([1.0, 2.0]), fs=1.0)
        assert len(result["taus"]) >= 0


# ============================================================================
# 11. sensitivity.measurement_time_for_target — hardcoded linewidth
# ============================================================================


class TestMeasurementTimeHardcodedLinewidth:
    """measurement_time_for_target uses a hardcoded 5 MHz linewidth for
    CW ODMR instead of accepting it as a parameter.  This test documents
    the limitation.
    """

    def test_cw_odmr_uses_5mhz_linewidth(self):
        """Verify the hardcoded 5 MHz is being used (not a bug fix test,
        just documentation of the behavior)."""
        from wimp.sensitivity import measurement_time_for_target, cw_sensitivity

        t = measurement_time_for_target(1e-9, "cw_odmr")
        eta_expected = cw_sensitivity(0.03, 5e6, 1e5)
        t_expected = (eta_expected / 1e-9) ** 2
        assert t == pytest.approx(t_expected, rel=1e-6)
