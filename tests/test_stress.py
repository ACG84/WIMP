"""Comprehensive stress / sanity tests for WIMP.

Exercises edge cases, boundary conditions, cross-module integration, and
unusual inputs across every major subsystem, with special attention to
the CW ODMR pathway introduced in Batch 4.
"""

import math
import numpy as np
import pytest

from wimp.constants import GAMMA_NV, D0, A_N, D0_TEMP_COEFF, MU0, HBAR, MU_B, G_NV
from wimp.relaxation import (
    lorentzian_dip,
    odmr_model,
    odmr_single_dip_model,
    fit_odmr,
    extract_field_odmr,
    fit_ramsey,
    fit_t2_decay,
    fit_t1_recovery,
)
from wimp.synthetic import (
    generate_odmr_spectrum,
    generate_ramsey_data,
    generate_echo_data,
    generate_t1_data,
    generate_full_experiment,
)
from wimp.sensitivity import (
    cw_sensitivity,
    dc_sensitivity,
    ac_sensitivity,
    t1_sensitivity,
    compare_protocols,
    optimal_protocol,
    measurement_time_for_target,
    sensitivity_vs_tau,
)
from wimp.calibration import (
    characterize_nv_cw,
    characterize_nv,
    temperature_correction,
)
from wimp.pipeline import (
    PipelineConfig,
    default_config,
    validate_config,
    _fit_single_nd,
    run_pipeline,
)
from wimp.realtime import (
    RealtimeProcessor,
    RealtimeConfig,
    MeasurementFrame,
)
from wimp.io import WIMPDataset
from wimp.viz import plot_odmr_spectrum
from matplotlib.figure import Figure


# ============================================================================
# 1. ODMR MODEL FUNCTIONS — edge cases
# ============================================================================


class TestLorentzianDipEdgeCases:
    def test_single_point_at_center(self):
        """Exactly at center should return -amplitude."""
        val = lorentzian_dip(np.array([1e9]), 1e9, 0.05, 5e6)
        assert val[0] == pytest.approx(-0.05, rel=1e-6)

    def test_symmetric_about_center(self):
        """Dip should be symmetric about center."""
        offsets = np.array([1e6, 5e6, 10e6, 50e6])
        center = 2.87e9
        left = lorentzian_dip(center - offsets, center, 0.03, 5e6)
        right = lorentzian_dip(center + offsets, center, 0.03, 5e6)
        np.testing.assert_allclose(left, right, atol=1e-15)

    def test_zero_amplitude_returns_zero(self):
        val = lorentzian_dip(np.linspace(1e9, 3e9, 50), 2e9, 0.0, 5e6)
        np.testing.assert_allclose(val, 0.0, atol=1e-15)

    def test_very_narrow_linewidth(self):
        """Very narrow linewidth should give near-zero away from center."""
        freq = np.array([1e9, 1e9 + 1e3])
        val = lorentzian_dip(freq, 1e9, 0.05, 1.0)  # 1 Hz linewidth
        # At center
        assert val[0] == pytest.approx(-0.05, rel=1e-6)
        # 1 kHz away should be very small
        assert abs(val[1]) < 1e-7

    def test_very_wide_linewidth(self):
        """Very wide linewidth should remain non-zero far from center."""
        freq = np.array([1e9 + 1e9])  # 1 GHz away
        val = lorentzian_dip(freq, 1e9, 0.05, 1e9)
        assert abs(val[0]) > 1e-3

    def test_large_array(self):
        """Test with many points."""
        freq = np.linspace(2.8e9, 2.96e9, 100_000)
        val = lorentzian_dip(freq, D0, 0.03, 5e6)
        assert val.shape == freq.shape
        assert np.all(val <= 0)  # all dips are negative


class TestODMRModelEdgeCases:
    def test_symmetric_dips(self):
        """Symmetric params should give symmetric spectrum about midpoint."""
        f0 = D0
        split = 30e6
        freq = np.linspace(f0 - 2 * split, f0 + 2 * split, 1001)
        mid = len(freq) // 2
        signal = odmr_model(freq, 1.0, f0 - split, f0 + split,
                            0.03, 0.03, 5e6, 5e6)
        # Should be symmetric around midpoint
        np.testing.assert_allclose(signal[:mid], signal[-mid:][::-1], atol=1e-12)

    def test_zero_field_dips_merge(self):
        """When f_minus == f_plus, we get a single deeper dip."""
        freq = np.linspace(2.8e9, 2.94e9, 500)
        signal = odmr_model(freq, 1.0, D0, D0, 0.03, 0.03, 5e6, 5e6)
        # Should have a single minimum at D0
        min_idx = np.argmin(signal)
        assert abs(freq[min_idx] - D0) < (freq[1] - freq[0])

    def test_baseline_scales_output(self):
        """Baseline should shift the entire spectrum up."""
        freq = np.linspace(2.8e9, 2.94e9, 200)
        s1 = odmr_model(freq, 1.0, D0 - 14e6, D0 + 14e6, 0.03, 0.03, 5e6, 5e6)
        s2 = odmr_model(freq, 2.0, D0 - 14e6, D0 + 14e6, 0.03, 0.03, 5e6, 5e6)
        np.testing.assert_allclose(s2 - s1, 1.0, atol=1e-10)


# ============================================================================
# 2. FIT_ODMR — various field strengths and noise levels
# ============================================================================


class TestFitODMRFieldRange:
    """Fit across different B-field magnitudes."""

    @pytest.mark.parametrize("b_field,freq_range", [
        (0.3e-3, (2.8e9, 2.94e9)),
        (0.5e-3, (2.8e9, 2.94e9)),
        (1e-3, (2.8e9, 2.96e9)),
        (5e-3, (2.5e9, 3.2e9)),
        (10e-3, (2.3e9, 3.5e9)),
    ])
    def test_double_dip_various_fields(self, b_field, freq_range):
        freq = np.linspace(freq_range[0], freq_range[1], 600)
        data = generate_odmr_spectrum(freq, b_field, snr=200, seed=42)
        result = fit_odmr(data["freq"], data["signal"])
        assert result["n_dips"] == 2
        assert result["b_field"] == pytest.approx(b_field, rel=0.10)

    def test_very_small_field_single_dip(self):
        """Near-zero field should produce a single merged dip."""
        freq = np.linspace(2.82e9, 2.92e9, 300)
        data = generate_odmr_spectrum(freq, 1e-6, snr=200, seed=1)
        # With very small B, we expect single dip
        result = fit_odmr(data["freq"], data["signal"])
        assert result["n_dips"] == 1
        assert result["b_field"] < 1e-3  # should be tiny

    def test_borderline_field_does_not_crash(self):
        """B=0.1 mT gives ~5.6 MHz splitting, near auto-detect threshold.
        May detect 1 or 2 dips — just verify it doesn't crash."""
        freq = np.linspace(2.8e9, 2.94e9, 500)
        data = generate_odmr_spectrum(freq, 0.1e-3, snr=200, seed=42)
        result = fit_odmr(data["freq"], data["signal"])
        assert result["n_dips"] in (1, 2)
        assert "b_field" in result

    def test_forced_single_dip(self):
        """Force n_dips=1 even on a double-dip spectrum."""
        freq = np.linspace(2.8e9, 2.94e9, 500)
        data = generate_odmr_spectrum(freq, 0.5e-3, snr=100, seed=5)
        result = fit_odmr(data["freq"], data["signal"], n_dips=1)
        assert result["n_dips"] == 1

    def test_forced_double_dip(self):
        """Force n_dips=2 even on a single-dip spectrum."""
        freq = np.linspace(2.82e9, 2.92e9, 400)
        # Very small field → effectively single dip
        data = generate_odmr_spectrum(freq, 1e-6, snr=100, seed=3)
        # Should still return a result (may not be great, but shouldn't crash)
        result = fit_odmr(data["freq"], data["signal"], n_dips=2)
        assert result["n_dips"] == 2
        assert "b_field" in result


class TestFitODMRNoise:
    """Test robustness under varying noise levels."""

    @pytest.mark.parametrize("snr", [10, 25, 50, 100, 500])
    def test_snr_sweep(self, snr):
        freq = np.linspace(2.8e9, 2.94e9, 500)
        data = generate_odmr_spectrum(freq, 0.5e-3, snr=snr, seed=77)
        result = fit_odmr(data["freq"], data["signal"])
        assert "b_field" in result
        assert result["b_field"] > 0
        # At low SNR, we just check it doesn't crash; at high SNR, accuracy
        if snr >= 50:
            assert result["b_field"] == pytest.approx(0.5e-3, rel=0.15)

    def test_zero_noise(self):
        """Perfect signal with zero noise should fit exactly."""
        freq = np.linspace(2.8e9, 2.94e9, 500)
        data = generate_odmr_spectrum(freq, 0.5e-3, snr=0, seed=None)
        # snr=0 means no noise
        result = fit_odmr(data["freq"], data["signal"])
        assert result["b_field"] == pytest.approx(0.5e-3, rel=0.01)

    def test_asymmetric_dips(self):
        """Asymmetric dip amplitudes should still fit correctly."""
        freq = np.linspace(2.8e9, 2.94e9, 500)
        data = generate_odmr_spectrum(freq, 0.5e-3, asymmetry=0.4, snr=100, seed=88)
        result = fit_odmr(data["freq"], data["signal"])
        assert result["n_dips"] == 2
        assert result["b_field"] == pytest.approx(0.5e-3, rel=0.10)


class TestFitODMRArrayEdges:
    """Test various frequency array configurations."""

    def test_sparse_points(self):
        """Only 30 frequency points."""
        freq = np.linspace(2.8e9, 2.94e9, 30)
        data = generate_odmr_spectrum(freq, 0.5e-3, snr=100, seed=10)
        result = fit_odmr(data["freq"], data["signal"])
        assert "b_field" in result

    def test_dense_points(self):
        """5000 frequency points."""
        freq = np.linspace(2.8e9, 2.94e9, 5000)
        data = generate_odmr_spectrum(freq, 0.5e-3, snr=100, seed=11)
        result = fit_odmr(data["freq"], data["signal"])
        assert result["b_field"] == pytest.approx(0.5e-3, rel=0.05)

    def test_narrow_window(self):
        """Frequency window barely captures both dips."""
        b = 0.3e-3
        split = GAMMA_NV * b
        freq = np.linspace(D0 - split - 20e6, D0 + split + 20e6, 400)
        data = generate_odmr_spectrum(freq, b, snr=100, seed=12)
        result = fit_odmr(data["freq"], data["signal"])
        assert result["b_field"] == pytest.approx(b, rel=0.15)


class TestExtractFieldODMR:
    def test_double_dip(self):
        fit = {"n_dips": 2, "f_minus": D0 - 14e6, "f_plus": D0 + 14e6,
               "splitting": 28e6}
        b = extract_field_odmr(fit)
        expected = 28e6 / (2 * GAMMA_NV)
        assert b == pytest.approx(expected, rel=1e-10)

    def test_single_dip(self):
        fit = {"n_dips": 1, "center": D0 + 14e6}
        b = extract_field_odmr(fit)
        expected = 14e6 / GAMMA_NV
        assert b == pytest.approx(expected, rel=1e-10)

    def test_single_dip_below_d0(self):
        fit = {"n_dips": 1, "center": D0 - 14e6}
        b = extract_field_odmr(fit)
        expected = 14e6 / GAMMA_NV
        assert b == pytest.approx(expected, rel=1e-10)


# ============================================================================
# 3. SYNTHETIC ODMR — edge cases
# ============================================================================


class TestGenerateODMRSpectrumEdgeCases:
    def test_output_keys(self):
        freq = np.linspace(2.8e9, 2.94e9, 100)
        data = generate_odmr_spectrum(freq, 0.5e-3, seed=1)
        for key in ("freq", "signal", "clean_signal", "ground_truth"):
            assert key in data

    def test_zero_field_produces_single_dip(self):
        freq = np.linspace(2.82e9, 2.92e9, 200)
        data = generate_odmr_spectrum(freq, 0.0, seed=2)
        assert data["ground_truth"]["n_dips"] == 1

    def test_large_field(self):
        """Large B should produce widely separated dips."""
        freq = np.linspace(2.0e9, 4.0e9, 1000)
        b = 10e-3  # 10 mT
        data = generate_odmr_spectrum(freq, b, seed=3)
        gt = data["ground_truth"]
        assert gt["n_dips"] == 2
        expected_split = 2 * GAMMA_NV * b
        assert gt["splitting"] == pytest.approx(expected_split, rel=1e-6)

    def test_deterministic_seed(self):
        freq = np.linspace(2.8e9, 2.94e9, 100)
        d1 = generate_odmr_spectrum(freq, 0.5e-3, seed=99)
        d2 = generate_odmr_spectrum(freq, 0.5e-3, seed=99)
        np.testing.assert_array_equal(d1["signal"], d2["signal"])

    def test_different_seeds_differ(self):
        freq = np.linspace(2.8e9, 2.94e9, 100)
        d1 = generate_odmr_spectrum(freq, 0.5e-3, snr=50, seed=99)
        d2 = generate_odmr_spectrum(freq, 0.5e-3, snr=50, seed=100)
        assert not np.array_equal(d1["signal"], d2["signal"])

    def test_negative_field(self):
        """Negative B should behave same as positive (abs used internally)."""
        freq = np.linspace(2.8e9, 2.94e9, 200)
        d_pos = generate_odmr_spectrum(freq, 0.5e-3, snr=0, seed=None)
        d_neg = generate_odmr_spectrum(freq, -0.5e-3, snr=0, seed=None)
        np.testing.assert_allclose(d_pos["clean_signal"], d_neg["clean_signal"], atol=1e-15)

    def test_custom_contrast(self):
        freq = np.linspace(2.8e9, 2.94e9, 200)
        data = generate_odmr_spectrum(freq, 0.5e-3, contrast=0.10, snr=0, seed=None)
        # Dips should be ~10% deep
        clean = data["clean_signal"]
        baseline = data["ground_truth"]["baseline"]
        min_val = np.min(clean)
        depth = (baseline - min_val) / baseline
        assert depth == pytest.approx(0.10, abs=0.01)

    def test_custom_baseline(self):
        freq = np.linspace(2.8e9, 2.94e9, 200)
        data = generate_odmr_spectrum(freq, 0.5e-3, baseline=5.0, snr=0, seed=None)
        # Maximum signal should be near 5.0
        assert np.max(data["clean_signal"]) == pytest.approx(5.0, abs=0.01)


class TestGenerateFullExperimentCWODMR:
    def test_cw_odmr_protocol(self):
        exp = generate_full_experiment(n_nds=3, n_neurons=2, n_timepoints=50,
                                       protocol="cw_odmr", seed=1)
        assert exp["protocol"] == "cw_odmr"
        assert exp["signal"].shape[0] == 3
        assert len(exp["fit_data"]) == 3

    def test_unknown_protocol_raises(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            generate_full_experiment(protocol="nonexistent")


# ============================================================================
# 4. SENSITIVITY — edge cases and CW ODMR
# ============================================================================


class TestCWSensitivity:
    def test_positive_output(self):
        eta = cw_sensitivity(0.03, 5e6, 1e5)
        assert eta > 0

    def test_improves_with_contrast(self):
        """Better contrast → lower (better) sensitivity."""
        eta_low = cw_sensitivity(0.01, 5e6, 1e5)
        eta_high = cw_sensitivity(0.10, 5e6, 1e5)
        assert eta_high < eta_low

    def test_improves_with_readout_rate(self):
        eta_low_r = cw_sensitivity(0.03, 5e6, 1e4)
        eta_high_r = cw_sensitivity(0.03, 5e6, 1e6)
        assert eta_high_r < eta_low_r

    def test_degrades_with_linewidth(self):
        """Wider linewidth → worse sensitivity."""
        eta_narrow = cw_sensitivity(0.03, 1e6, 1e5)
        eta_wide = cw_sensitivity(0.03, 20e6, 1e5)
        assert eta_wide > eta_narrow

    def test_scales_with_n_nv(self):
        eta_1 = cw_sensitivity(0.03, 5e6, 1e5, n_nv=1)
        eta_100 = cw_sensitivity(0.03, 5e6, 1e5, n_nv=100)
        assert eta_100 == pytest.approx(eta_1 / 10, rel=1e-10)

    def test_formula_value(self):
        """Check the formula gives the expected result."""
        c = 0.03
        lw = 5e6
        R = 1e5
        expected = (4 / (3 * math.sqrt(3))) * lw / (c * GAMMA_NV * math.sqrt(R))
        assert cw_sensitivity(c, lw, R) == pytest.approx(expected, rel=1e-12)


class TestCompareProtocolsCWRow:
    def test_cw_odmr_row_present(self):
        df = compare_protocols()
        assert "CW ODMR" in df["protocol"].values

    def test_custom_linewidth(self):
        df = compare_protocols(linewidth=10e6)
        row = df[df["protocol"] == "CW ODMR"].iloc[0]
        assert "10.0 MHz" in row["notes"]

    def test_cw_sensitivity_positive(self):
        df = compare_protocols()
        cw_row = df[df["protocol"] == "CW ODMR"].iloc[0]
        assert cw_row["sensitivity_T_sqrtHz"] > 0
        assert not math.isnan(cw_row["sensitivity_T_sqrtHz"])


class TestOptimalProtocolCWODMR:
    def test_cw_odmr_target(self):
        rec = optimal_protocol("cw_odmr")
        assert rec["protocol"] == "cw_odmr"
        assert "sensitivity_T_sqrtHz" in rec
        assert rec["sensitivity_T_sqrtHz"] > 0

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            optimal_protocol("quantum_teleportation")


class TestSensitivityVsTauCWFails:
    def test_cw_odmr_raises(self):
        """CW ODMR doesn't have tau dependence — should raise."""
        tau = np.linspace(1e-7, 1e-5, 50)
        with pytest.raises(ValueError, match="CW ODMR"):
            sensitivity_vs_tau("cw_odmr", tau)


class TestMeasurementTimeForTargetCW:
    def test_cw_odmr(self):
        t = measurement_time_for_target(1e-9, "cw_odmr")
        assert t > 0

    def test_unknown_protocol_raises(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            measurement_time_for_target(1e-9, "phaser_beam")


class TestSensitivityConsistency:
    """Cross-check sensitivity functions give consistent ordering."""

    def test_dc_vs_cw(self):
        """DC (Ramsey) and CW should give different but finite values."""
        eta_dc = dc_sensitivity(0.03, 1e-6, 1e5)
        eta_cw = cw_sensitivity(0.03, 5e6, 1e5)
        assert eta_dc > 0
        assert eta_cw > 0
        # Both should be in reasonable range (nT to mT / sqrt(Hz))
        assert 1e-15 < eta_dc < 1
        assert 1e-15 < eta_cw < 1


# ============================================================================
# 5. PIPELINE — CW ODMR dispatch
# ============================================================================


class TestPipelineConfigCWODMR:
    def test_default_config(self):
        cfg = default_config("cw_odmr")
        assert cfg.protocol == "cw_odmr"
        assert cfg.tau_range[0] < cfg.tau_range[1]

    def test_validate_config_accepts_cw(self):
        cfg = PipelineConfig(protocol="cw_odmr", tau_range=(2.8e9, 2.94e9))
        validate_config(cfg)  # should not raise

    def test_validate_config_rejects_unknown(self):
        cfg = PipelineConfig(protocol="antigravity", tau_range=(0, 1))
        with pytest.raises(ValueError, match="protocol must be one of"):
            validate_config(cfg)


class TestFitSingleNdCW:
    def test_dispatches_to_fit_odmr(self):
        freq = np.linspace(2.8e9, 2.94e9, 300)
        data = generate_odmr_spectrum(freq, 0.5e-3, snr=100, seed=50)
        result = _fit_single_nd("cw_odmr", data["freq"], data["signal"], {})
        assert "b_field" in result
        assert result["b_field"] > 0


class TestRunPipelineCWODMR:
    def test_end_to_end(self):
        freq = np.linspace(2.8e9, 2.94e9, 300)
        n_nds = 3
        signals = []
        for i in range(n_nds):
            data = generate_odmr_spectrum(freq, 0.5e-3, snr=100, seed=100 + i)
            signals.append(data["signal"])
        signal_arr = np.stack(signals)

        ds = WIMPDataset(
            tau_array=freq,
            signal=signal_arr,
            protocol="cw_odmr",
            metadata={"test": True},
        )
        cfg = PipelineConfig(
            protocol="cw_odmr",
            tau_range=(2.8e9, 2.94e9),
            source_localization=False,
        )
        results = run_pipeline(cfg, dataset=ds)
        assert len(results["fits"]) == n_nds
        assert results["field_timeseries"] is not None
        assert results["field_timeseries"].shape == (n_nds, 1)
        # All fields should be near 0.5 mT
        fields = results["field_timeseries"].ravel()
        for f in fields:
            assert f == pytest.approx(0.5e-3, rel=0.15)


# ============================================================================
# 6. REALTIME — CW ODMR dispatch
# ============================================================================


class TestRealtimeProcessorCW:
    def test_cw_odmr_process_once(self):
        cfg = RealtimeConfig(protocol="cw_odmr", averaging=1)
        proc = RealtimeProcessor(cfg)

        freq = np.linspace(2.8e9, 2.94e9, 300)
        data = generate_odmr_spectrum(freq, 0.5e-3, snr=100, seed=200)
        frame = MeasurementFrame(tau_array=freq, signal=data["signal"])
        proc.push(frame)

        result = proc.process_once()
        assert result is not None
        assert len(result.fits) == 1
        assert result.fits[0]["b_field"] > 0

    def test_cw_odmr_multi_nd(self):
        cfg = RealtimeConfig(protocol="cw_odmr", averaging=1)
        proc = RealtimeProcessor(cfg)

        freq = np.linspace(2.8e9, 2.94e9, 300)
        signals = []
        for i in range(4):
            data = generate_odmr_spectrum(freq, 0.5e-3, snr=100, seed=300 + i)
            signals.append(data["signal"])
        signal_2d = np.stack(signals)

        frame = MeasurementFrame(tau_array=freq, signal=signal_2d)
        proc.push(frame)
        result = proc.process_once()
        assert result is not None
        assert len(result.fits) == 4

    def test_cw_odmr_averaging(self):
        cfg = RealtimeConfig(protocol="cw_odmr", averaging=3)
        proc = RealtimeProcessor(cfg)

        freq = np.linspace(2.8e9, 2.94e9, 300)
        for i in range(3):
            data = generate_odmr_spectrum(freq, 0.5e-3, snr=50, seed=400 + i)
            frame = MeasurementFrame(tau_array=freq, signal=data["signal"])
            proc.push(frame)

        result = proc.process_once()
        assert result is not None
        assert result.frame_count == 3


# ============================================================================
# 7. CALIBRATION — CW ODMR
# ============================================================================


class TestCharacterizeNVCWStress:
    @pytest.mark.parametrize("b_field", [0.3e-3, 0.5e-3, 2e-3])
    def test_various_fields(self, b_field):
        freq = np.linspace(2.7e9, 3.0e9, 500)
        data = generate_odmr_spectrum(freq, b_field, snr=200, seed=500)
        result = characterize_nv_cw(data["freq"], data["signal"])
        assert result["b_field"] == pytest.approx(b_field, rel=0.15)
        assert result["linewidth"] > 0
        assert result["cw_sensitivity"] > 0

    def test_temperature_shift_near_zero_for_centered_spectrum(self):
        """When center is at D0, temperature shift should be near zero."""
        freq = np.linspace(2.8e9, 2.94e9, 400)
        data = generate_odmr_spectrum(freq, 0.5e-3, snr=200, seed=501)
        result = characterize_nv_cw(data["freq"], data["signal"])
        # Center freq should be near D0, so temp shift should be small
        assert abs(result["temperature_shift"]) < 50  # < 50 K shift

    def test_single_dip_characterization(self):
        """Near-zero field → characterize_nv_cw should still work."""
        freq = np.linspace(2.82e9, 2.92e9, 300)
        data = generate_odmr_spectrum(freq, 1e-6, snr=200, seed=502)
        result = characterize_nv_cw(data["freq"], data["signal"])
        assert "linewidth" in result
        assert result["contrast"] > 0


# ============================================================================
# 8. VIZ — ODMR plots
# ============================================================================


class TestPlotODMRSpectrumStress:
    def test_minimal_data(self):
        """Plotting with very few points."""
        freq = np.linspace(2.8e9, 2.94e9, 10)
        signal = np.ones(10)
        fig = plot_odmr_spectrum(freq, signal)
        assert isinstance(fig, Figure)

    def test_with_double_dip_fit(self):
        freq = np.linspace(2.8e9, 2.94e9, 500)
        data = generate_odmr_spectrum(freq, 0.5e-3, snr=100, seed=600)
        fit = fit_odmr(data["freq"], data["signal"])
        fig = plot_odmr_spectrum(data["freq"], data["signal"], fit_result=fit)
        assert isinstance(fig, Figure)

    def test_with_single_dip_fit(self):
        freq = np.linspace(2.82e9, 2.92e9, 300)
        data = generate_odmr_spectrum(freq, 1e-6, snr=100, seed=601)
        fit = fit_odmr(data["freq"], data["signal"])
        fig = plot_odmr_spectrum(data["freq"], data["signal"], fit_result=fit)
        assert isinstance(fig, Figure)

    def test_mhz_units(self):
        freq = np.linspace(2.8e9, 2.94e9, 100)
        signal = np.ones(100)
        fig = plot_odmr_spectrum(freq, signal, freq_units="MHz")
        assert isinstance(fig, Figure)


# ============================================================================
# 9. CROSS-MODULE INTEGRATION — generate → fit → pipeline → sensitivity
# ============================================================================


class TestSyntheticToFitRoundtrip:
    """Generate synthetic data, fit it, and verify field recovery."""

    @pytest.mark.parametrize("protocol,gen_kw,fit_fn,b_key", [
        ("ramsey",
         dict(b_field=50e-6, t2star=1e-6, snr=100, seed=700),
         "fit_ramsey", "b_field"),
        ("echo",
         dict(t2=100e-6, snr=100, seed=701),
         "fit_t2_decay", "t2"),
        ("t1",
         dict(t1=5e-3, snr=100, seed=702),
         "fit_t1_recovery", "t1"),
    ])
    def test_pulsed_protocols(self, protocol, gen_kw, fit_fn, b_key):
        if protocol == "ramsey":
            tau = np.linspace(0, 5e-6, 200)
            data = generate_ramsey_data(tau, **gen_kw)
            result = fit_ramsey(tau, data["signal"])
            assert result[b_key] == pytest.approx(gen_kw["b_field"], rel=0.15)
        elif protocol == "echo":
            tau = np.linspace(0, 300e-6, 200)
            data = generate_echo_data(tau, **gen_kw)
            result = fit_t2_decay(tau, data["signal"])
            assert result[b_key] == pytest.approx(gen_kw["t2"], rel=0.15)
        elif protocol == "t1":
            tau = np.linspace(0, 25e-3, 200)
            data = generate_t1_data(tau, **gen_kw)
            result = fit_t1_recovery(tau, data["signal"])
            assert result[b_key] == pytest.approx(gen_kw["t1"], rel=0.15)

    def test_odmr_roundtrip(self):
        """Generate ODMR → fit → extract field → compare."""
        b_true = 0.5e-3
        freq = np.linspace(2.8e9, 2.94e9, 500)
        data = generate_odmr_spectrum(freq, b_true, snr=200, seed=710)
        result = fit_odmr(data["freq"], data["signal"])
        b_fit = extract_field_odmr(result)
        assert b_fit == pytest.approx(b_true, rel=0.05)


class TestFullExperimentToPipelineRoundtrip:
    """Generate full experiment → run pipeline → check fields."""

    @pytest.mark.parametrize("protocol", ["ramsey", "echo", "t1", "cw_odmr"])
    def test_protocol(self, protocol):
        exp = generate_full_experiment(
            n_nds=2, n_neurons=2, n_timepoints=50,
            protocol=protocol, snr=100, seed=800,
        )
        ds = WIMPDataset(
            tau_array=exp["tau_array"],
            signal=exp["signal"],
            protocol=protocol,
        )
        cfg = PipelineConfig(
            protocol=protocol,
            tau_range=default_config(protocol).tau_range,
            source_localization=False,
        )
        results = run_pipeline(cfg, dataset=ds)
        assert len(results["fits"]) == 2
        for f in results["fits"]:
            assert isinstance(f, dict)


class TestODMREndToEndSensitivity:
    """Full chain: generate → fit → characterize → sensitivity."""

    def test_full_chain(self):
        b = 0.5e-3
        freq = np.linspace(2.8e9, 2.94e9, 500)
        data = generate_odmr_spectrum(freq, b, snr=200, seed=900)

        # Fit
        fit = fit_odmr(data["freq"], data["signal"])
        assert fit["b_field"] == pytest.approx(b, rel=0.10)

        # Characterize
        char = characterize_nv_cw(data["freq"], data["signal"])
        assert char["b_field"] == pytest.approx(b, rel=0.10)
        assert char["cw_sensitivity"] > 0
        assert char["linewidth"] > 0

        # Sensitivity from extracted params
        eta = cw_sensitivity(char["contrast"], char["linewidth"], 1e5)
        assert eta > 0
        assert eta == pytest.approx(char["cw_sensitivity"], rel=0.01)

        # Compare with other protocols
        df = compare_protocols(contrast=char["contrast"], linewidth=char["linewidth"])
        assert len(df) == 5
        cw_row = df[df["protocol"] == "CW ODMR"].iloc[0]
        assert cw_row["sensitivity_T_sqrtHz"] > 0


# ============================================================================
# 10. CONSTANTS — sanity checks for new constant
# ============================================================================


class TestANConstant:
    def test_value(self):
        assert A_N == pytest.approx(2.16e6, rel=1e-3)

    def test_hyperfine_splitting_relationship(self):
        """A_N << D0 (hyperfine is much smaller than zero-field splitting)."""
        assert A_N < D0 / 100

    def test_properly_exported(self):
        """Check A_N is importable from wimp.constants."""
        from wimp.constants import A_N as a
        assert a == A_N


# ============================================================================
# 11. EDGE CASES — misc malformed inputs
# ============================================================================


class TestMalformedInputs:
    def test_fit_odmr_with_flat_signal(self):
        """Flat signal (no dip) — should still return without crashing."""
        freq = np.linspace(2.8e9, 2.94e9, 200)
        signal = np.ones(200)
        # This may not give a meaningful fit but should not crash
        result = fit_odmr(freq, signal)
        assert "b_field" in result

    def test_fit_odmr_with_all_zeros(self):
        """Signal of all zeros."""
        freq = np.linspace(2.8e9, 2.94e9, 200)
        signal = np.zeros(200)
        result = fit_odmr(freq, signal)
        assert "b_field" in result

    def test_fit_odmr_with_negative_signal(self):
        """Negative signal values (unusual but possible)."""
        freq = np.linspace(2.8e9, 2.94e9, 300)
        data = generate_odmr_spectrum(freq, 0.5e-3, snr=100, seed=950)
        # Shift to all-negative
        signal = data["signal"] - 2.0
        result = fit_odmr(freq, signal)
        assert "b_field" in result

    def test_generate_odmr_very_high_contrast(self):
        """Contrast at 100%."""
        freq = np.linspace(2.8e9, 2.94e9, 200)
        data = generate_odmr_spectrum(freq, 0.5e-3, contrast=1.0, snr=0, seed=None)
        # Dips should reach near 0
        assert np.min(data["clean_signal"]) < 0.1

    def test_sensitivity_vs_tau_unknown_protocol(self):
        tau = np.linspace(1e-7, 1e-5, 50)
        with pytest.raises(ValueError, match="Unknown protocol"):
            sensitivity_vs_tau("quantum_tunneling", tau)

    def test_pipeline_rejects_bad_tau_range(self):
        cfg = PipelineConfig(protocol="ramsey", tau_range=(5, 1))  # reversed
        with pytest.raises(ValueError, match="tau_range"):
            validate_config(cfg)

    def test_pipeline_rejects_bad_lambda(self):
        cfg = PipelineConfig(protocol="ramsey", lambda_reg=-1)
        with pytest.raises(ValueError, match="lambda_reg"):
            validate_config(cfg)
