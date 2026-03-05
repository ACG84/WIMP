"""Tests for wimp.pulses."""

import math

import numpy as np
import pytest

from wimp.pulses import (
    PulseSequence,
    ramsey_sequence,
    hahn_echo_sequence,
    cpmg_sequence,
    xy8_sequence,
    filter_function,
    filter_peak_frequency,
    accumulated_phase,
    optimal_tau,
)


class TestSequenceFactories:
    def test_ramsey_basic(self):
        seq = ramsey_sequence(1e-6)
        assert seq.name == "ramsey"
        assert seq.n_pi_pulses == 0
        assert len(seq.pulse_phases) == 0
        assert seq.n_blocks == 1

    def test_ramsey_total_sensing_time(self):
        tau = 2e-6
        seq = ramsey_sequence(tau)
        assert float(seq.total_sensing_time) == pytest.approx(tau)

    def test_hahn_echo_basic(self):
        seq = hahn_echo_sequence(1e-6)
        assert seq.name == "hahn_echo"
        assert seq.n_pi_pulses == 1
        assert len(seq.pulse_phases) == 1

    def test_hahn_echo_total_time(self):
        tau = 5e-6
        seq = hahn_echo_sequence(tau)
        assert float(seq.total_sensing_time) == pytest.approx(2 * tau)

    def test_cpmg_basic(self):
        seq = cpmg_sequence(1e-6, 8)
        assert seq.name == "cpmg"
        assert seq.n_pi_pulses == 8
        assert len(seq.pulse_phases) == 8
        assert all(p == 0.0 for p in seq.pulse_phases)

    def test_cpmg_total_time(self):
        tau, n = 2e-6, 16
        seq = cpmg_sequence(tau, n)
        assert float(seq.total_sensing_time) == pytest.approx(2 * n * tau)

    def test_cpmg_requires_positive_pulses(self):
        with pytest.raises(ValueError):
            cpmg_sequence(1e-6, 0)

    def test_xy8_basic(self):
        seq = xy8_sequence(1e-6, n_blocks=2)
        assert seq.name == "xy8"
        assert seq.n_pi_pulses == 16
        assert len(seq.pulse_phases) == 16
        assert seq.n_blocks == 2

    def test_xy8_single_block_phases(self):
        seq = xy8_sequence(1e-6, n_blocks=1)
        expected = [0.0, math.pi / 2, 0.0, math.pi / 2,
                    math.pi / 2, 0.0, math.pi / 2, 0.0]
        np.testing.assert_allclose(seq.pulse_phases, expected)

    def test_xy8_requires_positive_blocks(self):
        with pytest.raises(ValueError):
            xy8_sequence(1e-6, 0)

    def test_ramsey_array_tau(self):
        tau = np.array([1e-6, 2e-6, 3e-6])
        seq = ramsey_sequence(tau)
        np.testing.assert_array_equal(seq.tau, tau)


class TestFilterFunction:
    def test_ramsey_filter_dc_peak(self):
        seq = ramsey_sequence(1e-6)
        freq = np.linspace(1e3, 1e8, 1000)
        W = filter_function(seq, freq)
        # Ramsey should have max near DC (low freq)
        assert W[0] > W[-1]

    def test_echo_filter_peak_location(self):
        tau = 10e-6
        seq = hahn_echo_sequence(tau)
        f_peak_expected = 1.0 / (2.0 * tau)  # 50 kHz
        freq = np.linspace(1e3, 200e3, 10000)
        W = filter_function(seq, freq)
        peak_idx = np.argmax(W)
        f_peak_actual = freq[peak_idx]
        assert abs(f_peak_actual - f_peak_expected) / f_peak_expected < 0.05

    def test_cpmg_filter_shape(self):
        seq = cpmg_sequence(5e-6, 8)
        freq = np.linspace(1e3, 1e6, 5000)
        W = filter_function(seq, freq)
        assert W.shape == freq.shape
        assert np.all(W >= 0)

    def test_xy8_filter_shape(self):
        seq = xy8_sequence(5e-6, n_blocks=1)
        freq = np.linspace(1e3, 1e6, 5000)
        W = filter_function(seq, freq)
        assert W.shape == freq.shape
        assert np.all(W >= 0)

    def test_unknown_sequence_raises(self):
        seq = PulseSequence(name="unknown", tau=1e-6)
        with pytest.raises(ValueError, match="Unknown sequence"):
            filter_function(seq, np.array([1e6]))

    def test_filter_normalized(self):
        seq = cpmg_sequence(5e-6, 16)
        freq = np.linspace(1e3, 1e6, 10000)
        W = filter_function(seq, freq)
        assert np.max(W) == pytest.approx(1.0, abs=0.01)


class TestFilterPeakFrequency:
    def test_ramsey_peak_is_dc(self):
        seq = ramsey_sequence(1e-6)
        assert filter_peak_frequency(seq) == 0.0

    def test_echo_peak_frequency(self):
        tau = 10e-6
        seq = hahn_echo_sequence(tau)
        expected = 1.0 / (2.0 * tau)
        assert filter_peak_frequency(seq) == pytest.approx(expected)


class TestAccumulatedPhase:
    def test_constant_field_ramsey(self):
        B0 = 1e-6  # 1 uT
        tau = 1e-6
        seq = ramsey_sequence(tau)
        phi = accumulated_phase(seq, lambda t: np.full_like(t, B0), dt=1e-10)
        expected = 2.0 * np.pi * 28.024e9 * B0 * tau
        assert phi == pytest.approx(expected, rel=0.01)

    def test_echo_cancels_dc(self):
        B0 = 1e-6
        tau = 5e-6
        seq = hahn_echo_sequence(tau)
        # Constant field should be refocused by echo
        phi = accumulated_phase(seq, lambda t: np.full_like(t, B0), dt=1e-10)
        assert abs(phi) < 0.01 * (2 * np.pi * 28.024e9 * B0 * 2 * tau)


class TestOptimalTau:
    def test_ramsey_optimal(self):
        t2star = 2e-6
        assert optimal_tau("ramsey", t2star=t2star) == pytest.approx(t2star)

    def test_echo_with_target_freq(self):
        freq = 50e3
        result = optimal_tau("hahn_echo", target_freq=freq)
        assert result == pytest.approx(1.0 / (2 * freq))

    def test_echo_with_t2(self):
        t2 = 100e-6
        result = optimal_tau("hahn_echo", t2=t2)
        assert result == pytest.approx(t2 / 2)

    def test_ramsey_requires_t2star(self):
        with pytest.raises(ValueError):
            optimal_tau("ramsey")

    def test_echo_requires_params(self):
        with pytest.raises(ValueError):
            optimal_tau("hahn_echo")

    def test_unknown_protocol(self):
        with pytest.raises(ValueError):
            optimal_tau("invalid")
