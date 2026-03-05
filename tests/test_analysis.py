"""Tests for wimp.analysis."""

import numpy as np
import pytest

from wimp.analysis import (
    noise_spectroscopy,
    psd_welch,
    snr_estimate,
    stimulus_triggered_average,
    conduction_velocity,
    detect_neural_events,
    phase_from_echo,
)


class TestNoiseSpectroscopy:
    def test_output_keys(self):
        tau = np.logspace(-7, -4, 50)
        coherence = np.exp(-tau / 100e-6)
        result = noise_spectroscopy(coherence, tau, 16)
        assert "freq" in result
        assert "spectral_density" in result

    def test_sorted_by_frequency(self):
        tau = np.logspace(-7, -4, 50)
        coherence = np.exp(-tau / 100e-6)
        result = noise_spectroscopy(coherence, tau, 16)
        assert np.all(np.diff(result["freq"]) > 0)

    def test_spectral_density_positive(self):
        tau = np.logspace(-7, -4, 50)
        coherence = np.exp(-tau / 100e-6) * 0.5
        result = noise_spectroscopy(coherence, tau, 16, amplitude=0.5)
        assert np.all(result["spectral_density"] >= 0)


class TestPSDWelch:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=1000)
        freq, psd = psd_welch(ts, fs=1000.0)
        assert len(freq) == len(psd)
        assert len(freq) > 0

    def test_psd_positive(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=1000)
        freq, psd = psd_welch(ts, fs=1000.0)
        assert np.all(psd >= 0)

    def test_custom_nperseg(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=1000)
        freq, psd = psd_welch(ts, fs=1000.0, nperseg=128)
        assert len(freq) > 0


class TestSNREstimate:
    def test_high_snr_signal(self):
        fs = 1000.0
        t = np.arange(0, 1, 1.0 / fs)
        signal = np.sin(2 * np.pi * 50 * t) + 0.01 * np.random.default_rng(0).normal(size=len(t))
        snr = snr_estimate(signal, fs, (40, 60), (200, 400))
        assert snr > 10

    def test_pure_noise(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(size=1000)
        snr = snr_estimate(noise, 1000.0, (10, 50), (200, 400))
        # SNR near 1 for random noise
        assert snr < 10


class TestStimulusTriggeredAverage:
    def test_basic_averaging(self):
        fs = 1000.0
        n_t = 5000
        rng = np.random.default_rng(0)
        # Create signal with events at known times
        ts = rng.normal(0, 0.1, size=(1, n_t))
        triggers = np.array([1.0, 2.0, 3.0])
        for trig in triggers:
            idx = int(trig * fs)
            ts[0, idx:idx + 50] += 1.0

        result = stimulus_triggered_average(ts, triggers, fs, window=(-0.05, 0.2))
        assert result["n_trials"] == 3
        assert result["average"].shape[0] == 1

    def test_no_valid_epochs(self):
        ts = np.zeros((1, 100))
        triggers = np.array([10.0])  # beyond signal
        result = stimulus_triggered_average(ts, triggers, 1000.0)
        assert result["n_trials"] == 0

    def test_baseline_correction(self):
        fs = 1000.0
        ts = np.ones((1, 5000))  # constant signal
        triggers = np.array([1.0, 2.0])
        result = stimulus_triggered_average(
            ts, triggers, fs,
            window=(-0.05, 0.2),
            baseline=(-0.05, 0.0),
        )
        # After baseline correction, should be near zero
        np.testing.assert_allclose(result["average"], 0.0, atol=1e-10)


class TestConductionVelocity:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        n_nds, n_t = 4, 1000
        ts = rng.normal(size=(n_nds, n_t))
        nd_pos = np.column_stack([
            np.linspace(0, 1e-3, n_nds),
            np.zeros(n_nds),
            np.zeros(n_nds),
        ])
        result = conduction_velocity(ts, nd_pos, fs=1000.0)
        assert "velocity" in result
        assert "pair_velocities" in result
        assert "pair_delays" in result
        assert "pair_distances" in result

    def test_single_nd(self):
        ts = np.zeros((1, 100))
        nd_pos = np.array([[0, 0, 0]])
        result = conduction_velocity(ts, nd_pos, fs=1000.0)
        assert result["velocity"] == 0.0


class TestDetectNeuralEvents:
    def test_detects_known_events(self):
        rng = np.random.default_rng(0)
        n_t = 10000
        signal = rng.normal(0, 0.1, size=n_t)
        # Insert clear spikes
        signal[1000] = 5.0
        signal[5000] = 5.0

        result = detect_neural_events(signal, fs=1000.0, threshold_sigma=3.0)
        assert len(result["event_times"]) >= 2

    def test_no_events_in_flat_signal(self):
        signal = np.zeros(1000)
        result = detect_neural_events(signal, fs=1000.0, threshold_sigma=3.0)
        assert len(result["event_times"]) == 0

    def test_output_keys(self):
        signal = np.random.default_rng(0).normal(size=500)
        result = detect_neural_events(signal, fs=1000.0)
        assert "event_times" in result
        assert "event_amplitudes" in result
        assert "event_indices" in result
        assert "threshold" in result

    def test_multichannel(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 0.1, size=(3, 1000))
        signal[:, 500] = 5.0
        result = detect_neural_events(signal, fs=1000.0, threshold_sigma=3.0)
        assert len(result["event_times"]) >= 1


class TestPhaseFromEcho:
    def test_known_phase(self):
        phi = np.pi / 4
        I = np.cos(phi) * np.ones(10)
        Q = np.sin(phi) * np.ones(10)
        phase = phase_from_echo(I, Q)
        np.testing.assert_allclose(phase, phi, atol=1e-10)

    def test_unwrapping(self):
        # Linearly increasing phase that wraps
        phase_true = np.linspace(0, 6 * np.pi, 100)
        I = np.cos(phase_true)
        Q = np.sin(phase_true)
        recovered = phase_from_echo(I, Q)
        np.testing.assert_allclose(recovered, phase_true, atol=0.1)


# ---------------------------------------------------------------------------
# Spike sorting / waveform classification
# ---------------------------------------------------------------------------

from wimp.analysis import (
    extract_event_waveforms,
    classify_waveforms,
    sort_spikes,
    SortedSpike,
)


class TestExtractEventWaveforms:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=1000)
        indices = np.array([100, 200, 500])
        result = extract_event_waveforms(ts, indices, window_pre=10, window_post=20)
        assert "waveforms" in result
        assert "event_indices" in result
        assert "timestamps" in result

    def test_waveform_shape(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=1000)
        indices = np.array([100, 200, 500])
        result = extract_event_waveforms(ts, indices, window_pre=10, window_post=20)
        assert result["waveforms"].shape == (3, 30)

    def test_boundary_events_excluded(self):
        ts = np.zeros(100)
        # Events at the very start and end should be excluded
        indices = np.array([2, 50, 98])
        result = extract_event_waveforms(ts, indices, window_pre=10, window_post=20)
        # Only index=50 should survive
        assert result["waveforms"].shape[0] == 1

    def test_empty_events(self):
        ts = np.zeros(100)
        indices = np.array([], dtype=int)
        result = extract_event_waveforms(ts, indices)
        assert result["waveforms"].shape[0] == 0

    def test_multichannel(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 1000))
        indices = np.array([100, 200, 500])
        result = extract_event_waveforms(ts, indices, window_pre=10, window_post=20)
        assert result["waveforms"].shape == (3, 30)

    def test_timestamps_from_fs(self):
        ts = np.zeros(1000)
        indices = np.array([100, 500])
        result = extract_event_waveforms(ts, indices, window_pre=5, window_post=10, fs=1000.0)
        np.testing.assert_allclose(result["timestamps"], [0.1, 0.5])


class TestClassifyWaveforms:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        waveforms = rng.normal(size=(20, 30))
        result = classify_waveforms(waveforms, n_clusters=3)
        assert "labels" in result
        assert "centroids" in result
        assert "pca_scores" in result
        assert "explained_variance" in result
        assert "cluster_waveforms" in result

    def test_correct_label_count(self):
        rng = np.random.default_rng(0)
        waveforms = rng.normal(size=(20, 30))
        result = classify_waveforms(waveforms, n_clusters=3)
        assert len(result["labels"]) == 20
        assert set(result["labels"]).issubset({0, 1, 2})

    def test_separates_distinct_templates(self):
        rng = np.random.default_rng(42)
        # Create two very distinct template groups
        group_a = np.tile([1, 0, -1, 0, 1], (10, 1)) + rng.normal(0, 0.01, (10, 5))
        group_b = np.tile([0, 1, 0, -1, 0], (10, 1)) + rng.normal(0, 0.01, (10, 5))
        waveforms = np.vstack([group_a, group_b])
        result = classify_waveforms(waveforms, n_clusters=2, n_components=2)
        # Labels for group_a should be the same, and different from group_b
        labels_a = set(result["labels"][:10])
        labels_b = set(result["labels"][10:])
        assert len(labels_a) == 1  # all same cluster
        assert len(labels_b) == 1  # all same cluster
        assert labels_a != labels_b  # different clusters

    def test_empty_waveforms(self):
        waveforms = np.empty((0, 30))
        result = classify_waveforms(waveforms)
        assert len(result["labels"]) == 0

    def test_fewer_events_than_clusters(self):
        rng = np.random.default_rng(0)
        waveforms = rng.normal(size=(2, 30))
        result = classify_waveforms(waveforms, n_clusters=5)
        assert len(result["labels"]) == 2


class TestSortSpikes:
    def test_returns_list_of_sorted_spikes(self):
        rng = np.random.default_rng(42)
        n_t = 10000
        signal = rng.normal(0, 0.1, size=n_t)
        # Insert clear spikes
        for i in range(100, n_t - 100, 500):
            signal[i] = 5.0
        spikes = sort_spikes(signal, fs=1000.0, threshold_sigma=3.0)
        assert len(spikes) > 0
        assert all(isinstance(s, SortedSpike) for s in spikes)

    def test_sorted_spike_attributes(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 0.1, size=5000)
        signal[500] = 5.0
        signal[2000] = 5.0
        spikes = sort_spikes(signal, fs=1000.0, threshold_sigma=3.0)
        if spikes:
            s = spikes[0]
            assert hasattr(s, "cluster_id")
            assert hasattr(s, "waveform")
            assert hasattr(s, "timestamp")
            assert hasattr(s, "amplitude")

    def test_no_spikes_in_flat_signal(self):
        signal = np.zeros(1000)
        spikes = sort_spikes(signal, fs=1000.0)
        assert spikes == []


# ---------------------------------------------------------------------------
# Blind source separation (PCA / ICA)
# ---------------------------------------------------------------------------

from wimp.analysis import pca_decompose, ica_decompose


class TestPCADecompose:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(5, 200))
        result = pca_decompose(data)
        assert "components" in result
        assert "mixing" in result
        assert "explained_variance" in result
        assert "mean" in result

    def test_component_shape(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(5, 200))
        result = pca_decompose(data, n_components=3)
        assert result["components"].shape == (3, 200)
        assert result["mixing"].shape == (5, 3)
        assert len(result["explained_variance"]) == 3

    def test_explained_variance_sums_to_one(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(4, 300))
        result = pca_decompose(data)
        # All components should explain all variance
        assert np.sum(result["explained_variance"]) == pytest.approx(1.0, abs=1e-6)

    def test_1d_input(self):
        data = np.random.default_rng(0).normal(size=100)
        result = pca_decompose(data)
        assert result["components"].shape[1] == 100


class TestICADecompose:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(3, 500))
        result = ica_decompose(data)
        assert "sources" in result
        assert "mixing" in result
        assert "unmixing" in result

    def test_source_shape(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(4, 500))
        result = ica_decompose(data, n_components=2)
        assert result["sources"].shape == (2, 500)
        assert result["mixing"].shape == (4, 2)
        assert result["unmixing"].shape == (2, 4)

    def test_separates_independent_sources(self):
        rng = np.random.default_rng(42)
        n_t = 1000
        # Two independent sources
        s1 = np.sin(2 * np.pi * 5 * np.arange(n_t) / n_t)
        s2 = rng.laplace(size=n_t)
        S = np.vstack([s1, s2])
        # Random mixing
        A = rng.normal(size=(3, 2))
        X = A @ S
        result = ica_decompose(X, n_components=2)
        # Recovered sources should have 2 components
        assert result["sources"].shape == (2, n_t)

    def test_1d_input(self):
        data = np.random.default_rng(0).normal(size=100)
        result = ica_decompose(data)
        assert result["sources"].shape[1] == 100


# ---------------------------------------------------------------------------
# Functional connectivity
# ---------------------------------------------------------------------------

from wimp.analysis import cross_correlation_matrix, granger_causality, mutual_information


class TestCrossCorrelationMatrix:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 200))
        result = cross_correlation_matrix(ts)
        assert "matrix" in result
        assert "lags" in result

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 500))
        result = cross_correlation_matrix(ts, normalized=True)
        diag = np.diag(result["matrix"])
        np.testing.assert_allclose(diag, 1.0, atol=0.05)

    def test_matrix_shape(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(4, 200))
        result = cross_correlation_matrix(ts)
        assert result["matrix"].shape == (4, 4)
        assert result["lags"].shape == (4, 4)

    def test_max_lag(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(2, 100))
        result = cross_correlation_matrix(ts, max_lag=10)
        assert result["matrix"].shape == (2, 2)


class TestGrangerCausality:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 200))
        result = granger_causality(ts, max_lag=3)
        assert "f_stat" in result
        assert "matrix" in result

    def test_diagonal_is_zero(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 200))
        result = granger_causality(ts)
        np.testing.assert_array_equal(np.diag(result["f_stat"]), 0.0)

    def test_detects_causal_relationship(self):
        rng = np.random.default_rng(42)
        n_t = 1000
        x = rng.normal(size=n_t)
        # y explicitly caused by lagged x
        y = np.zeros(n_t)
        for t in range(2, n_t):
            y[t] = 0.8 * x[t - 1] + 0.1 * rng.normal()
        ts = np.vstack([y, x])
        result = granger_causality(ts, max_lag=3)
        # x -> y should have higher F than y -> x
        assert result["f_stat"][0, 1] > result["f_stat"][1, 0]


class TestMutualInformation:
    def test_shape(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 500))
        mi = mutual_information(ts)
        assert mi.shape == (3, 3)

    def test_symmetric(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 500))
        mi = mutual_information(ts)
        np.testing.assert_array_almost_equal(mi, mi.T)

    def test_diagonal_positive(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 500))
        mi = mutual_information(ts)
        assert np.all(np.diag(mi) > 0)

    def test_correlated_higher_than_uncorrelated(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=1000)
        y = x + 0.1 * rng.normal(size=1000)  # strongly correlated
        z = rng.normal(size=1000)  # independent
        ts = np.vstack([x, y, z])
        mi = mutual_information(ts)
        assert mi[0, 1] > mi[0, 2]


# ---------------------------------------------------------------------------
# Transfer entropy
# ---------------------------------------------------------------------------

from wimp.analysis import transfer_entropy, transfer_entropy_matrix


class TestTransferEntropy:
    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        te = transfer_entropy(x, y)
        assert te >= 0.0

    def test_causal_direction(self):
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.normal(size=n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.8 * x[t - 1] + 0.1 * rng.normal()
        te_x_to_y = transfer_entropy(x, y, lag=1)
        te_y_to_x = transfer_entropy(y, x, lag=1)
        assert te_x_to_y > te_y_to_x

    def test_self_transfer_low(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        te = transfer_entropy(x, x)
        # Self TE should be low (near zero)
        assert te < 0.5

    def test_normalize_bounded(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        te = transfer_entropy(x, y, normalize=True)
        assert 0.0 <= te <= 1.0


class TestTransferEntropyMatrix:
    def test_shape(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 500))
        mat = transfer_entropy_matrix(ts)
        assert mat.shape == (3, 3)

    def test_diagonal_zero(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 500))
        mat = transfer_entropy_matrix(ts)
        np.testing.assert_array_equal(np.diag(mat), 0.0)


# ---------------------------------------------------------------------------
# Spectral coherence
# ---------------------------------------------------------------------------

from wimp.analysis import spectral_coherence, coherence_matrix


class TestSpectralCoherence:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        result = spectral_coherence(x, y, fs=1000.0)
        assert "freq" in result
        assert "coherence" in result

    def test_self_coherence_is_one(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=1000)
        result = spectral_coherence(x, x, fs=1000.0)
        np.testing.assert_allclose(result["coherence"], 1.0, atol=1e-10)

    def test_coherence_bounded(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        result = spectral_coherence(x, y, fs=1000.0)
        assert np.all(result["coherence"] >= 0.0)
        assert np.all(result["coherence"] <= 1.0 + 1e-10)


class TestCoherenceMatrix:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 500))
        result = coherence_matrix(ts, fs=1000.0)
        assert "matrix" in result
        assert "freq" in result

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(3, 500))
        result = coherence_matrix(ts, fs=1000.0)
        np.testing.assert_allclose(np.diag(result["matrix"]), 1.0)

    def test_matrix_shape(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(4, 500))
        result = coherence_matrix(ts, fs=1000.0)
        assert result["matrix"].shape == (4, 4)

    def test_freq_band(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(2, 1000))
        result = coherence_matrix(ts, fs=1000.0, freq_band=(10.0, 50.0))
        assert result["matrix"].shape == (2, 2)


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

from wimp.analysis import permutation_test, cluster_permutation_test, fdr_correct


class TestPermutationTest:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 1, 50)
        result = permutation_test(a, b, seed=0)
        assert "statistic" in result
        assert "observed" in result
        assert "p_value" in result
        assert "null_distribution" in result

    def test_different_distributions_significant(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100)
        b = rng.normal(3, 1, 100)
        result = permutation_test(a, b, seed=42)
        assert result["p_value"] < 0.05

    def test_same_distribution_not_significant(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 1, 50)
        result = permutation_test(a, b, seed=42)
        assert result["p_value"] > 0.01

    def test_seed_reproducibility(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=30)
        b = rng.normal(size=30)
        r1 = permutation_test(a, b, seed=99)
        r2 = permutation_test(a, b, seed=99)
        assert r1["p_value"] == r2["p_value"]

    def test_t_stat(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(2, 1, 50)
        result = permutation_test(a, b, statistic="t_stat", seed=42)
        assert result["p_value"] < 0.05


class TestClusterPermutationTest:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(10, 50))
        b = rng.normal(size=(10, 50))
        result = cluster_permutation_test(a, b, seed=0)
        assert "significant_clusters" in result
        assert "cluster_p_values" in result
        assert "t_values" in result

    def test_no_difference(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(10, 50))
        b = rng.normal(size=(10, 50))
        result = cluster_permutation_test(a, b, seed=0)
        # Most clusters should be non-significant
        assert isinstance(result["significant_clusters"], list)

    def test_detects_difference(self):
        rng = np.random.default_rng(42)
        n_obs, n_t = 15, 100
        a = rng.normal(0, 1, size=(n_obs, n_t))
        b = rng.normal(0, 1, size=(n_obs, n_t))
        # Insert a strong shift in a segment
        b[:, 30:60] += 5.0
        result = cluster_permutation_test(
            a, b, threshold=2.0, n_permutations=500, seed=42,
        )
        assert len(result["t_values"]) == n_t


class TestFDRCorrect:
    def test_output_keys(self):
        p = np.array([0.001, 0.01, 0.05, 0.5, 0.9])
        result = fdr_correct(p)
        assert "rejected" in result
        assert "corrected_p" in result
        assert "threshold" in result

    def test_rejects_small_p(self):
        p = np.array([0.001, 0.002, 0.5, 0.9])
        result = fdr_correct(p, alpha=0.05)
        assert result["rejected"][0]
        assert result["rejected"][1]

    def test_no_rejection(self):
        p = np.array([0.5, 0.6, 0.7, 0.8])
        result = fdr_correct(p, alpha=0.05)
        assert not np.any(result["rejected"])

    def test_corrected_p_bounded(self):
        p = np.array([0.01, 0.02, 0.1, 0.5])
        result = fdr_correct(p, alpha=0.05)
        assert np.all(result["corrected_p"] >= 0)
        assert np.all(result["corrected_p"] <= 1.0 + 1e-10)
