"""Tests for wimp.pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from wimp.pipeline import (
    PipelineConfig,
    default_config,
    validate_config,
    run_pipeline,
    batch_process,
)
from wimp.io import WIMPDataset, save_hdf5
from wimp.synthetic import generate_full_experiment


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.protocol == "ramsey"
        assert config.lambda_reg == 0.1
        assert config.source_localization is True

    def test_custom(self):
        config = PipelineConfig(protocol="echo", lambda_reg=0.5)
        assert config.protocol == "echo"
        assert config.lambda_reg == 0.5


class TestDefaultConfig:
    @pytest.mark.parametrize("protocol", ["ramsey", "echo", "t1", "dd"])
    def test_valid_protocols(self, protocol):
        config = default_config(protocol)
        assert config.protocol == protocol

    def test_hahn_echo_alias(self):
        config = default_config("hahn_echo")
        assert config.protocol == "echo"

    def test_unknown_protocol(self):
        with pytest.raises(ValueError):
            default_config("invalid")


class TestValidateConfig:
    def test_valid_config(self):
        config = PipelineConfig()
        validate_config(config)  # should not raise

    def test_invalid_protocol(self):
        config = PipelineConfig(protocol="invalid")
        with pytest.raises(ValueError, match="protocol"):
            validate_config(config)

    def test_invalid_lambda(self):
        config = PipelineConfig(lambda_reg=-1)
        with pytest.raises(ValueError, match="lambda_reg"):
            validate_config(config)

    def test_invalid_tau_range(self):
        config = PipelineConfig(tau_range=(5e-6, 0.0))
        with pytest.raises(ValueError, match="tau_range"):
            validate_config(config)


class TestRunPipeline:
    def test_with_dataset(self):
        exp = generate_full_experiment(
            n_nds=4, n_neurons=2, n_timepoints=50,
            protocol="ramsey", seed=42,
        )
        ds = WIMPDataset(
            protocol="ramsey",
            tau_array=exp["tau_array"],
            signal=exp["signal"],
            nd_positions=exp["nd_positions"],
        )
        config = PipelineConfig(
            protocol="ramsey",
            source_localization=False,
        )
        results = run_pipeline(config, dataset=ds)
        assert results["n_nds"] == 4
        assert len(results["fits"]) == 4
        assert results["field_timeseries"] is not None

    def test_with_source_localization(self):
        exp = generate_full_experiment(
            n_nds=6, n_neurons=3, n_timepoints=50,
            protocol="ramsey", seed=42,
        )
        ds = WIMPDataset(
            protocol="ramsey",
            tau_array=exp["tau_array"],
            signal=exp["signal"],
            nd_positions=exp["nd_positions"],
        )
        config = PipelineConfig(
            protocol="ramsey",
            source_localization=True,
            registration_params={"neuron_positions": exp["neuron_positions"].tolist()},
        )
        results = run_pipeline(config, dataset=ds)
        assert results["source_estimate"] is not None

    def test_from_file(self):
        exp = generate_full_experiment(n_nds=3, protocol="ramsey", seed=42)
        ds = WIMPDataset(
            protocol="ramsey",
            tau_array=exp["tau_array"],
            signal=exp["signal"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.h5"
            save_hdf5(ds, path)
            config = PipelineConfig(protocol="ramsey", source_localization=False)
            results = run_pipeline(config, data_path=path)
            assert results["n_nds"] == 3

    def test_requires_data(self):
        config = PipelineConfig()
        with pytest.raises(ValueError, match="Provide either"):
            run_pipeline(config)

    def test_1d_signal(self):
        tau = np.linspace(0, 5e-6, 100)
        sig = np.cos(2 * np.pi * 1e6 * tau)  # 1-D signal
        ds = WIMPDataset(protocol="ramsey", tau_array=tau, signal=sig)
        config = PipelineConfig(protocol="ramsey", source_localization=False)
        results = run_pipeline(config, dataset=ds)
        assert results["n_nds"] == 1

    def test_echo_protocol(self):
        exp = generate_full_experiment(
            n_nds=3, n_neurons=2, protocol="echo", seed=42,
        )
        ds = WIMPDataset(
            protocol="echo",
            tau_array=exp["tau_array"],
            signal=exp["signal"],
        )
        config = PipelineConfig(protocol="echo", source_localization=False)
        results = run_pipeline(config, dataset=ds)
        assert len(results["fits"]) == 3

    def test_save_intermediates(self):
        exp = generate_full_experiment(n_nds=2, protocol="ramsey", seed=42)
        ds = WIMPDataset(
            protocol="ramsey",
            tau_array=exp["tau_array"],
            signal=exp["signal"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "out"
            config = PipelineConfig(
                protocol="ramsey",
                source_localization=False,
                save_intermediates=True,
                output_dir=str(outdir),
            )
            run_pipeline(config, dataset=ds)
            assert (outdir / "results.json").exists()


class TestBatchProcess:
    def test_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                exp = generate_full_experiment(n_nds=2, protocol="ramsey", seed=i)
                ds = WIMPDataset(
                    protocol="ramsey",
                    tau_array=exp["tau_array"],
                    signal=exp["signal"],
                )
                p = Path(tmpdir) / f"data_{i}.h5"
                save_hdf5(ds, p)
                paths.append(p)

            config = PipelineConfig(protocol="ramsey", source_localization=False)
            results = batch_process(config, paths)
            assert len(results) == 3


class TestPipelineDeformable:
    def test_deformable_flag(self):
        exp = generate_full_experiment(
            n_nds=6, n_neurons=3, n_timepoints=50,
            protocol="ramsey", seed=42,
        )
        ds = WIMPDataset(
            protocol="ramsey",
            tau_array=exp["tau_array"],
            signal=exp["signal"],
            nd_positions=exp["nd_positions"],
        )
        config = PipelineConfig(
            protocol="ramsey",
            source_localization=False,
            deformable=True,
        )
        results = run_pipeline(config, dataset=ds)
        assert results["deformable_registration"] is not None

    def test_compute_crosstalk_with_source_loc(self):
        exp = generate_full_experiment(
            n_nds=6, n_neurons=3, n_timepoints=50,
            protocol="ramsey", seed=42,
        )
        ds = WIMPDataset(
            protocol="ramsey",
            tau_array=exp["tau_array"],
            signal=exp["signal"],
            nd_positions=exp["nd_positions"],
        )
        config = PipelineConfig(
            protocol="ramsey",
            source_localization=True,
            compute_crosstalk=True,
            registration_params={"neuron_positions": exp["neuron_positions"].tolist()},
        )
        results = run_pipeline(config, dataset=ds)
        assert results["resolution_matrix"] is not None
        assert results["crosstalk_metrics"] is not None
        assert "diagonal" in results["crosstalk_metrics"]

    def test_config_defaults(self):
        config = PipelineConfig()
        assert config.deformable is False
        assert config.compute_crosstalk is False


# ---------------------------------------------------------------------------
# Parallel batch processing
# ---------------------------------------------------------------------------

from wimp.pipeline import batch_process_parallel


class TestBatchProcessParallel:
    def test_parallel_produces_same_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                exp = generate_full_experiment(n_nds=2, protocol="ramsey", seed=i)
                ds = WIMPDataset(
                    protocol="ramsey",
                    tau_array=exp["tau_array"],
                    signal=exp["signal"],
                )
                p = Path(tmpdir) / f"data_{i}.h5"
                save_hdf5(ds, p)
                paths.append(p)

            config = PipelineConfig(protocol="ramsey", source_localization=False)
            results = batch_process_parallel(config, paths)
            assert len(results) == 3

    def test_progress_callback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(2):
                exp = generate_full_experiment(n_nds=2, protocol="ramsey", seed=i)
                ds = WIMPDataset(
                    protocol="ramsey",
                    tau_array=exp["tau_array"],
                    signal=exp["signal"],
                )
                p = Path(tmpdir) / f"data_{i}.h5"
                save_hdf5(ds, p)
                paths.append(p)

            progress = []
            config = PipelineConfig(protocol="ramsey", source_localization=False)
            batch_process_parallel(
                config, paths,
                progress_callback=lambda done, total: progress.append((done, total)),
            )
            assert len(progress) == 2
            assert progress[-1] == (2, 2)

    def test_empty_paths(self):
        config = PipelineConfig(protocol="ramsey", source_localization=False)
        results = batch_process_parallel(config, [])
        assert results == []

    def test_save_intermediates_subdirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(2):
                exp = generate_full_experiment(n_nds=2, protocol="ramsey", seed=i)
                ds = WIMPDataset(
                    protocol="ramsey",
                    tau_array=exp["tau_array"],
                    signal=exp["signal"],
                )
                p = Path(tmpdir) / f"data_{i}.h5"
                save_hdf5(ds, p)
                paths.append(p)

            outdir = Path(tmpdir) / "output"
            config = PipelineConfig(
                protocol="ramsey",
                source_localization=False,
                save_intermediates=True,
                output_dir=str(outdir),
            )
            batch_process_parallel(config, paths)
            # Each dataset should have its own subdirectory
            assert (outdir / "data_0" / "results.json").exists()
            assert (outdir / "data_1" / "results.json").exists()


# ---------------------------------------------------------------------------
# Pipeline denoising integration
# ---------------------------------------------------------------------------


class TestPipelineDenoising:
    def test_denoise_in_pipeline(self):
        exp = generate_full_experiment(
            n_nds=3, n_neurons=2, n_timepoints=50,
            protocol="ramsey", seed=42,
        )
        ds = WIMPDataset(
            protocol="ramsey",
            tau_array=exp["tau_array"],
            signal=exp["signal"],
        )
        config = PipelineConfig(
            protocol="ramsey",
            source_localization=False,
            denoise_method="savgol",
            denoise_params={"window_length": 11},
        )
        results = run_pipeline(config, dataset=ds)
        assert len(results["fits"]) == 3

    def test_denoise_config_defaults(self):
        config = PipelineConfig()
        assert config.denoise_method is None
        assert config.denoise_params == {}


class TestCWODMRPipeline:
    def test_cw_odmr_in_valid_protocols(self):
        config = PipelineConfig(protocol="cw_odmr")
        validate_config(config)  # should not raise

    def test_cw_odmr_default_config(self):
        config = default_config("cw_odmr")
        assert config.protocol == "cw_odmr"

    def test_run_pipeline_cw_odmr(self):
        from wimp.synthetic import generate_odmr_spectrum
        from wimp.constants import D0, GAMMA_NV

        b = 1e-3
        freq = np.linspace(2.78e9, 2.96e9, 500)
        data = generate_odmr_spectrum(freq, b, snr=100, seed=42)

        ds = WIMPDataset(
            protocol="cw_odmr",
            tau_array=freq,
            signal=data["signal"],
        )
        config = PipelineConfig(
            protocol="cw_odmr",
            tau_range=(2.78e9, 2.96e9),
            source_localization=False,
        )
        results = run_pipeline(config, dataset=ds)
        assert len(results["fits"]) == 1
        assert results["field_timeseries"] is not None

