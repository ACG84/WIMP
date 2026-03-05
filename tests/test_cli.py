"""Tests for wimp.cli."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from wimp.cli import main, _build_parser
from wimp.io import WIMPDataset, save_hdf5
from wimp.synthetic import generate_full_experiment


class TestParser:
    def test_parser_builds(self):
        parser = _build_parser()
        assert parser is not None

    def test_version_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_run_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "data.h5", "-p", "echo"])
        assert args.command == "run"
        assert args.protocol == "echo"

    def test_synthetic_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["synthetic", "--n-nds", "5", "--snr", "40"])
        assert args.command == "synthetic"
        assert args.n_nds == 5
        assert args.snr == 40.0

    def test_sensitivity_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["sensitivity", "--contrast", "0.1"])
        assert args.command == "sensitivity"
        assert args.contrast == 0.1

    def test_info_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["info", "data.h5"])
        assert args.command == "info"
        assert args.data == "data.h5"


class TestMainVersion:
    def test_version_prints(self, capsys):
        main(["--version"])
        captured = capsys.readouterr()
        assert "wimp" in captured.out

    def test_no_args_shows_help(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "WIMP" in captured.out


class TestMainRun:
    def test_run_from_hdf5(self, capsys):
        exp = generate_full_experiment(n_nds=3, protocol="ramsey", seed=42)
        ds = WIMPDataset(
            protocol="ramsey",
            tau_array=exp["tau_array"],
            signal=exp["signal"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.h5"
            save_hdf5(ds, path)
            main(["run", str(path), "--no-source-loc"])
            captured = capsys.readouterr()
            assert "Pipeline complete" in captured.out


class TestMainSynthetic:
    def test_synthetic_hdf5(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "synth.h5"
            main(["synthetic", "-o", str(path), "--n-nds", "3", "--seed", "0"])
            captured = capsys.readouterr()
            assert "Saved" in captured.out
            assert path.exists()

    def test_synthetic_npz(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "synth_dir"
            main(["synthetic", "-o", str(path), "--n-nds", "3", "--seed", "0"])
            captured = capsys.readouterr()
            assert "Saved" in captured.out


class TestMainSensitivity:
    def test_sensitivity_output(self, capsys):
        main(["sensitivity"])
        captured = capsys.readouterr()
        assert "Protocol" in captured.out or "Ramsey" in captured.out

    def test_sensitivity_csv(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sens.csv"
            main(["sensitivity", "-o", str(path)])
            assert path.exists()


class TestMainInfo:
    def test_info_hdf5(self, capsys):
        exp = generate_full_experiment(n_nds=3, protocol="ramsey", seed=42)
        ds = WIMPDataset(
            protocol="ramsey",
            tau_array=exp["tau_array"],
            signal=exp["signal"],
            metadata={"test": True},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.h5"
            save_hdf5(ds, path)
            main(["info", str(path)])
            captured = capsys.readouterr()
            assert "ramsey" in captured.out


class TestParserNewFlags:
    def test_deformable_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "data.h5", "--deformable"])
        assert args.deformable is True

    def test_compute_crosstalk_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "data.h5", "--compute-crosstalk"])
        assert args.compute_crosstalk is True

    def test_crosstalk_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["crosstalk", "--n-nds", "8", "--n-neurons", "4"])
        assert args.command == "crosstalk"
        assert args.n_nds == 8
        assert args.n_neurons == 4

    def test_crosstalk_lambda(self):
        parser = _build_parser()
        args = parser.parse_args(["crosstalk", "--lambda-reg", "0.5"])
        assert args.lambda_reg == 0.5


class TestMainCrosstalk:
    def test_crosstalk_output(self, capsys):
        main(["crosstalk", "--n-nds", "6", "--n-neurons", "3"])
        captured = capsys.readouterr()
        assert "Crosstalk Analysis" in captured.out
        assert "self-res" in captured.out
