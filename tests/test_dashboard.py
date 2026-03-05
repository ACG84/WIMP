"""Tests for the WIMP dashboard module."""

import numpy as np
import pytest

pn = pytest.importorskip("panel")

from wimp.dashboard import WIMPDashboard, _format_results, _format_dataset_info
from wimp.io import WIMPDataset


class TestWIMPDashboard:
    def test_construction_default(self):
        dashboard = WIMPDashboard()
        assert dashboard.config is not None
        assert dashboard.config.protocol == "ramsey"

    def test_construction_with_config(self):
        from wimp.pipeline import PipelineConfig
        cfg = PipelineConfig(protocol="echo", lambda_reg=0.5)
        dashboard = WIMPDashboard(config=cfg)
        assert dashboard.config.protocol == "echo"
        assert dashboard.config.lambda_reg == 0.5

    def test_servable_returns_template(self):
        dashboard = WIMPDashboard()
        tmpl = dashboard.servable()
        assert tmpl is not None
        assert hasattr(tmpl, "main")


class TestFormatResults:
    def test_basic(self):
        results = {
            "n_nds": 3,
            "fits": [{}, {}, {}],
            "field_timeseries": np.array([1e-6, 2e-6, 3e-6]),
            "config": {"protocol": "ramsey", "lambda_reg": 0.1},
        }
        text = _format_results(results)
        assert "3" in text
        assert "ramsey" in text
        assert "Pipeline Results" in text

    def test_empty_results(self):
        text = _format_results({})
        assert "Pipeline Results" in text


class TestFormatDatasetInfo:
    def test_basic(self):
        ds = WIMPDataset(
            protocol="echo",
            tau_array=np.linspace(0, 5e-6, 100),
            signal=np.ones((3, 100)),
            metadata={"temp": 295},
        )
        text = _format_dataset_info(ds)
        assert "echo" in text
        assert "100" in text
        assert "temp" in text
