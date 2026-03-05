"""Interactive Panel dashboard for WIMP.

Provides :class:`WIMPDashboard`, a self-contained Panel application with
four tabs: Pipeline Runner, Realtime Monitor, Connectome Explorer, and
Data Browser.

Requires the optional ``panel`` dependency.  Install with
``pip install 'wimp[dashboard]'``.

Usage::

    from wimp.dashboard import WIMPDashboard

    dashboard = WIMPDashboard()
    dashboard.show()           # launch in browser
    dashboard.notebook()       # inline in Jupyter
    dashboard.servable()       # for ``panel serve``
"""

from __future__ import annotations

from typing import Any


def _import_panel():
    """Lazy import with helpful error message."""
    try:
        import panel as pn
        return pn
    except ImportError:
        raise ImportError(
            "The WIMP dashboard requires panel. "
            "Install with: pip install 'wimp[dashboard]'"
        ) from None


# ---------------------------------------------------------------------------
# Helper formatters
# ---------------------------------------------------------------------------


def _format_results(results: dict[str, Any]) -> str:
    """Format pipeline results as a readable text summary."""
    import numpy as np

    lines = ["## Pipeline Results\n"]

    n_nds = results.get("n_nds", "?")
    lines.append(f"- **NDs processed:** {n_nds}")

    fits = results.get("fits", [])
    lines.append(f"- **Fits returned:** {len(fits)}")

    ft = results.get("field_timeseries")
    if ft is not None:
        ft = np.asarray(ft)
        lines.append(
            f"- **Field range:** {ft.min():.3e} – {ft.max():.3e} T"
        )

    se = results.get("source_estimate")
    if se is not None:
        se = np.asarray(se)
        lines.append(f"- **Source estimate shape:** {se.shape}")

    cfg = results.get("config", {})
    if cfg:
        lines.append(f"- **Protocol:** {cfg.get('protocol', '?')}")
        lines.append(f"- **Lambda reg:** {cfg.get('lambda_reg', '?')}")

    return "\n".join(lines)


def _format_dataset_info(ds: Any) -> str:
    """Format a WIMPDataset summary."""
    import numpy as np

    lines = ["## Dataset Info\n"]
    lines.append(f"- **Protocol:** {ds.protocol}")
    lines.append(f"- **Tau array:** {len(ds.tau_array)} points")

    sig = np.asarray(ds.signal)
    lines.append(f"- **Signal shape:** {sig.shape}")

    if ds.field_timeseries is not None:
        lines.append(f"- **Field timeseries shape:** {np.asarray(ds.field_timeseries).shape}")
    if ds.nd_positions is not None:
        lines.append(f"- **ND positions shape:** {np.asarray(ds.nd_positions).shape}")
    if ds.metadata:
        for k, v in ds.metadata.items():
            lines.append(f"- **{k}:** {v}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------


class WIMPDashboard:
    """Interactive Panel dashboard for WIMP.

    Parameters
    ----------
    config : PipelineConfig | None
        Initial pipeline configuration.  If *None*, a default is used.
    """

    def __init__(self, config: Any | None = None) -> None:
        pn = _import_panel()
        pn.extension(sizing_mode="stretch_width")

        if config is None:
            from wimp.pipeline import PipelineConfig
            config = PipelineConfig()
        self.config = config

        self._results: dict[str, Any] | None = None
        self._dataset: Any | None = None
        self._realtime_proc: Any | None = None

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_pipeline_tab(self):
        pn = _import_panel()

        protocol = pn.widgets.Select(
            name="Protocol",
            options=["ramsey", "echo", "t1", "dd"],
            value=self.config.protocol,
        )
        lambda_reg = pn.widgets.FloatInput(
            name="Lambda (reg.)", value=self.config.lambda_reg,
            start=0.0, step=0.01,
        )
        source_loc = pn.widgets.Checkbox(
            name="Source localisation", value=self.config.source_localization,
        )
        denoise = pn.widgets.Select(
            name="Denoise method",
            options=["None", "wavelet", "savgol", "median"],
            value=self.config.denoise_method or "None",
        )
        run_btn = pn.widgets.Button(name="Run Pipeline", button_type="primary")
        status = pn.pane.Markdown("*Ready.*")
        results_pane = pn.pane.Markdown("")
        plot_pane = pn.pane.Matplotlib(None, tight=True, dpi=100)

        def _run(event):
            from dataclasses import replace
            from wimp.pipeline import run_pipeline

            status.object = "*Running pipeline…*"
            dm = denoise.value if denoise.value != "None" else None
            cfg = replace(
                self.config,
                protocol=protocol.value,
                lambda_reg=lambda_reg.value,
                source_localization=source_loc.value,
                denoise_method=dm,
            )
            try:
                res = run_pipeline(cfg, dataset=self._dataset)
                self._results = res
                results_pane.object = _format_results(res)
                status.object = "**Done.**"

                # Plot field timeseries if available
                import numpy as np
                ft = res.get("field_timeseries")
                if ft is not None:
                    from wimp.viz import plot_field_timeseries
                    fig = plot_field_timeseries(np.asarray(ft))
                    if fig is not None:
                        plot_pane.object = fig
            except Exception as exc:
                status.object = f"**Error:** {exc}"

        run_btn.on_click(_run)

        sidebar = pn.Column(protocol, lambda_reg, source_loc, denoise, run_btn)
        main = pn.Column(status, results_pane, plot_pane)
        return pn.Row(sidebar, main, name="Pipeline Runner")

    def _build_realtime_tab(self):
        pn = _import_panel()

        start_btn = pn.widgets.Button(name="Start", button_type="success")
        stop_btn = pn.widgets.Button(name="Stop", button_type="danger")
        kalman_chk = pn.widgets.Checkbox(name="Kalman filter", value=True)
        status = pn.pane.Markdown("*Stopped.*")
        plot_pane = pn.pane.Matplotlib(None, tight=True, dpi=100)
        frame_counter = pn.pane.Markdown("Frames: 0")

        _callback_id = [None]

        def _update():
            if self._realtime_proc is None:
                return
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            latest = getattr(self._realtime_proc, "latest_result", None)
            if latest is None:
                return
            fv = latest.field_values
            if fv is not None:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(np.asarray(fv) * 1e6, "o-")
                ax.set_ylabel("B field (uT)")
                ax.set_xlabel("ND index")
                ax.set_title(f"Frame {latest.frame_count}")
                fig.tight_layout()
                plot_pane.object = fig
                plt.close(fig)
            frame_counter.object = f"Frames: {latest.frame_count}"

        def _start(event):
            from wimp.realtime import RealtimeConfig, RealtimeProcessor

            rconfig = RealtimeConfig(use_kalman=kalman_chk.value)
            self._realtime_proc = RealtimeProcessor(rconfig)
            self._realtime_proc.start()
            status.object = "*Running…*"
            _callback_id[0] = pn.state.add_periodic_callback(_update, period=500)

        def _stop(event):
            if self._realtime_proc is not None:
                self._realtime_proc.stop()
                status.object = "*Stopped.*"
            if _callback_id[0] is not None:
                _callback_id[0].stop()
                _callback_id[0] = None

        start_btn.on_click(_start)
        stop_btn.on_click(_stop)

        controls = pn.Row(start_btn, stop_btn, kalman_chk)
        return pn.Column(controls, status, frame_counter, plot_pane, name="Realtime Monitor")

    def _build_connectome_tab(self):
        pn = _import_panel()

        circuit_sel = pn.widgets.Select(
            name="Circuit",
            options=["full_connectome", "tap_withdrawal", "thermotaxis", "chemotaxis", "nose_touch"],
            value="tap_withdrawal",
        )
        n_steps = pn.widgets.IntSlider(name="Propagation steps", start=1, end=20, value=5)
        decay = pn.widgets.FloatSlider(name="Decay", start=0.0, end=1.0, value=0.5, step=0.05)
        propagate_btn = pn.widgets.Button(name="Propagate", button_type="primary")
        plot_pane = pn.pane.Matplotlib(None, tight=True, dpi=100)

        def _propagate(event):
            import numpy as np
            from wimp.connectome import load_connectome, known_circuit, connectome_propagation
            from wimp.viz import plot_connectome_graph

            if circuit_sel.value == "full_connectome":
                conn = load_connectome()
            else:
                conn = known_circuit(circuit_sel.value)

            names = conn.get("neuron_names", conn.get("neurons", []))
            n = len(names)

            # Use pipeline results activity if available
            a0 = np.zeros(n)
            if self._results is not None:
                se = self._results.get("source_estimate")
                if se is not None:
                    se = np.asarray(se).ravel()
                    a0[:min(len(se), n)] = se[:min(len(se), n)]
                else:
                    a0[0] = 1.0
            else:
                a0[0] = 1.0

            traj = connectome_propagation(
                a0, conn["adjacency"],
                n_steps=n_steps.value,
                decay=decay.value,
            )

            fig = plot_connectome_graph(
                conn, activity=traj, propagation_step=traj.shape[0] - 1,
            )
            if fig is not None:
                plot_pane.object = fig

        propagate_btn.on_click(_propagate)

        controls = pn.Column(circuit_sel, n_steps, decay, propagate_btn)
        return pn.Row(controls, plot_pane, name="Connectome Explorer")

    def _build_data_tab(self):
        pn = _import_panel()

        info_pane = pn.pane.Markdown("*No dataset loaded.*")
        plot_pane = pn.pane.Matplotlib(None, tight=True, dpi=100)
        load_btn = pn.widgets.Button(name="Load HDF5", button_type="primary")
        save_h5_btn = pn.widgets.Button(name="Save HDF5")
        save_zarr_btn = pn.widgets.Button(name="Save Zarr")

        file_input = pn.widgets.TextInput(
            name="File path", placeholder="path/to/dataset.h5",
        )

        def _load(event):
            import numpy as np
            from pathlib import Path

            path = Path(file_input.value)
            if not path.exists():
                info_pane.object = f"**File not found:** {path}"
                return

            suffix = path.suffix.lower()
            try:
                if suffix in (".h5", ".hdf5"):
                    from wimp.io import load_hdf5
                    ds = load_hdf5(path)
                elif suffix == ".zarr" or path.is_dir():
                    from wimp.io import load_zarr
                    ds = load_zarr(path)
                else:
                    info_pane.object = f"**Unsupported format:** {suffix}"
                    return

                self._dataset = ds
                info_pane.object = _format_dataset_info(ds)

                # Preview signal
                sig = np.asarray(ds.signal)
                tau = np.asarray(ds.tau_array)
                if sig.size > 0 and tau.size > 0:
                    from wimp.viz import plot_ramsey_fringe
                    fig = plot_ramsey_fringe(tau, sig[0] if sig.ndim > 1 else sig)
                    if fig is not None:
                        plot_pane.object = fig

            except Exception as exc:
                info_pane.object = f"**Error:** {exc}"

        def _save_h5(event):
            if self._dataset is None:
                return
            from wimp.io import save_hdf5
            from pathlib import Path
            out = Path(file_input.value).with_suffix(".h5")
            save_hdf5(self._dataset, out)
            info_pane.object += f"\n\n*Saved to {out}*"

        def _save_zarr(event):
            if self._dataset is None:
                return
            from wimp.io import save_zarr
            from pathlib import Path
            out = Path(file_input.value).with_suffix(".zarr")
            save_zarr(self._dataset, out)
            info_pane.object += f"\n\n*Saved to {out}*"

        load_btn.on_click(_load)
        save_h5_btn.on_click(_save_h5)
        save_zarr_btn.on_click(_save_zarr)

        controls = pn.Column(file_input, load_btn, pn.Row(save_h5_btn, save_zarr_btn))
        return pn.Row(controls, pn.Column(info_pane, plot_pane), name="Data Browser")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def servable(self):
        """Build and return a Panel template, ready for ``panel serve``.

        Returns
        -------
        pn.template.MaterialTemplate
        """
        pn = _import_panel()

        template = pn.template.MaterialTemplate(title="WIMP Dashboard")

        tabs = pn.Tabs(
            self._build_pipeline_tab(),
            self._build_realtime_tab(),
            self._build_connectome_tab(),
            self._build_data_tab(),
        )

        template.main.append(tabs)
        return template

    def show(self, **kwargs) -> None:
        """Launch the dashboard in the default browser.

        Parameters
        ----------
        **kwargs
            Forwarded to ``pn.serve``.
        """
        pn = _import_panel()
        tmpl = self.servable()
        tmpl.show(**kwargs)

    def notebook(self):
        """Display the dashboard inline in a Jupyter notebook.

        Returns
        -------
        Panel object suitable for display.
        """
        return self.servable()
