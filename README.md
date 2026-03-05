# WIMP - Worm IMage Processor

Time-domain NV-centre magnetometry signal processing for *C. elegans*
neural imaging.

WIMP provides an end-to-end pipeline for extracting neural activity from
nitrogen-vacancy (NV) diamond magnetometry experiments on *C. elegans*.
It supports multiple pulse protocols, deformable body-atlas registration,
source localisation, and real-time streaming analysis.

**iGEM 2026 -- Team UIUC**

---

## Features

- **Multi-protocol support** -- Ramsey, Hahn echo, CPMG, XY-8, and
  T1 relaxometry pulse sequences with filter-function computation.
- **Curve fitting** -- Non-linear least-squares extraction of B-field,
  T2\*, T2, and T1 from time-domain signals.
- **Source localisation** -- Minimum-norm estimation (MNE) and LCMV
  beamforming to reconstruct neural current sources from nanodiamond
  field measurements.
- **Deformable atlas registration** -- Per-frame spline fitting through
  tracked nanodiamond positions, warping a canonical *C. elegans*
  neuroanatomical atlas onto each video frame.
- **Crosstalk analysis** -- Resolution matrix computation and
  per-neuron crosstalk metrics to quantify inter-source signal leakage.
- **Real-time processing** -- Threaded streaming processor with
  configurable buffering, incremental fitting, and callback hooks.
- **Sensitivity analysis** -- Analytical sensitivity estimates
  (T/sqrt(Hz)) for each protocol given NV parameters.
- **Data I/O** -- HDF5 and NumPy serialisation via a unified
  `WIMPDataset` container, plus JSON/CSV export.
- **Visualisation** -- Publication-quality matplotlib figures for
  Ramsey fringes, decay curves, field maps, source maps, resolution
  matrices, and neuron trajectories.
- **CLI** -- `wimp run`, `wimp synthetic`, `wimp sensitivity`,
  `wimp crosstalk`, and `wimp info` subcommands.
- **Synthetic data** -- Configurable synthetic experiment generator
  including deformable worm locomotion models.

---

## Installation

Requires **Python 3.10+**.

```bash
# From the project root
pip install -e .
```

With optional dependencies:

```bash
# Nanodiamond tracking (trackpy)
pip install -e ".[tracking]"

# Interactive visualisation (napari, Jupyter)
pip install -e ".[interactive]"

# Everything
pip install -e ".[all]"
```

Core dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`, `h5py`.

---

## Quick start

### Python API

```python
import numpy as np
from wimp.synthetic import generate_ramsey_data
from wimp.relaxation import fit_ramsey
from wimp import viz

# Generate a synthetic Ramsey fringe
tau = np.linspace(0, 5e-6, 200)
data = generate_ramsey_data(tau, b_field=50e-6, t2star=1e-6, snr=40, seed=42)

# Fit to extract the magnetic field
fit = fit_ramsey(data["tau"], data["signal"])
print(f"B-field: {fit['b_field']*1e6:.1f} uT")
print(f"T2*:     {fit['t2star']*1e6:.2f} us")

# Plot
fig = viz.plot_ramsey_fringe(data["tau"], data["signal"], fit)
fig.savefig("ramsey.png", dpi=150)
```

### Full pipeline

```python
from wimp.synthetic import generate_full_experiment
from wimp.io import WIMPDataset
from wimp.pipeline import PipelineConfig, run_pipeline

exp = generate_full_experiment(n_nds=8, n_neurons=4, protocol="ramsey", seed=99)

ds = WIMPDataset(
    protocol="ramsey",
    tau_array=exp["tau_array"],
    signal=exp["signal"],
    nd_positions=exp["nd_positions"],
)

config = PipelineConfig(
    protocol="ramsey",
    source_localization=True,
    lambda_reg=0.1,
    registration_params={"neuron_positions": exp["neuron_positions"].tolist()},
    compute_crosstalk=True,
)

results = run_pipeline(config, dataset=ds)
print(f"Source estimate shape: {results['source_estimate'].shape}")
```

### CLI

```bash
# Generate synthetic data and save to HDF5
wimp synthetic -p ramsey --n-nds 10 --n-neurons 5 -o experiment.h5

# Run the processing pipeline
wimp run experiment.h5 -p ramsey --plot

# Compare protocol sensitivities
wimp sensitivity --t2star 1e-6 --t2 100e-6 --t1 5e-3

# Analyse crosstalk for a given geometry
wimp crosstalk --n-nds 10 --n-neurons 5 --lambda-reg 0.1 --plot

# Inspect a dataset
wimp info experiment.h5
```

---

## Project structure

```
src/wimp/
  __init__.py        Public API exports
  constants.py       Physical constants (GAMMA_NV, D0, MU0, ...)
  pulses.py          Pulse sequence definitions & filter functions
  relaxation.py      T1/T2/T2* fitting and B-field extraction
  sensitivity.py     Analytical sensitivity estimates
  source.py          Lead-field matrix, MNE inverse, resolution matrix
  registration.py    Centerline fitting, atlas registration, deformable reg.
  synthetic.py       Synthetic data generation (static & deformable)
  analysis.py        Noise spectroscopy, SNR, event detection
  calibration.py     NV characterisation & temperature correction
  localization.py    Nanodiamond detection via trackpy
  realtime.py        Threaded real-time streaming processor
  pipeline.py        End-to-end pipeline orchestration
  io.py              HDF5 / NumPy / CSV I/O
  viz.py             Matplotlib visualisation functions
  cli.py             Command-line interface

tests/               14 test modules + conftest.py
demo.ipynb           Interactive demo notebook
```

---

## Protocols

| Protocol | Measures | Typical sensitivity |
|----------|----------|---------------------|
| **Ramsey** | DC magnetic field | ~1 nT/sqrt(Hz) at T2\*=1 us |
| **Hahn echo** | AC magnetic field | ~100 pT/sqrt(Hz) at T2=100 us |
| **CPMG-N** | AC field (narrow band) | ~10 pT/sqrt(Hz) (N=64) |
| **XY-8** | AC field (robust) | Similar to CPMG |
| **T1 relaxometry** | Noise spectral density | Broadband, ~GHz range |

---

## Key physical constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GAMMA_NV` | 28.024 GHz/T | NV gyromagnetic ratio |
| `D0` | 2.870 GHz | Zero-field splitting |
| `MU0` | 4pi x 10^-7 T m/A | Vacuum permeability |
| `CANONICAL_BODY_LENGTH` | 1.0 mm | Adult *C. elegans* body length |

---

## Demo notebook

The included `demo.ipynb` walks through all major capabilities:

1. Synthetic Ramsey data generation & fitting
2. Echo & T1 fitting
3. Protocol sensitivity comparison
4. Full synthetic experiment & MNE source localisation
5. Neural event detection
6. Pipeline processing
7. Real-time streaming
8. Deformable atlas registration
9. Crosstalk analysis
10. Data I/O round-trips

---

## Testing

```bash
# Run the full test suite
python -m pytest tests/ -v

# Quick run
python -m pytest tests/ -x -q
```

301 tests, 2 skipped (optional trackpy dependency).

---

## License

Released into the public domain under the [Unlicense](https://unlicense.org).
See [LICENSE](LICENSE) for details.
