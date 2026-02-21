# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pytelpoint` is a Python library for analyzing telescope pointing performance at the MMT Observatory using Bayesian/probabilistic methods via PyMC. It fits standard TPOINT-compatible pointing models to az/el encoder data.

## Environment Setup

```bash
pip install -e ".[test]"
```

On macOS, `fitting.py` enforces `/usr/bin/clang++` to avoid conflicts between conda-forge's clang and Xcode command-line tools.

## Common Commands

```bash
# Run all tests
pytest --pyargs pytelpoint docs

# Run a single test file
pytest pytelpoint/tests/test_io.py

# Run a single test
pytest pytelpoint/tests/test_io.py::test_read_azel_datfile

# Check code style (max line length 132)
flake8 pytelpoint --count --max-line-length=132

# Format code
black pytelpoint

# Run via tox (uses conda for pymc)
tox -e py313-test

# Build docs
tox -e build_docs
```

## Architecture

All core modules use `SkyCoord` (AstropY) as the primary data carrier. The typical workflow is:

1. **IO** (`io.py`): Read pointing run data → returns `(coo_ref, coo_meas)` pair of `SkyCoord` objects in an `AltAz` frame tied to `MMT_LOCATION`.
2. **Stats** (`stats.py`): Compute sky RMS / population SD from coordinate pairs (TPOINT-equivalent diagnostics).
3. **Transform** (`transform.py`): Apply an 8-term az/el pointing model (`azel_model`) analytically to raw encoder coordinates.
4. **Fitting** (`fitting.py`): Fit the pointing model probabilistically using PyMC (`azel_fit`), returning an `arviz.InferenceData` object. `best_fit_pars` summarizes the posterior.
5. **Visualization** (`visualization.py`): `plot_posterior` and `plot_corner` for inspecting PyMC fit results.

### Pointing model terms (TPOINT-compatible)

All terms are in arcseconds. The 8 standard az/el terms are: `ia` (az index), `ie` (el index), `an` (N-S axis tilt), `aw` (E-W axis tilt), `ca` (collimation), `npae` (az/el non-perpendicularity), `tf` (tube flexure ∝ cos(el)), `tx` (tube flexure ∝ cot(el)).

### Key design note

`fitting.py` uses `pm.math` (Theano/pytensor ops) for the model functions, while `transform.py` uses `numpy` — they implement the same math but for different contexts (PyMC model graph vs. direct application).

### Test data

`pytelpoint/test_data/` contains three real pointing run files used in tests. `k_and_e.dat` is the processed (4-column) format; `point_20210821.dat` and `point_20250326.dat` are raw mount code output files.

## Testing Notes

- Tests live in `pytelpoint/tests/`. Doctests in `.rst` files under `docs/` are also run.
- `test_mc_fitting` runs a real PyMC MCMC chain (with reduced `nsamp=200, ntune=200`) — it is slow.
- `pytest.ini` options (in `pyproject.toml`) enable `--doctest-rst` and treat most warnings as errors.
