[![DOI](https://zenodo.org/badge/1265152551.svg)](https://doi.org/10.5281/zenodo.20634584)

# Walsh--Haar FCGR: theory-aligned genomic descriptor

This repository contains the reproducible Python implementation accompanying the article:

> **Robust Multiscale Dyadic Analysis of Genomic Sequences: FCGR Representations, Walsh--Haar Features, Toric Smoothing, and a Fixed Local Polar Dictionary**

The code is intended for verification of the mathematical constructions and for preparation of later benchmark studies. Synthetic demonstrations are consistency checks, not biological evidence.

## Included components

- normalized FCGR matrices with masking of ambiguous IUPAC windows;
- reverse-complement equivariance checks;
- orthonormal two-dimensional Walsh--Hadamard and Haar transforms;
- fixed spectral block energies;
- regularized spectral entropy;
- torically smoothed Walsh functions;
- exact cell-average coefficients for step FCGR densities;
- fixed polar Walsh dictionary with midpoint quadrature;
- synthetic edit operators;
- reproducibility manifest;
- automated tests for key identities and stability bounds.

## Installation

Open a terminal in this folder and create an isolated environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the package and test dependency:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the verification checks

```bash
walsh-haar-fcgr verify --output-dir output
```

## Run the synthetic demonstration

```bash
walsh-haar-fcgr demo --output-dir output
```

The command creates:

- `output/demo_report.json`;
- `output/run_manifest.json`.

## Run the automated tests

```bash
pytest -q
```

## Run the beginner example

```bash
python examples/run_synthetic_demo.py
```

## Reproducibility note

The polar coefficients are analytical integrals in the article. The implementation approximates them using a fixed midpoint quadrature that is uniform in the transformed radial coordinate `u=r^2` and uniform in the angular coordinate. Quadrature error must be studied separately by convergence analysis when benchmark results are reported.

## License

MIT License.
