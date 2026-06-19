````markdown
# Stable Walsh--Haar FCGR Descriptor for Alignment-Free Genomic Sequence Analysis

This repository contains the implementation, configuration files, automated tests, synthetic verification examples, machine-readable verification reports, generated LaTeX tables, and accession-fixed benchmark scripts for the manuscript:

**A Stable Walsh--Haar FCGR Descriptor for Alignment-Free Genomic Sequence Analysis: Edit-Distance Bounds, Toric Smoothing, Fixed Local Spectral Features, an Accession-Fixed Benchmark, and Reproducible Robustness Diagnostics**

The code implements a deterministic and interpretable alignment-free genomic descriptor based on frequency chaos-game representations (FCGRs), Walsh--Hadamard transforms, Haar transforms, torically smoothed Walsh coefficients, regularized spectral entropy, fixed local polar Walsh features, and reverse-complement averaging.

The repository is designed to support full computational reproducibility of the manuscript.

---

## Repository

GitHub repository:

```text
https://github.com/sergoep/Walsh-haar-fcgr-genomics
````

Archived release:

```text
https://doi.org/10.5281/zenodo.20634584
```

---

## Code availability statement

The accompanying Python implementation, configuration files, automated tests, synthetic verification examples, machine-readable verification reports, and the illustrative real-sequence verification output are publicly available in the GitHub repository at:

```text
https://github.com/sergoep/Walsh-haar-fcgr-genomics
```

and are archived in Zenodo at:

```text
https://doi.org/10.5281/zenodo.20634584
```

---

## Main features

The repository provides:

* normalized FCGR construction for DNA sequences;
* raw lexicographic k-mer baseline;
* flattened FCGR baseline;
* D2-style k-mer composition baseline;
* ML-DSP-style FFT proxy baseline;
* two-dimensional Walsh--Hadamard spectral block energies;
* two-dimensional Haar spectral block energies;
* regularized spectral entropy with dimension-normalized regularization;
* true torically smoothed Walsh coefficients computed by exact cell averages;
* fixed local polar Walsh features computed by deterministic quadrature;
* optional reverse-complement averaging;
* accession-fixed public viral benchmark;
* group-aware and stratified group-aware cross-validation;
* train-fold-only feature standardization;
* bootstrap confidence intervals;
* runtime measurements after warmup;
* synthetic edit-robustness diagnostics;
* sequencing-error robustness diagnostics;
* finite Riesz smoothing diagnostics;
* polar quadrature convergence diagnostics;
* sensitivity analysis for scale, smoothing and entropy parameters;
* automatically generated LaTeX tables for the manuscript;
* automated implementation self-tests.

---

## Repository structure

The expected repository structure is:

```text
Walsh-haar-fcgr-genomics/
├── README.md
├── requirements.txt
├── run_accession_benchmark_v8.py
├── run_submission.sh
├── submission_check.sh
├── CHECKLIST.md
├── LICENSE
├── manuscript/
│   └── manuscript_v8.tex
├── generated_tables/
│   ├── manifest_accessions_table.tex
│   ├── benchmark_results_table.tex
│   ├── runtime_results_table.tex
│   ├── edit_robustness_table.tex
│   ├── sequencing_robustness_table.tex
│   ├── finite_riesz_table.tex
│   ├── polar_convergence_table.tex
│   ├── sensitivity_k_table.tex
│   ├── sensitivity_c_alpha_table.tex
│   ├── method_comparison_table.tex
│   └── block_importance_table.tex
├── outputs/
│   ├── benchmark_manifest.tsv
│   ├── benchmark_results.tsv
│   ├── runtime_results.tsv
│   ├── fold_predictions.tsv
│   ├── run_config.json
│   ├── edit_robustness.tsv
│   ├── sequencing_robustness.tsv
│   ├── finite_riesz.tsv
│   ├── polar_convergence.tsv
│   ├── sensitivity_k.tsv
│   ├── sensitivity_c_alpha.tsv
│   ├── method_comparison.tsv
│   └── block_importance.tsv
└── figures/
    ├── stability_curves.pdf
    └── smoothing_tradeoff.pdf
```

Some output files are generated automatically after running the scripts. The `cache/genbank/` directory is used locally for downloaded GenBank records and is not required to be committed to GitHub.

---

## Installation

Python 3.10 or newer is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

The main Python dependencies are:

```text
biopython
numpy
scipy
scikit-learn
matplotlib
```

The manuscript compilation additionally requires a standard LaTeX distribution with packages such as:

```text
amsmath
amssymb
amsthm
mathtools
booktabs
tabularx
algorithm
algpseudocode
tikz
pgfplots
hyperref
```

---

## Quick start

Run implementation self-tests:

```bash
python run_accession_benchmark_v8.py --self-test
```

Run synthetic diagnostics without downloading GenBank records:

```bash
python run_accession_benchmark_v8.py \
  --self-test \
  --robustness \
  --sequencing-robustness \
  --finite-riesz \
  --quadrature-test \
  --sensitivity-k \
  --sensitivity-c-alpha \
  --method-comparison \
  --block-importance
```

Run the full accession-fixed benchmark:

```bash
export NCBI_EMAIL="your.email@example.com"
python run_accession_benchmark_v8.py \
  --download \
  --self-test \
  --run-benchmark \
  --runtime \
  --robustness \
  --sequencing-robustness \
  --finite-riesz \
  --quadrature-test \
  --sensitivity-k \
  --sensitivity-c-alpha \
  --method-comparison \
  --block-importance
```

Alternatively, run the full submission workflow:

```bash
export NCBI_EMAIL="your.email@example.com"
bash run_submission.sh
```

After the first download, the benchmark can be rerun offline using cached GenBank files:

```bash
python run_accession_benchmark_v8.py \
  --offline \
  --self-test \
  --run-benchmark \
  --runtime
```

---

## NCBI email requirement

The accession-fixed benchmark downloads public GenBank records through NCBI Entrez. NCBI requires users to provide an email address.

Before running the download step, set:

```bash
export NCBI_EMAIL="your.email@example.com"
```

On Windows PowerShell:

```powershell
$env:NCBI_EMAIL="your.email@example.com"
```

---

## Accession-fixed benchmark

The public benchmark uses a fixed list of viral accessions from four families:

* Coronaviridae;
* Flaviviridae;
* Herpesviridae;
* Adenoviridae.

The exact accession list is stored in the code and exported to:

```text
generated_tables/manifest_accessions_table.tex
outputs/benchmark_manifest.tsv
```

The manifest includes:

* requested accession;
* actual accession returned by GenBank;
* family label;
* leakage-control taxonomic group;
* organism name;
* full taxonomy string;
* sequence length;
* SHA-256 sequence hash;
* masked-window fractions for retained scales.

---

## Descriptor variants evaluated

The script evaluates the following descriptors:

1. Raw lexicographic k-mer profile;
2. Flattened FCGR;
3. D2-style k-mer composition;
4. ML-DSP-style FFT proxy;
5. Walsh-only spectral descriptor;
6. Haar-only spectral descriptor;
7. Walsh--Haar descriptor;
8. Walsh--Haar plus torically smoothed coefficients;
9. Full descriptor with polar Walsh features;
10. Full descriptor with reverse-complement averaging.

The ML-DSP-style baseline is a reproducible FFT-based proxy implemented in this repository. It is not the official ML-DSP software pipeline.

---

## Cross-validation protocol

The benchmark uses group-aware cross-validation to reduce leakage across closely related viral records.

When available, the script uses:

```text
StratifiedGroupKFold
```

Otherwise, it falls back to:

```text
GroupKFold
```

Feature standardization is performed inside the training fold only through a scikit-learn pipeline:

```text
StandardScaler + balanced LogisticRegression
```

This avoids information leakage from test folds into the training preprocessing.

---

## Generated outputs

After running the full pipeline, the script writes machine-readable files to:

```text
outputs/
```

Important files include:

```text
outputs/benchmark_manifest.tsv
outputs/benchmark_results.tsv
outputs/runtime_results.tsv
outputs/fold_predictions.tsv
outputs/run_config.json
outputs/edit_robustness.tsv
outputs/sequencing_robustness.tsv
outputs/finite_riesz.tsv
outputs/polar_convergence.tsv
outputs/sensitivity_k.tsv
outputs/sensitivity_c_alpha.tsv
outputs/method_comparison.tsv
outputs/block_importance.tsv
```

The script also writes manuscript-ready LaTeX tables to:

```text
generated_tables/
```

Important generated tables include:

```text
generated_tables/manifest_accessions_table.tex
generated_tables/benchmark_results_table.tex
generated_tables/runtime_results_table.tex
generated_tables/edit_robustness_table.tex
generated_tables/sequencing_robustness_table.tex
generated_tables/finite_riesz_table.tex
generated_tables/polar_convergence_table.tex
generated_tables/sensitivity_k_table.tex
generated_tables/sensitivity_c_alpha_table.tex
generated_tables/method_comparison_table.tex
generated_tables/block_importance_table.tex
```

Generated figures are written to:

```text
figures/
```

At minimum:

```text
figures/stability_curves.pdf
figures/smoothing_tradeoff.pdf
```

---

## Manuscript compilation

The main manuscript source is:

```text
manuscript/manuscript_v8.tex
```

Before compiling the manuscript, run the benchmark/diagnostic script so that all required generated tables exist.

Then compile with:

```bash
pdflatex manuscript/manuscript_v8.tex
pdflatex manuscript/manuscript_v8.tex
```

If the manuscript is configured with strict generated-table checking, compilation will fail if required generated tables are missing. This is intentional and prevents accidental submission with placeholder tables.

---

## Full submission check

Run:

```bash
bash submission_check.sh
```

This script checks that:

* the Python script compiles;
* implementation self-tests pass;
* required generated tables exist;
* required output files exist;
* required figures exist;
* the manuscript source is present.

---

## Mathematical self-tests

The implementation includes self-tests for the main mathematical components:

* FCGR normalization;
* Walsh--Hadamard orthogonality;
* Haar orthogonality;
* Frobenius energy preservation;
* reverse-complement descriptor consistency;
* smoothed Walsh DC cell weights;
* smoothed coefficient finite values;
* polar feature finite values;
* descriptor dimension consistency.

Run:

```bash
python run_accession_benchmark_v8.py --self-test
```

---

## Reproducibility notes

The repository is designed so that the tables in the manuscript are not manually edited.

The intended workflow is:

1. Run the Python script.
2. Generate `.tsv`, `.json`, `.tex`, and `.pdf` outputs.
3. Compile the manuscript.
4. Archive the exact release on Zenodo.

This ensures that benchmark tables and machine-readable reports correspond to the same code version.

---

## Recommended final workflow before journal submission

```bash
git clone https://github.com/sergoep/Walsh-haar-fcgr-genomics.git
cd Walsh-haar-fcgr-genomics

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export NCBI_EMAIL="your.email@example.com"

bash run_submission.sh
bash submission_check.sh
```

After successful completion, verify that the following files exist:

```text
outputs/benchmark_results.tsv
outputs/benchmark_manifest.tsv
outputs/runtime_results.tsv
generated_tables/benchmark_results_table.tex
generated_tables/runtime_results_table.tex
figures/stability_curves.pdf
figures/smoothing_tradeoff.pdf
```

---

## Citation

If you use this repository, please cite the archived Zenodo release:

```text
Sergo A. Episkoposian.
Stable Walsh--Haar FCGR Descriptor for Alignment-Free Genomic Sequence Analysis.
Zenodo.
https://doi.org/10.5281/zenodo.20634584
```

A BibTeX entry may be added after the final Zenodo metadata are confirmed.

---

## License

This repository is released for reproducible academic research. Please see the `LICENSE` file for details.

If no license has yet been added, the recommended license for code reuse is:

```text
MIT License
```

For manuscript text, figures and tables, use the journal-compatible license required by the publisher.

---

## Contact

Sergo A. Episkoposian
National Polytechnic University of Armenia
Yerevan State University
Email: [sergoep@ysu.am](mailto:sergoep@ysu.am)

GitHub:

```text
https://github.com/sergoep
```

---

## Version

Current submission package:

```text
Version 8
```

Recommended GitHub release tag:

```text
v8.0
```

Recommended release title:

```text
Version 8 submission package: stable Walsh--Haar FCGR descriptor
```

```
```
