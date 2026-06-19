#!/usr/bin/env bash
set -euo pipefail

echo "[check] Python syntax"
python -m py_compile run_accession_benchmark_v8.py

echo "[check] Running lightweight diagnostics"
python run_accession_benchmark_v8.py \
  --self-test \
  --finite-riesz \
  --robustness \
  --sequencing-robustness \
  --quadrature-test \
  --sensitivity-k \
  --sensitivity-c-alpha \
  --figures

echo "[check] Required core files"
required_files=(
  "README.md"
  "requirements.txt"
  "run_accession_benchmark_v8.py"
  "run_submission.sh"
  "submission_check.sh"
  "CHECKLIST.md"
)

for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[missing] $f" >&2
    exit 1
  fi
done

echo "[check] Required diagnostic generated tables"
required_diagnostic_tables=(
  "generated_tables/finite_riesz_table.tex"
  "generated_tables/edit_robustness_table.tex"
  "generated_tables/sequencing_robustness_table.tex"
  "generated_tables/polar_convergence_table.tex"
  "generated_tables/sensitivity_k_table.tex"
  "generated_tables/sensitivity_c_alpha_table.tex"
)

for f in "${required_diagnostic_tables[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "[missing or empty] $f" >&2
    exit 1
  fi
done

echo "[check] Required diagnostic machine-readable reports"
required_diagnostic_outputs=(
  "outputs/edit_robustness.tsv"
  "outputs/sequencing_robustness.tsv"
  "outputs/finite_riesz.tsv"
  "outputs/polar_convergence.tsv"
  "outputs/sensitivity_k.tsv"
  "outputs/sensitivity_c_alpha.tsv"
  "outputs/run_config.json"
)

for f in "${required_diagnostic_outputs[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "[missing or empty] $f" >&2
    exit 1
  fi
done

echo "[check] Required figures"
required_figures=(
  "figures/stability_curves.pdf"
  "figures/smoothing_tradeoff.pdf"
)

for f in "${required_figures[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "[missing or empty] $f" >&2
    exit 1
  fi
done

echo "[check] Optional accession-fixed benchmark outputs"
optional_benchmark_files=(
  "generated_tables/manifest_accessions_table.tex"
  "generated_tables/benchmark_results_table.tex"
  "generated_tables/runtime_results_table.tex"
  "outputs/benchmark_manifest.tsv"
  "outputs/benchmark_results.tsv"
  "outputs/runtime_results.tsv"
  "outputs/fold_predictions.tsv"
)

missing_optional=0
for f in "${optional_benchmark_files[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "[optional missing] $f"
    missing_optional=1
  fi
done

if [[ "$missing_optional" -eq 1 ]]; then
  echo "[note] Accession-fixed benchmark files are missing. Run:"
  echo "       export NCBI_EMAIL=\"your.email@example.com\""
  echo "       bash run_submission.sh"
else
  echo "[check] Accession-fixed benchmark outputs are present."
fi

echo "[check] OK: lightweight reproducibility package checks passed."
