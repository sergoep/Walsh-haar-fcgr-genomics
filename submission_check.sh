#!/usr/bin/env bash
set -euo pipefail
python -m py_compile run_accession_benchmark_v8.py
python run_accession_benchmark_v8.py --self-test --finite-riesz --robustness --sequencing-robustness --quadrature-test --sensitivity-k --sensitivity-c-alpha
for f in \
  generated_tables/finite_riesz_table.tex \
  generated_tables/edit_robustness_table.tex \
  generated_tables/sequencing_robustness_table.tex \
  generated_tables/polar_convergence_table.tex \
  generated_tables/sensitivity_k_table.tex \
  generated_tables/sensitivity_c_alpha_table.tex; do
  test -s "$f" || { echo "Missing $f" >&2; exit 1; }
done
echo "Lightweight checks passed. Run run_submission.sh to download accessions and build the final PDF."
