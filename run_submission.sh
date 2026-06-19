#!/usr/bin/env bash
set -euo pipefail

MODE="--download"
if [[ "${1:-}" == "--offline" ]]; then
  MODE="--offline"
fi

if [[ "$MODE" == "--download" && -z "${NCBI_EMAIL:-}" ]]; then
  echo "ERROR: Set NCBI_EMAIL before downloading from NCBI." >&2
  exit 1
fi

python run_accession_benchmark_v8.py \
  $MODE \
  --self-test \
  --all-diagnostics \
  --quadrature-test \
  --run-benchmark \
  --runtime \
  --figures

pdflatex -interaction=nonstopmode manuscript_v8.tex
pdflatex -interaction=nonstopmode manuscript_v8.tex

echo "Submission PDF built: manuscript_v8.pdf"
