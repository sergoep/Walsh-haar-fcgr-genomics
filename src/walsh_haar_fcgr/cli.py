"""Command-line interface for verification and synthetic demonstrations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .descriptors import DescriptorConfig, theory_aligned_descriptor
from .fcgr import fcgr_matrix, verify_rc_equivariance
from .manifest import write_run_manifest
from .polar import PolarDisk
from .robustness import delete, insert, substitute
from .smoothing import SmoothedWalsh, numerical_l2_error_sq
from .transforms import fwht2, haar2


def _default_sequence() -> str:
    return "ACGTGCAATGCCGTTAACCGGTTACGTACGATCGATCGTACGATCGTTAGCA" * 3


def run_demo(output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence = _default_sequence()
    config = DescriptorConfig()
    baseline = theory_aligned_descriptor(sequence, config)
    variants = {
        "substitution_3": substitute(sequence, 3, seed=seed),
        "insertion_3": insert(sequence, 3, seed=seed),
        "deletion_3": delete(sequence, 3, seed=seed),
    }
    distances = {
        name: float(np.linalg.norm(theory_aligned_descriptor(value, config) - baseline))
        for name, value in variants.items()
    }
    k = 4
    matrix = fcgr_matrix(sequence, k).matrix
    smooth = SmoothedWalsh(index=5, epsilon=0.01)
    report = {
        "descriptor_length": int(baseline.size),
        "descriptor_distances": distances,
        "fcgr_sum": float(np.sum(matrix)),
        "fwht_energy": float(np.linalg.norm(fwht2(matrix))),
        "haar_energy": float(np.linalg.norm(haar2(matrix))),
        "fcgr_energy": float(np.linalg.norm(matrix)),
        "reverse_complement_equivariance": verify_rc_equivariance(sequence, k),
        "smoothing_l2_error_sq_theoretical": smooth.theoretical_l2_error_sq,
        "smoothing_l2_error_sq_numerical": numerical_l2_error_sq(smooth, samples=100_000),
    }
    (output_dir / "demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_run_manifest(output_dir / "run_manifest.json", seed=seed, extra={"command": "demo"})
    print(json.dumps(report, indent=2))


def run_verify(output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence = _default_sequence()
    matrix = fcgr_matrix(sequence, 4).matrix
    energy = float(np.linalg.norm(matrix))
    checks = {
        "fcgr_normalization": bool(np.isclose(np.sum(matrix), 1.0)),
        "fwht_energy_preservation": bool(np.isclose(np.linalg.norm(fwht2(matrix)), energy)),
        "haar_energy_preservation": bool(np.isclose(np.linalg.norm(haar2(matrix)), energy)),
        "reverse_complement_equivariance": verify_rc_equivariance(sequence, 4),
    }
    (output_dir / "verification_report.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    write_run_manifest(output_dir / "run_manifest.json", seed=seed, extra={"command": "verify"})
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "verify"))
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if args.command == "demo":
        run_demo(output_dir, args.seed)
    else:
        run_verify(output_dir, args.seed)


if __name__ == "__main__":
    main()
