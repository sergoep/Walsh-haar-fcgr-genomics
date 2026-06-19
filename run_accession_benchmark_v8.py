#!/usr/bin/env python3
"""
Accession-fixed benchmark for the stable Walsh--Haar FCGR descriptor.

This script implements the descriptor used in the manuscript:

    - raw lexicographic k-mer baseline;
    - flattened FCGR baseline;
    - Walsh--Hadamard spectral block energies;
    - Haar spectral block energies;
    - regularized spectral entropies with delta_k = alpha * 4^{-k};
    - true torically smoothed low-order Walsh coefficients computed by
      exact cell averages for the piecewise-linear smoothed Walsh probes;
    - fixed local polar Walsh coefficients computed by deterministic quadrature;
    - optional reverse-complement averaging;
    - accession-fixed public viral benchmark;
    - group-aware, preferably stratified, cross-validation;
    - train-fold-only standardization;
    - bootstrap confidence intervals;
    - runtime measurements with warmup;
    - reproducibility manifest with sequence hashes.

Requirements:
    biopython
    numpy
    scipy
    scikit-learn

Usage:
    export NCBI_EMAIL="your.email@example.com"
    python run_accession_benchmark_v8.py --download --self-test --all-diagnostics --run-benchmark --runtime --figures

Offline usage after download:
    python run_accession_benchmark_v8.py --offline --self-test --all-diagnostics --run-benchmark --runtime --figures

Outputs:
    cache/genbank/*.gb
    outputs/benchmark_manifest.tsv
    outputs/benchmark_results.tsv
    outputs/runtime_results.tsv
    outputs/fold_predictions.tsv
    generated_tables/benchmark_results_table.tex
    generated_tables/runtime_results_table.tex
    generated_tables/manifest_accessions_table.tex
    generated_tables/finite_riesz_table.tex
    generated_tables/edit_robustness_table.tex
    generated_tables/sequencing_robustness_table.tex
    generated_tables/polar_convergence_table.tex
    generated_tables/sensitivity_k_table.tex
    generated_tables/sensitivity_c_alpha_table.tex
    generated_tables/method_comparison_table.tex
    generated_tables/block_importance_table.tex
    figures/embedding_pca.pdf
    figures/stability_curves.pdf
    figures/smoothing_tradeoff.pdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

try:
    from Bio import Entrez, SeqIO
except ImportError as exc:
    raise SystemExit("Biopython is required. Install with: pip install biopython") from exc

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    from sklearn.model_selection import GroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = False


# ---------------------------------------------------------------------
# Fixed accession list.
# ---------------------------------------------------------------------

ACCESSIONS: Dict[str, List[str]] = {
    "Coronaviridae": [
        "NC_045512.2", "NC_004718.3", "NC_019843.3", "NC_005147.1", "NC_006577.2",
        "NC_005831.2", "NC_002645.1", "NC_001451.1", "NC_001846.1", "NC_003045.1",
        "NC_003436.1", "NC_002306.3", "NC_010646.1", "NC_009019.1", "NC_009020.1",
    ],
    "Flaviviridae": [
        "NC_001477.1", "NC_001474.2", "NC_001475.2", "NC_002640.1", "NC_012532.1",
        "NC_009942.1", "NC_001437.1", "NC_009029.2", "NC_001563.2", "NC_004102.1",
        "NC_003690.1", "NC_001461.1", "NC_002031.1", "NC_001672.1", "NC_003996.1",
    ],
    "Herpesviridae": [
        "NC_001806.2", "NC_001798.2", "NC_001348.1", "NC_007605.1", "NC_006273.2",
        "NC_001664.4", "NC_000898.1", "NC_001716.2", "NC_009333.1", "NC_004065.1",
        "NC_001491.2", "NC_006151.1", "NC_002229.3", "NC_001826.2", "NC_001847.1",
    ],
    "Adenoviridae": [
        "NC_001405.1", "NC_004001.2", "NC_011203.1", "NC_012959.1", "NC_014899.1",
        "NC_015225.1", "NC_016895.1", "NC_020074.1", "NC_026117.1", "NC_026131.1",
        "NC_028829.1", "NC_029050.1", "NC_030785.1", "NC_031946.1", "NC_038311.1",
    ],
}

DNA_ALPHABET = "ACGT"
CODE = {
    "A": (0, 0),
    "C": (0, 1),
    "G": (1, 1),
    "T": (1, 0),
}
COMPLEMENT = str.maketrans("ACGT", "TGCA")


# ---------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DescriptorConfig:
    scales: Tuple[int, ...] = (3, 4, 5)
    alpha_entropy: float = 0.2
    smooth_c: float = 0.025
    smooth_order: int = 4
    polar_order_m: int = 2
    polar_order_n: int = 2
    polar_radial_nodes: int = 16
    polar_angular_nodes: int = 32
    polar_disks: Tuple[Tuple[float, float, float], ...] = (
        (0.25, 0.25, 0.25),
        (0.50, 0.50, 0.25),
        (0.75, 0.75, 0.25),
    )


@dataclass
class RecordInfo:
    accession_requested: str
    accession_actual: str
    label: str
    group: str
    organism: str
    taxonomy: str
    length: int
    sha256: str
    masked_fraction_k3: float
    masked_fraction_k4: float
    masked_fraction_k5: float
    sequence: str


@dataclass
class MetricSummary:
    descriptor: str
    n_features: int
    accuracy_mean: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    balanced_accuracy_mean: float
    balanced_accuracy_ci_low: float
    balanced_accuracy_ci_high: float
    macro_f1_mean: float
    macro_f1_ci_low: float
    macro_f1_ci_high: float
    macro_auroc_mean: Optional[float]
    macro_auroc_ci_low: Optional[float]
    macro_auroc_ci_high: Optional[float]


# ---------------------------------------------------------------------
# Utility functions.
# ---------------------------------------------------------------------

def ensure_dirs() -> None:
    Path("cache/genbank").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("generated_tables").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def masked_fraction(seq: str, k: int) -> float:
    total = max(0, len(seq) - k + 1)
    if total == 0:
        return 1.0
    valid = 0
    for i in range(total):
        u = seq[i:i + k]
        if all(ch in CODE for ch in u):
            valid += 1
    return 1.0 - valid / total


def safe_float(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:.4f}"


# ---------------------------------------------------------------------
# NCBI fetching and manifest.
# ---------------------------------------------------------------------

def fetch_genbank_to_cache(accession: str, email: str, delay: float, retries: int = 3) -> Path:
    cache_path = Path("cache/genbank") / f"{accession}.gb"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    Entrez.email = email
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            print(f"[download] Fetching {accession} (attempt {attempt}/{retries})")
            with Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text") as handle:
                text = handle.read()
            if not text.strip():
                raise RuntimeError(f"Empty GenBank response for {accession}")
            cache_path.write_text(text, encoding="utf-8")
            time.sleep(delay)
            return cache_path
        except Exception as exc:
            last_error = exc
            wait = delay + 2 ** (attempt - 1)
            print(f"[warning] Failed to fetch {accession}: {exc}. Retrying in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Could not fetch {accession}") from last_error


def read_cached_record(accession: str) -> SeqIO.SeqRecord:
    cache_path = Path("cache/genbank") / f"{accession}.gb"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached GenBank file: {cache_path}")
    with cache_path.open("r", encoding="utf-8") as handle:
        return SeqIO.read(handle, "genbank")


def infer_taxonomic_group(organism: str, taxonomy: Sequence[str], family_label: str) -> str:
    """
    Infer a leakage-control group from GenBank taxonomy.

    The first taxonomic term below the family label is used by default. This is a
    conservative genus/subfamily-like grouping for viral records and is less
    leakage-prone than using the most specific species/strain-like term. The
    full taxonomy is still stored in the manifest so that users can audit or
    replace this grouping rule.
    """
    def clean(x: str) -> str:
        return x.strip().replace(" ", "_").replace("/", "_") or "unknown"

    if taxonomy:
        if family_label in taxonomy:
            idx = taxonomy.index(family_label)
            below_family = [t for t in taxonomy[idx + 1:] if t and t != family_label]
            if below_family:
                return clean(below_family[0])
        # Fallback: use the least specific informative term rather than the
        # most specific species/strain name when possible.
        informative = [t for t in taxonomy if t and t.lower() not in {"viruses", "riboviria", "duplodnaviria"}]
        if informative:
            return clean(informative[-1] if len(informative) == 1 else informative[-2])

    words = organism.split()
    if len(words) >= 2:
        return clean(" ".join(words[:2]))
    if organism:
        return clean(organism)
    return "unknown"


def infer_taxonomic_group_specific(organism: str, taxonomy: Sequence[str], family_label: str) -> str:
    """Return the most specific available group for manifest auditing only."""
    if taxonomy:
        if family_label in taxonomy:
            idx = taxonomy.index(family_label)
            below_family = taxonomy[idx + 1:]
            if below_family:
                return below_family[-1].replace(" ", "_")
        return taxonomy[-1].replace(" ", "_")
    return organism.replace(" ", "_") if organism else "unknown"

def load_records(email: str, download: bool, offline: bool, delay: float) -> List[RecordInfo]:
    ensure_dirs()
    records: List[RecordInfo] = []

    if download and offline:
        raise ValueError("Use either --download or --offline, not both.")

    if download:
        if not email:
            raise SystemExit("Set NCBI_EMAIL or pass --email before downloading.")
        for _, accs in ACCESSIONS.items():
            for acc in accs:
                fetch_genbank_to_cache(acc, email=email, delay=delay)

    for label, accs in ACCESSIONS.items():
        for acc in accs:
            rec = read_cached_record(acc)
            seq = str(rec.seq).upper().replace("U", "T")
            organism = rec.annotations.get("organism", "unknown")
            taxonomy = rec.annotations.get("taxonomy", [])
            actual_accession = rec.id
            group = infer_taxonomic_group(organism, taxonomy, family_label=label)

            info = RecordInfo(
                accession_requested=acc,
                accession_actual=actual_accession,
                label=label,
                group=group,
                organism=organism,
                taxonomy=";".join(taxonomy),
                length=len(seq),
                sha256=sha256_text(seq),
                masked_fraction_k3=masked_fraction(seq, 3),
                masked_fraction_k4=masked_fraction(seq, 4),
                masked_fraction_k5=masked_fraction(seq, 5),
                sequence=seq,
            )
            records.append(info)

    return records


def write_manifest(records: Sequence[RecordInfo]) -> None:
    path = Path("outputs/benchmark_manifest.tsv")
    fields = [
        "accession_requested",
        "accession_actual",
        "label",
        "group",
        "organism",
        "taxonomy",
        "length",
        "sha256",
        "masked_fraction_k3",
        "masked_fraction_k4",
        "masked_fraction_k5",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            row.pop("sequence")
            writer.writerow(row)


# ---------------------------------------------------------------------
# FCGR and raw baselines.
# ---------------------------------------------------------------------

def fcgr(seq: str, k: int) -> np.ndarray:
    n = 2 ** k
    mat = np.zeros((n, n), dtype=float)
    valid = 0

    if len(seq) < k:
        return mat

    for i in range(0, len(seq) - k + 1):
        u = seq[i:i + k]
        if any(ch not in CODE for ch in u):
            continue
        r = 0
        c = 0
        for ch in u:
            a, b = CODE[ch]
            r = (r << 1) | a
            c = (c << 1) | b
        mat[r, c] += 1.0
        valid += 1

    if valid > 0:
        mat /= valid
    return mat


def raw_kmer_lexicographic(seq: str, ks: Sequence[int]) -> np.ndarray:
    features: List[float] = []
    for k in ks:
        words = ["".join(p) for p in product(DNA_ALPHABET, repeat=k)]
        counts = {w: 0.0 for w in words}
        valid = 0

        if len(seq) >= k:
            for i in range(len(seq) - k + 1):
                u = seq[i:i + k]
                if any(ch not in CODE for ch in u):
                    continue
                counts[u] += 1.0
                valid += 1

        if valid > 0:
            features.extend(counts[w] / valid for w in words)
        else:
            features.extend(0.0 for _ in words)

    return np.asarray(features, dtype=float)


def flattened_fcgr(seq: str, ks: Sequence[int]) -> np.ndarray:
    return np.concatenate([fcgr(seq, k).ravel() for k in ks]).astype(float)




def d2_statistic_features(seq: str, ks: Sequence[int]) -> np.ndarray:
    """D2-style composition features: centered normalized k-mer counts.

    This is not a full implementation of every D2-family statistic; it is an
    explicit, reproducible D2 baseline built from k-mer count inner-product
    geometry. For classification, the centered vector is passed to the same
    fold-local standardization and classifier as all other descriptors.
    """
    feats = []
    for k in ks:
        v = raw_kmer_lexicographic(seq, (k,))
        if v.size:
            v = v - np.mean(v)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
        feats.append(v)
    return np.concatenate(feats) if feats else np.zeros(0, dtype=float)


def mldsp_proxy_features(seq: str, ks: Sequence[int]) -> np.ndarray:
    """Simple ML-DSP-like baseline using numerical mapping and FFT magnitudes.

    This proxy is included for reproducible comparison without external code. It
    should be labeled as an ML-DSP-style DSP baseline, not as the official ML-DSP
    software pipeline.
    """
    mapping = {"A": -1.5, "C": -0.5, "G": 0.5, "T": 1.5}
    x = np.asarray([mapping.get(ch, 0.0) for ch in seq], dtype=float)
    if x.size == 0:
        return np.zeros(32 + sum(4**k for k in ks), dtype=float)
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    if spec.size < 32:
        spec = np.pad(spec, (0, 32-spec.size))
    fft_feats = spec[:32]
    fft_feats = fft_feats / (np.linalg.norm(fft_feats) + 1e-15)
    return np.concatenate([fft_feats, d2_statistic_features(seq, ks)])

# ---------------------------------------------------------------------
# Orthogonal transforms.
# ---------------------------------------------------------------------

_HADAMARD_CACHE: Dict[int, np.ndarray] = {}
_HAAR_CACHE: Dict[int, np.ndarray] = {}


def hadamard_matrix(n: int) -> np.ndarray:
    if not is_power_of_two(n):
        raise ValueError("Hadamard size must be a positive power of two.")
    if n in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[n]

    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    H = H / math.sqrt(n)
    _HADAMARD_CACHE[n] = H
    return H


def haar_matrix(n: int) -> np.ndarray:
    """
    Construct a complete orthonormal dyadic Haar matrix.

    Row 0 is the scaling vector. The remaining rows are Haar wavelets from
    coarse to fine scales. The loop stops exactly when n rows have been built.
    """
    if not is_power_of_two(n):
        raise ValueError("Haar size must be a positive power of two.")
    if n in _HAAR_CACHE:
        return _HAAR_CACHE[n]

    rows: List[np.ndarray] = [np.ones(n, dtype=float) / math.sqrt(n)]

    level = 0
    while len(rows) < n:
        block = n // (2 ** level)
        if block < 2:
            break
        half = block // 2

        for j in range(2 ** level):
            if len(rows) >= n:
                break
            start = j * block
            v = np.zeros(n, dtype=float)
            v[start:start + half] = 1.0
            v[start + half:start + block] = -1.0
            norm = np.linalg.norm(v)
            if norm == 0:
                raise RuntimeError("Generated a zero Haar vector; check dyadic loop.")
            rows.append(v / norm)

        level += 1

    Q = np.vstack(rows)
    if Q.shape != (n, n):
        raise RuntimeError(f"Invalid Haar matrix shape {Q.shape}, expected {(n, n)}")

    err = np.linalg.norm(Q @ Q.T - np.eye(n), ord="fro")
    if err > 1e-10:
        raise RuntimeError(f"Haar matrix is not orthonormal; Frobenius error {err}")

    _HAAR_CACHE[n] = Q
    return Q


def block_energies(C: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    a = max(1, n // 4)
    b = max(2, n // 2)
    blocks = [
        (slice(0, a), slice(0, a)),
        (slice(a, b), slice(a, b)),
        (slice(b, n), slice(b, n)),
    ]
    return np.asarray([float(np.sum(C[rs, cs] ** 2)) for rs, cs in blocks], dtype=float)


def regularized_entropy(C: np.ndarray, alpha: float = 0.2) -> float:
    x = C.ravel()
    d = x.size
    delta = alpha / d
    denom = float(np.sum(x * x) + d * delta)
    if denom <= 0:
        return 0.0
    p = (x * x + delta) / denom
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------
# Walsh functions and true toric smoothing.
# ---------------------------------------------------------------------

def walsh_value(m: int, x: float) -> float:
    """
    Paley-ordered Walsh function w_m(x), x in [0,1).

    The function uses binary digits of m and the Rademacher functions.
    It is evaluated away from dyadic ambiguity by clipping x into [0, 1).
    """
    if m == 0:
        return 1.0

    x = x % 1.0
    # Avoid x == 1.0 through modulo; avoid exact dyadic ambiguity by using floor.
    value = 1.0
    bit = 0
    mm = m
    while mm:
        if mm & 1:
            r = math.floor((2 ** (bit + 1)) * x)
            value *= 1.0 if (r % 2 == 0) else -1.0
        mm >>= 1
        bit += 1
    return value


def toric_signed_distance(x: float, t: float) -> float:
    """
    Signed distance from t to x on the torus, in (-1/2, 1/2].
    """
    d = (x - t + 0.5) % 1.0 - 0.5
    return d


_WALSH_JUMP_CACHE: Dict[int, List[Tuple[float, float, float]]] = {}

def walsh_jump_points(m: int) -> List[Tuple[float, float, float]]:
    """
    Return jump points of w_m on the torus.

    Each tuple is (t, left_value, right_value), where left_value is the value
    just before t and right_value is the value just after t. Results are cached
    because smoothed cell averages query the same Walsh indices repeatedly.
    """
    if m in _WALSH_JUMP_CACHE:
        return _WALSH_JUMP_CACHE[m]
    if m == 0:
        _WALSH_JUMP_CACHE[m] = []
        return []

    resolution_power = max(1, m.bit_length())
    denom = 2 ** resolution_power
    eps_probe = 1.0 / (denom * 16.0)

    jumps: List[Tuple[float, float, float]] = []
    for a in range(denom):
        t = a / denom
        left = walsh_value(m, (t - eps_probe) % 1.0)
        right = walsh_value(m, (t + eps_probe) % 1.0)
        if left != right:
            jumps.append((t, left, right))
    _WALSH_JUMP_CACHE[m] = jumps
    return jumps


def smoothed_walsh_value(m: int, x: float, eps: float) -> float:
    """
    Torically smoothed Walsh function with linear ramps around all jumps.
    """
    if m == 0:
        return 1.0

    x = x % 1.0
    for t, left, right in walsh_jump_points(m):
        d = toric_signed_distance(x, t)
        if -eps <= d <= eps:
            s = (d + eps) / (2.0 * eps)
            return (1.0 - s) * left + s * right

    return walsh_value(m, x)


def arc_endpoints(t: float, eps: float) -> Tuple[float, float]:
    return ((t - eps) % 1.0, (t + eps) % 1.0)


def smoothed_walsh_cell_weights(m: int, n: int, c: float) -> np.ndarray:
    """
    Compute exact cell averages of the torically smoothed Walsh probe.

    The returned weights are
        omega_{m,j} = n * int_{j/n}^{(j+1)/n} \tilde w_m(x) dx.

    Since the smoothed Walsh function is piecewise affine, integration over
    a partition containing all cell boundaries and all smoothing-ramp endpoints
    is exact by the trapezoidal rule on each affine subinterval.
    """
    if not is_power_of_two(n):
        raise ValueError("n must be a power of two.")

    if m == 0:
        return np.ones(n, dtype=float)

    eps = c / n
    jumps = walsh_jump_points(m)
    if not jumps:
        return np.ones(n, dtype=float)

    # Check non-overlap conservatively on the retained finite grid.
    jump_positions = sorted(t for t, _, _ in jumps)
    if len(jump_positions) >= 2:
        distances = []
        for i in range(len(jump_positions)):
            a = jump_positions[i]
            b = jump_positions[(i + 1) % len(jump_positions)]
            d = (b - a) % 1.0
            distances.append(d)
        if distances and 2 * eps >= min(distances) - 1e-15:
            raise ValueError(
                f"Smoothing arcs may overlap for m={m}, n={n}, c={c}. "
                f"Use smaller c."
            )

    base_breaks = {0.0, 1.0}
    for j in range(n + 1):
        base_breaks.add(j / n)

    for t, _, _ in jumps:
        a, b = arc_endpoints(t, eps)
        base_breaks.add(a)
        base_breaks.add(b)
        if a > b:
            # Wrapped arc creates a split at 0.
            base_breaks.add(0.0)
            base_breaks.add(1.0)

    breaks = sorted(base_breaks)
    weights = np.zeros(n, dtype=float)

    for left, right in zip(breaks[:-1], breaks[1:]):
        if right <= left:
            continue
        mid = 0.5 * (left + right)
        cell = min(n - 1, int(math.floor(mid * n)))
        f_left = smoothed_walsh_value(m, left + 1e-15, eps)
        f_right = smoothed_walsh_value(m, right - 1e-15, eps)
        integral = 0.5 * (f_left + f_right) * (right - left)
        weights[cell] += n * integral

    return weights


_SMOOTH_WEIGHT_CACHE: Dict[Tuple[int, int, float], np.ndarray] = {}


def get_smoothed_weights(m: int, n: int, c: float) -> np.ndarray:
    key = (m, n, c)
    if key not in _SMOOTH_WEIGHT_CACHE:
        _SMOOTH_WEIGHT_CACHE[key] = smoothed_walsh_cell_weights(m, n, c)
    return _SMOOTH_WEIGHT_CACHE[key]


def smoothed_walsh_coefficients(M: np.ndarray, order: int, c: float) -> np.ndarray:
    """
    Compute low-order torically smoothed Walsh coefficients by exact cell averages:
        A = Omega M Omega^T,
    where Omega[p,j] = omega_{p,j}.
    """
    n = M.shape[0]
    order = min(order, n)
    Omega = np.vstack([get_smoothed_weights(p, n, c) for p in range(order)])
    coeffs = Omega @ M @ Omega.T
    return coeffs.ravel().astype(float)


# ---------------------------------------------------------------------
# Polar Walsh dictionary by deterministic quadrature.
# ---------------------------------------------------------------------

def step_density_value(M: np.ndarray, x: float, y: float) -> float:
    """
    Evaluate FCGR step density F[M](x,y) = n^2 M[row, col].
    x is column coordinate; y is row coordinate.
    """
    n = M.shape[0]
    if x < 0.0 or x >= 1.0 or y < 0.0 or y >= 1.0:
        return 0.0
    col = min(n - 1, int(math.floor(x * n)))
    row = min(n - 1, int(math.floor(y * n)))
    return float(n * n * M[row, col])


def polar_walsh_value(m: int, n: int, r: float, theta: float) -> float:
    return (1.0 / math.sqrt(math.pi)) * walsh_value(m, r * r) * walsh_value(n, (theta / (2.0 * math.pi)) % 1.0)


def polar_walsh_block_energies(
    M: np.ndarray,
    disks: Sequence[Tuple[float, float, float]],
    max_m: int,
    max_n: int,
    radial_nodes: int,
    angular_nodes: int,
) -> np.ndarray:
    """
    Compute fixed local polar Walsh coefficient energies by deterministic midpoint quadrature.

    The exact theory is stated for analytical polar integrals. This function computes
    their deterministic quadrature approximations and should be paired with a
    convergence check when used for publication-level experiments.
    """
    features: List[float] = []

    dr = 1.0 / radial_nodes
    dtheta = 2.0 * math.pi / angular_nodes

    radial_grid = (np.arange(radial_nodes, dtype=float) + 0.5) * dr
    theta_grid = (np.arange(angular_nodes, dtype=float) + 0.5) * dtheta

    for cx, cy, rho in disks:
        coeffs = np.zeros((max_m, max_n), dtype=float)

        for r in radial_grid:
            for theta in theta_grid:
                x = cx + rho * r * math.cos(theta)
                y = cy + rho * r * math.sin(theta)
                fval = step_density_value(M, x, y)
                weight = r * dr * dtheta
                for p in range(max_m):
                    for q in range(max_n):
                        coeffs[p, q] += fval * polar_walsh_value(p, q, r, theta) * weight

        # Use interpretable blocks: DC, low non-DC, all retained energy.
        dc = coeffs[0, 0] ** 2
        total = float(np.sum(coeffs ** 2))
        low_non_dc = max(0.0, total - dc)
        features.extend([float(dc), float(low_non_dc), float(total)])

    return np.asarray(features, dtype=float)


# ---------------------------------------------------------------------
# Descriptor modes.
# ---------------------------------------------------------------------

def spectral_core_features(seq: str, cfg: DescriptorConfig, include_walsh: bool, include_haar: bool) -> List[float]:
    features: List[float] = []
    for k in cfg.scales:
        M = fcgr(seq, k)
        n = 2 ** k
        if include_walsh:
            W = hadamard_matrix(n)
            CW = W @ M @ W.T
            features.extend(block_energies(CW))
            features.append(regularized_entropy(CW, alpha=cfg.alpha_entropy))
        if include_haar:
            Q = haar_matrix(n)
            CH = Q @ M @ Q.T
            features.extend(block_energies(CH))
            features.append(regularized_entropy(CH, alpha=cfg.alpha_entropy))
    return features


def descriptor_no_rc(seq: str, mode: str, cfg: DescriptorConfig) -> np.ndarray:
    features: List[float] = []

    if mode == "raw_kmer":
        return raw_kmer_lexicographic(seq, cfg.scales)

    if mode == "flattened_fcgr":
        return flattened_fcgr(seq, cfg.scales)

    if mode == "d2_statistic":
        return d2_statistic_features(seq, cfg.scales)

    if mode == "mldsp_proxy":
        return mldsp_proxy_features(seq, cfg.scales)

    if mode == "walsh_only":
        return np.asarray(spectral_core_features(seq, cfg, include_walsh=True, include_haar=False), dtype=float)

    if mode == "haar_only":
        return np.asarray(spectral_core_features(seq, cfg, include_walsh=False, include_haar=True), dtype=float)

    if mode in {"walsh_haar", "walsh_haar_smooth", "full"}:
        features.extend(spectral_core_features(seq, cfg, include_walsh=True, include_haar=True))

    if mode in {"walsh_haar_smooth", "full"}:
        for k in cfg.scales:
            M = fcgr(seq, k)
            features.extend(smoothed_walsh_coefficients(M, order=cfg.smooth_order, c=cfg.smooth_c))

    if mode == "full":
        for k in cfg.scales:
            M = fcgr(seq, k)
            features.extend(
                polar_walsh_block_energies(
                    M,
                    disks=cfg.polar_disks,
                    max_m=cfg.polar_order_m,
                    max_n=cfg.polar_order_n,
                    radial_nodes=cfg.polar_radial_nodes,
                    angular_nodes=cfg.polar_angular_nodes,
                )
            )

    if not features:
        raise ValueError(f"Unknown descriptor mode: {mode}")

    return np.asarray(features, dtype=float)


def descriptor(seq: str, mode: str, cfg: DescriptorConfig, rc_average: bool = False) -> np.ndarray:
    x = descriptor_no_rc(seq, mode, cfg)
    if not rc_average:
        return x
    xr = descriptor_no_rc(reverse_complement(seq), mode, cfg)
    if xr.shape != x.shape:
        raise RuntimeError("Reverse-complement descriptor dimension mismatch.")
    return 0.5 * (x + xr)


# ---------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------

def build_cv_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    repeats: int,
    random_seed: int,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    unique_groups = np.unique(groups)
    class_counts = np.bincount(y)
    groups_per_class = [len(set(groups[y == cls])) for cls in np.unique(y)]
    max_splits = min(n_splits, len(unique_groups), int(class_counts.min()), min(groups_per_class))

    if max_splits < 2:
        raise RuntimeError(
            f"Not enough independent groups/classes for group-aware CV. "
            f"unique_groups={len(unique_groups)}, class_counts={class_counts.tolist()}, "
            f"groups_per_class={groups_per_class}"
        )

    splits: List[Tuple[np.ndarray, np.ndarray, int]] = []

    if HAS_STRATIFIED_GROUP_KFOLD:
        for r in range(repeats):
            cv = StratifiedGroupKFold(
                n_splits=max_splits,
                shuffle=True,
                random_state=random_seed + r,
            )
            for train, test in cv.split(np.zeros_like(y), y, groups):
                splits.append((train, test, r))
    else:
        print("[warning] StratifiedGroupKFold is unavailable; falling back to GroupKFold.")
        cv = GroupKFold(n_splits=max_splits)
        for r in range(repeats):
            for train, test in cv.split(np.zeros_like(y), y, groups):
                splits.append((train, test, r))

    return splits


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, seed: int = 20260619, n_boot: int = 2000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = np.mean(sample)
    alpha = 1.0 - confidence
    return (float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0)))


def evaluate_descriptor(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: Sequence[str],
    splits: Sequence[Tuple[np.ndarray, np.ndarray, int]],
    random_seed: int,
) -> Tuple[MetricSummary, List[Dict[str, object]]]:
    fold_rows: List[Dict[str, object]] = []

    fold_acc: List[float] = []
    fold_bacc: List[float] = []
    fold_f1: List[float] = []
    fold_auc: List[float] = []

    for fold_id, (train, test, repeat_id) in enumerate(splits):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto",
                random_state=random_seed,
            ),
        )
        clf.fit(X[train], y[train])
        pred = clf.predict(X[test])

        try:
            prob = clf.predict_proba(X[test])
        except Exception:
            prob = None

        acc = accuracy_score(y[test], pred)
        bacc = balanced_accuracy_score(y[test], pred)
        mf1 = f1_score(y[test], pred, average="macro")

        auc = float("nan")
        if prob is not None and len(np.unique(y[test])) == len(class_names):
            try:
                auc = roc_auc_score(y[test], prob, multi_class="ovr", average="macro")
            except Exception:
                auc = float("nan")

        fold_acc.append(acc)
        fold_bacc.append(bacc)
        fold_f1.append(mf1)
        fold_auc.append(auc)

        for idx, true_label, pred_label in zip(test, y[test], pred):
            fold_rows.append({
                "descriptor": name,
                "repeat": repeat_id,
                "fold": fold_id,
                "sample_index": int(idx),
                "true_label": class_names[int(true_label)],
                "predicted_label": class_names[int(pred_label)],
                "group": str(groups[idx]),
            })

    acc_ci = bootstrap_ci(np.asarray(fold_acc), seed=random_seed + 1)
    bacc_ci = bootstrap_ci(np.asarray(fold_bacc), seed=random_seed + 2)
    f1_ci = bootstrap_ci(np.asarray(fold_f1), seed=random_seed + 3)
    auc_arr = np.asarray(fold_auc, dtype=float)
    auc_arr = auc_arr[np.isfinite(auc_arr)]
    auc_ci = bootstrap_ci(auc_arr, seed=random_seed + 4) if auc_arr.size else (None, None)

    summary = MetricSummary(
        descriptor=name,
        n_features=int(X.shape[1]),
        accuracy_mean=float(np.mean(fold_acc)),
        accuracy_ci_low=acc_ci[0],
        accuracy_ci_high=acc_ci[1],
        balanced_accuracy_mean=float(np.mean(fold_bacc)),
        balanced_accuracy_ci_low=bacc_ci[0],
        balanced_accuracy_ci_high=bacc_ci[1],
        macro_f1_mean=float(np.mean(fold_f1)),
        macro_f1_ci_low=f1_ci[0],
        macro_f1_ci_high=f1_ci[1],
        macro_auroc_mean=float(np.mean(auc_arr)) if auc_arr.size else None,
        macro_auroc_ci_low=auc_ci[0],
        macro_auroc_ci_high=auc_ci[1],
    )

    return summary, fold_rows


def write_results(results: Sequence[MetricSummary]) -> None:
    path = Path("outputs/benchmark_results.tsv")
    fields = list(asdict(results[0]).keys()) if results else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def write_fold_predictions(rows: Sequence[Dict[str, object]]) -> None:
    path = Path("outputs/fold_predictions.tsv")
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Runtime with warmup.
# ---------------------------------------------------------------------

def measure_runtime(records: Sequence[RecordInfo], cfg: DescriptorConfig, repeats: int = 30, warmup: int = 5) -> List[Dict[str, object]]:
    lengths = [1000, 3000, 6000]
    if not records:
        return []

    base_seq = max(records, key=lambda r: r.length).sequence
    rows: List[Dict[str, object]] = []

    for L in lengths:
        seq = (base_seq * ((L // len(base_seq)) + 1))[:L]

        def time_fn(fn: Callable[[], np.ndarray]) -> Tuple[float, float]:
            for _ in range(warmup):
                fn()
            vals = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn()
                vals.append(time.perf_counter() - t0)
            return float(np.median(vals)), float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))

        raw_median, raw_iqr = time_fn(lambda: raw_kmer_lexicographic(seq, (4,)))
        full_median, full_iqr = time_fn(lambda: descriptor(seq, "full", cfg, rc_average=True))

        rows.append({
            "length": L,
            "raw_4mer_median": raw_median,
            "raw_4mer_iqr": raw_iqr,
            "full_descriptor_median": full_median,
            "full_descriptor_iqr": full_iqr,
            "ratio": full_median / raw_median if raw_median > 0 else float("nan"),
        })

    path = Path("outputs/runtime_results.tsv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return rows


# ---------------------------------------------------------------------
# LaTeX table generation.
# ---------------------------------------------------------------------

def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("#", "\\#")
    )


def generate_benchmark_results_table(results: Sequence[MetricSummary]) -> None:
    path = Path("generated_tables/benchmark_results_table.tex")
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Accession-fixed benchmark results. Values are means over repeated group-aware folds; brackets show bootstrap 95\\% confidence intervals over fold-level scores.}\n")
        f.write("\\label{tab:real-benchmark-results}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Descriptor & Features & Accuracy & Balanced accuracy & Macro-F1 \\\\\n")
        f.write("\\midrule\n")
        for r in results:
            f.write(
                f"{latex_escape(r.descriptor)} & {r.n_features} & "
                f"{r.accuracy_mean:.3f} [{r.accuracy_ci_low:.3f},{r.accuracy_ci_high:.3f}] & "
                f"{r.balanced_accuracy_mean:.3f} [{r.balanced_accuracy_ci_low:.3f},{r.balanced_accuracy_ci_high:.3f}] & "
                f"{r.macro_f1_mean:.3f} [{r.macro_f1_ci_low:.3f},{r.macro_f1_ci_high:.3f}] \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def generate_runtime_table(rows: Sequence[Dict[str, object]]) -> None:
    path = Path("generated_tables/runtime_results_table.tex")
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Feature-extraction runtime after warmup. Values are median seconds with interquartile range in parentheses.}\n")
        f.write("\\label{tab:runtime-results}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{rrrr}\n")
        f.write("\\toprule\n")
        f.write("Length & Raw 4-mer & Full descriptor + RC & Ratio \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{int(row['length'])} & "
                f"{row['raw_4mer_median']:.5f} ({row['raw_4mer_iqr']:.5f}) & "
                f"{row['full_descriptor_median']:.5f} ({row['full_descriptor_iqr']:.5f}) & "
                f"{row['ratio']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def generate_manifest_accessions_table() -> None:
    path = Path("generated_tables/manifest_accessions_table.tex")
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[p]\n")
        f.write("\\centering\n")
        f.write("\\caption{Fixed accession list used in the public viral benchmark.}\n")
        f.write("\\label{tab:accession-list}\n")
        f.write("\\scriptsize\n")
        f.write("\\begin{tabularx}{\\textwidth}{lX}\n")
        f.write("\\toprule\n")
        f.write("Family & Accessions \\\\\n")
        f.write("\\midrule\n")
        for label, accs in ACCESSIONS.items():
            f.write(f"{latex_escape(label)} & {latex_escape(', '.join(accs))} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabularx}\n")
        f.write("\\end{table}\n")


# ---------------------------------------------------------------------
# Self-tests.
# ---------------------------------------------------------------------

def self_test(cfg: DescriptorConfig) -> None:
    print("[self-test] Running mathematical implementation checks")

    for cx, cy, rho in cfg.polar_disks:
        if not (rho > 0 and rho <= cx <= 1.0 - rho and rho <= cy <= 1.0 - rho):
            raise AssertionError(f"Polar disk {(cx, cy, rho)} is not contained in the unit square.")

    seq = "ACGTACGTACGTNNACGT"
    M = fcgr(seq, 3)
    if not (0.0 <= np.sum(M) <= 1.0 + 1e-12):
        raise AssertionError("FCGR normalization failed.")

    for n in (2, 4, 8, 16, 32):
        W = hadamard_matrix(n)
        Q = haar_matrix(n)
        if np.linalg.norm(W @ W.T - np.eye(n), ord="fro") > 1e-10:
            raise AssertionError(f"Hadamard orthogonality failed for n={n}")
        if np.linalg.norm(Q @ Q.T - np.eye(n), ord="fro") > 1e-10:
            raise AssertionError(f"Haar orthogonality failed for n={n}")

    seq2 = "ACGTACGTACGT"
    M2 = fcgr(seq2, 3)
    W8 = hadamard_matrix(8)
    C = W8 @ M2 @ W8.T
    if abs(np.linalg.norm(C, "fro") - np.linalg.norm(M2, "fro")) > 1e-10:
        raise AssertionError("Walsh energy preservation failed.")

    weights0 = get_smoothed_weights(0, 8, cfg.smooth_c)
    if np.linalg.norm(weights0 - np.ones(8)) > 1e-12:
        raise AssertionError("Smoothed DC cell weights failed.")

    coeffs = smoothed_walsh_coefficients(M2, order=cfg.smooth_order, c=cfg.smooth_c)
    if not np.all(np.isfinite(coeffs)):
        raise AssertionError("Smoothed coefficients contain non-finite values.")

    polar = polar_walsh_block_energies(
        M2,
        disks=cfg.polar_disks,
        max_m=cfg.polar_order_m,
        max_n=cfg.polar_order_n,
        radial_nodes=cfg.polar_radial_nodes,
        angular_nodes=cfg.polar_angular_nodes,
    )
    if not np.all(np.isfinite(polar)):
        raise AssertionError("Polar features contain non-finite values.")

    x = descriptor(seq2, "full", cfg, rc_average=True)
    if not np.all(np.isfinite(x)):
        raise AssertionError("Full descriptor contains non-finite values.")


    # Reverse-complement averaging should be invariant.
    x_rc = descriptor(seq2, "full", cfg, rc_average=True)
    x_rc2 = descriptor(reverse_complement(seq2), "full", cfg, rc_average=True)
    if np.linalg.norm(x_rc - x_rc2) > 1e-10:
        raise AssertionError("Reverse-complement averaging is not invariant.")

    # Refined FCGR edit bound for one internal substitution.
    s0 = "ACGT" * 20
    s1 = s0[:10] + ("A" if s0[10] != "A" else "C") + s0[11:]
    k = 4
    bound = (2 * k * 1 + 0 + abs(len(s0) - len(s1))) / min(len(s0)-k+1, len(s1)-k+1)
    if np.linalg.norm(fcgr(s0, k) - fcgr(s1, k), ord=1) > bound + 1e-12:
        raise AssertionError("FCGR single-substitution bound failed.")

    # Entropy Lipschitz sanity check on random bounded vectors.
    rng = np.random.default_rng(123)
    x0 = rng.normal(size=16); x0 = x0 / max(1.0, np.linalg.norm(x0))
    y0 = x0 + 0.001 * rng.normal(size=16); y0 = y0 / max(1.0, np.linalg.norm(y0))
    alpha = cfg.alpha_entropy; d = 16; delta = alpha / d
    L_const = 4.0/(d*delta) * (1.0 + abs(math.log(delta/(1.0+d*delta))))
    hx = -np.sum(((x0*x0+delta)/(np.sum(x0*x0)+d*delta))*np.log((x0*x0+delta)/(np.sum(x0*x0)+d*delta)))
    hy = -np.sum(((y0*y0+delta)/(np.sum(y0*y0)+d*delta))*np.log((y0*y0+delta)/(np.sum(y0*y0)+d*delta)))
    if abs(hx-hy) > L_const*np.linalg.norm(x0-y0) + 1e-10:
        raise AssertionError("Entropy Lipschitz sanity check failed.")

    # Constant-density polar check: finite quadrature should produce finite nonnegative energies.
    M_const = np.ones((8, 8), dtype=float) / 64.0
    polar_const = polar_walsh_block_energies(M_const, cfg.polar_disks, 2, 2, 16, 32)
    if not (np.all(np.isfinite(polar_const)) and np.all(polar_const >= -1e-12)):
        raise AssertionError("Polar constant-density test failed.")

    print("[self-test] All checks passed")



# ---------------------------------------------------------------------
# Robustness, quadrature and sensitivity diagnostics.
# ---------------------------------------------------------------------

def random_dna(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list(DNA_ALPHABET), size=length))


def mutate_sequence(seq: str, rate: float, rng: np.random.Generator, indel_fraction: float = 0.33) -> Tuple[str, int, int, int]:
    chars = list(seq)
    if rate <= 0:
        return seq, 0, 0, 0
    n_edits = max(1, int(round(rate * len(seq))))
    e_sub = e_indel = 0
    for _ in range(n_edits):
        if not chars:
            chars.append(str(rng.choice(list(DNA_ALPHABET))))
            e_indel += 1
            continue
        op = rng.random()
        pos = int(rng.integers(0, len(chars)))
        if op < 1.0 - indel_fraction:
            old = chars[pos]
            choices = [b for b in DNA_ALPHABET if b != old]
            chars[pos] = str(rng.choice(choices))
            e_sub += 1
        elif op < 1.0 - indel_fraction / 2.0:
            chars.insert(pos, str(rng.choice(list(DNA_ALPHABET))))
            e_indel += 1
        else:
            del chars[pos]
            e_indel += 1
    return "".join(chars), n_edits, e_sub, e_indel


def max_delta_for_path(seq: str, mut: str, cfg: DescriptorConfig, e_sub: int, e_indel: int) -> float:
    vals = []
    for k in cfg.scales:
        ns = len(seq) - k + 1
        nt = len(mut) - k + 1
        nmin = max(1, min(ns, nt))
        vals.append((2*k*e_sub + (2*k-1)*e_indel + abs(len(seq)-len(mut))) / nmin)
    return float(max(vals))


def generate_robustness_tables(cfg: DescriptorConfig, seed: int = 20260619) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for rate in [0.005, 0.01, 0.02, 0.05]:
        dists = []
        coss = []
        deltas = []
        edits = []
        for _ in range(24):
            seq = random_dna(2000, rng)
            mut, n_edits, e_sub, e_indel = mutate_sequence(seq, rate, rng, indel_fraction=0.33)
            x = descriptor(seq, "full", cfg, rc_average=True)
            y = descriptor(mut, "full", cfg, rc_average=True)
            dists.append(float(np.linalg.norm(x-y)))
            denom = float(np.linalg.norm(x)*np.linalg.norm(y))
            coss.append(float(np.dot(x,y)/denom) if denom > 0 else float("nan"))
            deltas.append(max_delta_for_path(seq, mut, cfg, e_sub, e_indel))
            edits.append(n_edits)
        rows.append((rate, np.mean(edits), np.mean(deltas), np.mean(dists), np.std(dists), np.mean(coss), np.std(coss)))
    tex = ["\\begin{table}[t]", "\\centering", "\\caption{Synthetic robustness under controlled Levenshtein edits. Values are generated by the companion script.}", "\\label{tab:edit-robustness}", "\\small", "\\begin{tabular}{lrrrr}", "\\toprule", "Edit rate & Mean edits & Mean $\\max_k\\Delta_k$ & Descriptor $\\ell^2$ distance & Cosine similarity\\\\", "\\midrule"]
    for rate, me, md, dist, dist_sd, cos, cos_sd in rows:
        tex.append(f"{100*rate:.1f}\\% & {me:.1f} & {md:.4f} & ${dist:.4f}\\pm{dist_sd:.4f}$ & ${cos:.5f}\\pm{cos_sd:.5f}$\\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path("generated_tables/edit_robustness_table.tex").write_text("\n".join(tex), encoding="utf-8")
    if HAS_MATPLOTLIB:
        Path("figures").mkdir(exist_ok=True)
        xs = [r[2] for r in rows]
        ys = [r[3] for r in rows]
        yerr = [r[4] for r in rows]
        plt.figure(figsize=(6,4))
        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3)
        plt.xlabel(r"Mean normalized edit level $\max_k\Delta_k$")
        plt.ylabel(r"Descriptor $\ell^2$ distance")
        plt.title("Synthetic edit robustness")
        plt.tight_layout()
        plt.savefig("figures/stability_curves.pdf")
        plt.close()


def generate_polar_quadrature_table(cfg: DescriptorConfig, seed: int = 20260619) -> None:
    rng = np.random.default_rng(seed)
    seq = random_dna(2500, rng)
    M = fcgr(seq, max(cfg.scales))
    grids = [(8,16), (16,32), (32,64)]
    ref = polar_walsh_block_energies(M, cfg.polar_disks, cfg.polar_order_m, cfg.polar_order_n, 64, 128)
    rows = []
    for br, bt in grids:
        t0 = time.perf_counter()
        val = polar_walsh_block_energies(M, cfg.polar_disks, cfg.polar_order_m, cfg.polar_order_n, br, bt)
        elapsed = 1000.0*(time.perf_counter()-t0)
        rel = float(np.linalg.norm(val-ref)/(np.linalg.norm(ref)+1e-15))
        rows.append((br,bt,rel,elapsed))
    tex = ["\\begin{table}[t]", "\\centering", "\\caption{Convergence of deterministic polar quadrature. Relative error is measured against a $64\\times128$ reference grid on a fixed synthetic sequence.}", "\\label{tab:polar-convergence}", "\\small", "\\begin{tabular}{lrr}", "\\toprule", "Grid $(B_r\\times B_\\theta)$ & Relative $\\ell^2$ error & Runtime (ms)\\\\", "\\midrule"]
    for br,bt,rel,elapsed in rows:
        tex.append(f"${br}\\times{bt}$ & {rel:.6f} & {elapsed:.1f}\\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path("generated_tables/polar_convergence_table.tex").write_text("\n".join(tex), encoding="utf-8")


def feature_dimension(mode: str, cfg: DescriptorConfig) -> int:
    seq = "ACGT" * 200
    return int(descriptor(seq, mode, cfg, rc_average=False).shape[0])


def generate_sensitivity_k_table(seed: int = 20260619) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for kmax in [3,4,5,6]:
        cfg = DescriptorConfig(scales=tuple(range(3,kmax+1)))
        seq = random_dna(3000, rng)
        mut, _, e_sub, e_indel = mutate_sequence(seq, 0.01, rng, indel_fraction=0.33)
        t0 = time.perf_counter()
        x = descriptor(seq, "full", cfg, rc_average=True)
        y = descriptor(mut, "full", cfg, rc_average=True)
        runtime = time.perf_counter()-t0
        dist = float(np.linalg.norm(x-y))
        cos = float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
        rows.append((kmax, x.size, dist, cos, runtime))
    tex=["\\begin{table}[t]","\\centering","\\caption{Sensitivity with respect to the maximum FCGR scale $k_{\\max}$ on a controlled 1\\% edit experiment. Classification columns are generated by the accession benchmark when run with the corresponding scale set.}","\\label{tab:sensitivity-k}","\\small","\\begin{tabular}{lrrrr}","\\toprule","$k_{\\max}$ & Dimension & Mean $\\ell^2$ stability & Cosine similarity & Runtime pair (s)\\\\","\\midrule"]
    for kmax, dim, dist, cos, runtime in rows:
        tex.append(f"{kmax} & {dim} & {dist:.5f} & {cos:.5f} & {runtime:.3f}\\\\")
    tex += ["\\bottomrule","\\end{tabular}","\\end{table}"]
    Path("generated_tables/sensitivity_k_table.tex").write_text("\n".join(tex), encoding="utf-8")



def finite_q_value(k: int, c: float) -> float:
    return math.sqrt((2.0/3.0) * sum(len(walsh_jump_points(m)) * c * (2.0 ** (-k)) for m in range(2 ** k)))


def generate_finite_riesz_table(cfg: DescriptorConfig) -> None:
    c_values = [0.005, 0.010, 0.025, 0.050]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Finite smoothing diagnostics for retained Walsh systems. The interval is meaningful as a perturbative finite-system bound only when $q_k(c)<1$.}",
        "\\label{tab:finite-riesz}",
        "\\small",
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "$k$ & $c$ & $q_k(c)$ & interval $(1\\pm q_k)^2$\\\\",
        "\\midrule",
    ]
    for k in cfg.scales:
        for c in c_values:
            q = finite_q_value(k, c)
            lo = (1.0-q)**2 if q < 1 else float('nan')
            hi = (1.0+q)**2
            interval = f"[{lo:.3f},{hi:.3f}]" if q < 1 else f"not used ({hi:.3f})"
            lines.append(f"{k} & {c:.3f} & {q:.3f} & {interval}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path("generated_tables/finite_riesz_table.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_sequencing_robustness_table(cfg: DescriptorConfig, seed: int = 20260619) -> None:
    rng = np.random.default_rng(seed + 17)
    scenarios = [
        ("Substitution-dominant 0.1\\%", 0.001, 0.00),
        ("Mixed short-read 0.5\\%", 0.005, 0.20),
        ("Mixed long-read 1.0\\%", 0.010, 0.45),
        ("Homopolymer-biased 1.0\\%", 0.010, 0.65),
        ("No-edit control", 0.000, 0.00),
    ]
    rows = []
    for name, rate, indel_fraction in scenarios:
        edits=[]; dists=[]; coss=[]; deltas=[]
        for _ in range(12):
            seq = random_dna(1500, rng)
            mut, n_edits, e_sub, e_indel = mutate_sequence(seq, rate, rng, indel_fraction=indel_fraction)
            x = descriptor(seq, "full", cfg, rc_average=True)
            y = descriptor(mut, "full", cfg, rc_average=True)
            denom = float(np.linalg.norm(x)*np.linalg.norm(y))
            edits.append(n_edits)
            dists.append(float(np.linalg.norm(x-y)))
            coss.append(float(np.dot(x,y)/denom) if denom > 0 else float("nan"))
            deltas.append(max_delta_for_path(seq, mut, cfg, e_sub, e_indel))
        rows.append((name, np.mean(edits), np.mean(deltas), np.mean(dists), np.std(dists), np.mean(coss)))
    tex=["\\begin{table}[t]","\\centering","\\caption{Robustness under sequencing-like error profiles generated by the companion script. These controlled simulations test perturbation response and are not platform-specific empirical calibration.}","\\label{tab:sequencing-robustness}","\\small","\\begin{tabular}{lrrrr}","\\toprule","Scenario & Mean edits & Mean $\\max_k\\Delta_k$ & Descriptor $\\ell^2$ distance & Cosine similarity\\\\","\\midrule"]
    for name, edits, delta, dist, sd, cos in rows:
        tex.append(f"{latex_escape(name)} & {edits:.1f} & {delta:.4f} & ${dist:.5f}\\pm{sd:.5f}$ & {cos:.6f}\\\\")
    tex += ["\\bottomrule","\\end{tabular}","\\end{table}"]
    Path("generated_tables/sequencing_robustness_table.tex").write_text("\n".join(tex), encoding="utf-8")


def generate_sensitivity_c_alpha_table(seed: int = 20260619) -> None:
    rng = np.random.default_rng(seed + 23)
    seq = random_dna(1500, rng)
    mut, _, e_sub, e_indel = mutate_sequence(seq, 0.01, rng, indel_fraction=0.33)
    configs = [(0.005,0.05),(0.010,0.20),(0.025,0.20),(0.050,0.20),(0.025,1.00)]
    tex=["\\begin{table}[t]","\\centering","\\caption{Sensitivity to the toric smoothing parameter $c$ and entropy parameter $\\alpha$ on a fixed 1\\% edit experiment.}","\\label{tab:sensitivity-c-alpha}","\\small","\\begin{tabular}{rrrrr}","\\toprule","$c$ & $\\alpha$ & finite $q_{k_{\\max}}(c)$ & Descriptor $\\ell^2$ distance & Cosine similarity\\\\","\\midrule"]
    for c, alpha in configs:
        cfg = DescriptorConfig(alpha_entropy=alpha, smooth_c=c)
        x = descriptor(seq, "full", cfg, rc_average=True)
        y = descriptor(mut, "full", cfg, rc_average=True)
        denom = float(np.linalg.norm(x)*np.linalg.norm(y))
        q = finite_q_value(max(cfg.scales), c)
        tex.append(f"{c:.3f} & {alpha:.2f} & {q:.3f} & {np.linalg.norm(x-y):.5f} & {float(np.dot(x,y)/denom):.6f}\\\\")
    tex += ["\\bottomrule","\\end{tabular}","\\end{table}"]
    Path("generated_tables/sensitivity_c_alpha_table.tex").write_text("\n".join(tex), encoding="utf-8")
    if HAS_MATPLOTLIB:
        Path("figures").mkdir(exist_ok=True)
        c_vals = [c for c, _ in configs]
        sims = []
        for c, alpha in configs:
            cfg = DescriptorConfig(alpha_entropy=alpha, smooth_c=c)
            x = descriptor(seq, "full", cfg, rc_average=True)
            y = descriptor(mut, "full", cfg, rc_average=True)
            denom = float(np.linalg.norm(x)*np.linalg.norm(y))
            sims.append(float(np.dot(x,y)/denom) if denom > 0 else float("nan"))
        plt.figure(figsize=(6,4))
        plt.plot(c_vals, sims, marker="o")
        plt.xlabel("Smoothing parameter c")
        plt.ylabel("Cosine similarity under fixed edit")
        plt.title("Smoothing/entropy sensitivity")
        plt.tight_layout()
        plt.savefig("figures/smoothing_tradeoff.pdf")
        plt.close()


def read_benchmark_results_tsv() -> List[MetricSummary]:
    path = Path("outputs/benchmark_results.tsv")
    if not path.exists():
        return []
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        reader=csv.DictReader(f, delimiter="\t")
        for r in reader:
            def opt(name: str) -> Optional[float]:
                val = r.get(name, "")
                return None if val in {"", "None", "NA"} else float(val)
            rows.append(MetricSummary(
                descriptor=r["descriptor"], n_features=int(r["n_features"]),
                accuracy_mean=float(r["accuracy_mean"]), accuracy_ci_low=float(r["accuracy_ci_low"]), accuracy_ci_high=float(r["accuracy_ci_high"]),
                balanced_accuracy_mean=float(r["balanced_accuracy_mean"]), balanced_accuracy_ci_low=float(r["balanced_accuracy_ci_low"]), balanced_accuracy_ci_high=float(r["balanced_accuracy_ci_high"]),
                macro_f1_mean=float(r["macro_f1_mean"]), macro_f1_ci_low=float(r["macro_f1_ci_low"]), macro_f1_ci_high=float(r["macro_f1_ci_high"]),
                macro_auroc_mean=opt("macro_auroc_mean"), macro_auroc_ci_low=opt("macro_auroc_ci_low"), macro_auroc_ci_high=opt("macro_auroc_ci_high"),
            ))
    return rows


def generate_method_comparison_table(results: Optional[Sequence[MetricSummary]] = None) -> None:
    if results is None:
        results = read_benchmark_results_tsv()
    if not results:
        raise RuntimeError("Cannot generate method_comparison_table.tex before benchmark_results.tsv exists. Run --run-benchmark first.")
    selected = [r for r in results if any(key in r.descriptor for key in ["Raw", "Flattened", "D2", "ML-DSP", "Full descriptor + RC"])]
    tex=["\\begin{table}[t]","\\centering","\\caption{Comparison with composition, flattened-FCGR, D2-style and DSP-style baselines. The ML-DSP-style row is a transparent FFT proxy, not the official ML-DSP software.}","\\label{tab:method-comparison}","\\small","\\begin{tabular}{lrrrr}","\\toprule","Descriptor & Features & Accuracy & Balanced accuracy & Macro-F1\\\\","\\midrule"]
    for r in selected:
        tex.append(f"{latex_escape(r.descriptor)} & {r.n_features} & {r.accuracy_mean:.3f} & {r.balanced_accuracy_mean:.3f} & {r.macro_f1_mean:.3f}\\\\")
    tex += ["\\bottomrule","\\end{tabular}","\\end{table}"]
    Path("generated_tables/method_comparison_table.tex").write_text("\n".join(tex), encoding="utf-8")


def spectral_block_matrix(seq: str, cfg: DescriptorConfig) -> np.ndarray:
    vals=[]
    for k in cfg.scales:
        M = fcgr(seq,k); n=2**k
        W=hadamard_matrix(n); Q=haar_matrix(n)
        vals.append(block_energies(W@M@W.T))
        vals.append(block_energies(Q@M@Q.T))
    return np.concatenate(vals)


def generate_block_importance_table(records: Sequence[RecordInfo], cfg: DescriptorConfig) -> None:
    if not records:
        raise RuntimeError("Cannot generate block_importance_table.tex without benchmark records. Run with --run-benchmark or --download/--offline.")
    labels=sorted({r.label for r in records})
    X=np.vstack([spectral_block_matrix(r.sequence,cfg) for r in records])
    y=np.asarray([labels.index(r.label) for r in records])
    # columns repeat [low,mid,high] for WH and Haar at each scale; aggregate same position.
    rows=[]
    for bidx, name in enumerate(["low", "mid", "high"]):
        cols=[i for i in range(X.shape[1]) if i % 3 == bidx]
        Z=X[:,cols]
        overall=np.mean(Z,axis=0)
        between=0.0; within=0.0
        for c in range(len(labels)):
            Zc=Z[y==c]
            if Zc.size==0: continue
            mu=np.mean(Zc,axis=0)
            between += Zc.shape[0]*float(np.sum((mu-overall)**2))
            within += float(np.sum((Zc-mu)**2))
        score=between/(within+1e-15)
        rows.append((name, len(cols), score, between, within))
    tex=["\\begin{table}[t]","\\centering","\\caption{Block-level descriptive importance based on the between/within class scatter ratio of spectral block-energy groups. This is an interpretability diagnostic, not a causal feature-importance claim.}","\\label{tab:block-importance}","\\small","\\begin{tabular}{lrrrr}","\\toprule","Block group & Coordinates & Scatter ratio & Between scatter & Within scatter\\\\","\\midrule"]
    for name, ncols, score, between, within in rows:
        tex.append(f"{name} & {ncols} & {score:.4f} & {between:.4e} & {within:.4e}\\\\")
    tex += ["\\bottomrule","\\end{tabular}","\\end{table}"]
    Path("generated_tables/block_importance_table.tex").write_text("\n".join(tex), encoding="utf-8")


def generate_embedding_figure(records: Sequence[RecordInfo], cfg: DescriptorConfig) -> None:
    if not HAS_MATPLOTLIB:
        print("[warning] matplotlib is unavailable; embedding figure was not generated.")
        return
    if not records:
        print("[warning] No records available; embedding figure was not generated.")
        return
    X=np.vstack([descriptor(r.sequence,"full",cfg,rc_average=True) for r in records])
    labels=sorted({r.label for r in records})
    y=np.asarray([labels.index(r.label) for r in records])
    X=(X-np.mean(X,axis=0))/(np.std(X,axis=0)+1e-12)
    U,S,Vt=np.linalg.svd(X,full_matrices=False)
    Z=U[:,:2]*S[:2]
    Path("figures").mkdir(exist_ok=True)
    plt.figure(figsize=(6,4))
    for i,lab in enumerate(labels):
        pts=Z[y==i]
        plt.scatter(pts[:,0],pts[:,1],label=lab,s=28)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of full descriptor + RC")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/embedding_pca.pdf")
    plt.close()


# ---------------------------------------------------------------------
# Main benchmark.
# ---------------------------------------------------------------------

def run_benchmark(records: Sequence[RecordInfo], cfg: DescriptorConfig, repeats: int, folds: int, seed: int) -> List[MetricSummary]:
    class_names = sorted({r.label for r in records})
    y_map = {name: i for i, name in enumerate(class_names)}
    y = np.asarray([y_map[r.label] for r in records], dtype=int)
    groups = np.asarray([r.group for r in records], dtype=object)

    splits = build_cv_splits(y, groups, n_splits=folds, repeats=repeats, random_seed=seed)

    modes: List[Tuple[str, str, bool]] = [
        ("Raw lexicographic k-mer", "raw_kmer", False),
        ("Flattened FCGR (permutation baseline)", "flattened_fcgr", False),
        ("D2-style centered k-mer", "d2_statistic", False),
        ("ML-DSP-style FFT proxy", "mldsp_proxy", False),
        ("Walsh only", "walsh_only", False),
        ("Haar only", "haar_only", False),
        ("Walsh--Haar", "walsh_haar", False),
        ("Walsh--Haar + toric smoothing", "walsh_haar_smooth", False),
        ("Full descriptor", "full", False),
        ("Full descriptor + RC averaging", "full", True),
    ]

    results: List[MetricSummary] = []
    all_fold_rows: List[Dict[str, object]] = []

    for display_name, mode, rc_average in modes:
        print(f"[benchmark] Computing features: {display_name}")
        X = np.vstack([descriptor(r.sequence, mode, cfg, rc_average=rc_average) for r in records])
        summary, fold_rows = evaluate_descriptor(
            display_name,
            X,
            y,
            groups,
            class_names=class_names,
            splits=splits,
            random_seed=seed,
        )
        results.append(summary)
        all_fold_rows.extend(fold_rows)
        print(
            f"[benchmark] {display_name}: "
            f"accuracy={summary.accuracy_mean:.3f}, "
            f"balanced_accuracy={summary.balanced_accuracy_mean:.3f}, "
            f"macro_f1={summary.macro_f1_mean:.3f}, "
            f"features={summary.n_features}"
        )

    write_results(results)
    write_fold_predictions(all_fold_rows)
    generate_benchmark_results_table(results)

    return results


def save_config(cfg: DescriptorConfig, args: argparse.Namespace) -> None:
    payload = {
        "descriptor_config": asdict(cfg),
        "arguments": vars(args),
        "has_stratified_group_kfold": HAS_STRATIFIED_GROUP_KFOLD,
        "python": sys.version,
        "numpy": np.__version__,
    }
    Path("outputs/run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accession-fixed Walsh--Haar FCGR benchmark.")
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL", ""), help="NCBI Entrez email.")
    parser.add_argument("--download", action="store_true", help="Download GenBank records into cache.")
    parser.add_argument("--offline", action="store_true", help="Use existing cached GenBank records only.")
    parser.add_argument("--run-benchmark", action="store_true", help="Run accession-fixed benchmark.")
    parser.add_argument("--self-test", action="store_true", help="Run mathematical implementation self-tests.")
    parser.add_argument("--runtime", action="store_true", help="Measure feature extraction runtime.")
    parser.add_argument("--delay", type=float, default=0.34, help="Delay between NCBI requests.")
    parser.add_argument("--folds", type=int, default=5, help="Maximum number of group-aware folds.")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeated CV runs.")
    parser.add_argument("--seed", type=int, default=20260619, help="Random seed.")
    parser.add_argument("--smooth-c", type=float, default=0.025, help="Finite toric smoothing parameter c.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Entropy regularization alpha.")
    parser.add_argument("--robustness", action="store_true", help="Generate synthetic robustness tables.")
    parser.add_argument("--quadrature-test", action="store_true", help="Generate polar quadrature convergence table.")
    parser.add_argument("--sensitivity-k", action="store_true", help="Generate scale sensitivity table.")
    parser.add_argument("--finite-riesz", action="store_true", help="Generate finite smoothing diagnostics table.")
    parser.add_argument("--sequencing-robustness", action="store_true", help="Generate sequencing-like robustness table.")
    parser.add_argument("--sensitivity-c-alpha", action="store_true", help="Generate c/alpha sensitivity table.")
    parser.add_argument("--figures", action="store_true", help="Generate descriptor embedding figure when records are available.")
    parser.add_argument("--all-diagnostics", action="store_true", help="Generate all diagnostics that do not require network access; benchmark-dependent tables are generated after --run-benchmark.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    cfg = DescriptorConfig(
        alpha_entropy=args.alpha,
        smooth_c=args.smooth_c,
    )

    save_config(cfg, args)
    generate_manifest_accessions_table()

    if args.all_diagnostics:
        args.finite_riesz = True
        args.robustness = True
        args.sequencing_robustness = True
        args.quadrature_test = True
        args.sensitivity_k = True
        args.sensitivity_c_alpha = True

    if args.self_test:
        self_test(cfg)

    if args.finite_riesz:
        generate_finite_riesz_table(cfg)

    if args.robustness:
        generate_robustness_tables(cfg, seed=args.seed)

    if args.sequencing_robustness:
        generate_sequencing_robustness_table(cfg, seed=args.seed)

    if args.quadrature_test:
        generate_polar_quadrature_table(cfg, seed=args.seed)

    if args.sensitivity_k:
        generate_sensitivity_k_table(seed=args.seed)

    if args.sensitivity_c_alpha:
        generate_sensitivity_c_alpha_table(seed=args.seed)

    if not args.download and not args.offline and (args.run_benchmark or args.runtime):
        print("[info] Neither --download nor --offline was specified. Assuming --offline.")
        args.offline = True

    records: List[RecordInfo] = []
    if args.download or args.offline or args.run_benchmark or args.runtime:
        records = load_records(
            email=args.email,
            download=args.download,
            offline=args.offline,
            delay=args.delay,
        )
        write_manifest(records)

    if args.run_benchmark:
        results = run_benchmark(
            records,
            cfg=cfg,
            repeats=args.repeats,
            folds=args.folds,
            seed=args.seed,
        )
        generate_benchmark_results_table(results)
        generate_method_comparison_table(results)
        generate_block_importance_table(records, cfg)
        if args.figures:
            generate_embedding_figure(records, cfg)

    if args.runtime:
        runtime_rows = measure_runtime(records, cfg=cfg)
        generate_runtime_table(runtime_rows)

    print("[done] Outputs written to outputs/ and generated_tables/")


if __name__ == "__main__":
    main()
