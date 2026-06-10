
# ============================================================
# Walsh--Haar FCGR Genomics: Colab-ready single-cell script
# Reproducibility and mathematical verification companion
# ============================================================

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import textwrap
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------
# Global settings
# ------------------------------------------------------------

OUTPUT_DIR = Path("walsh_haar_fcgr_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DNA_ALPHABET = ("A", "C", "G", "T")
DNA_SET = set(DNA_ALPHABET)
GAMMA = {
    "A": (0, 0),
    "C": (0, 1),
    "G": (1, 1),
    "T": (1, 0),
}
COMPLEMENT = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path | str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def assert_power_of_two(n: int) -> None:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Expected a positive power of two, got {n}")


def vector_l1(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(x, dtype=float))))


def frobenius_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float), ord="fro"))


def safe_isclose(a: float, b: float, atol: float = 1e-10, rtol: float = 1e-10) -> bool:
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=rtol))


# ------------------------------------------------------------
# FASTA and sequence processing
# ------------------------------------------------------------

@dataclass(frozen=True)
class FastaRecord:
    identifier: str
    sequence: str


@dataclass(frozen=True)
class FCGRQualityControl:
    sequence_length: int
    k: int
    total_windows: int
    valid_windows: int
    invalid_windows: int
    valid_window_fraction: float


def normalize_sequence(sequence: str) -> str:
    """
    Uppercase the sequence and remove whitespace only.
    IUPAC symbols such as N are retained so that windows containing them
    can be masked without creating artificial adjacencies.
    """
    return "".join(sequence.split()).upper()


def parse_fasta(text: str) -> List[FastaRecord]:
    records: List[FastaRecord] = []
    current_id: str | None = None
    current_parts: List[str] = []
    seen: set[str] = set()

    def flush() -> None:
        nonlocal current_id, current_parts
        if current_id is None:
            return
        seq = normalize_sequence("".join(current_parts))
        if not seq:
            raise ValueError(f"FASTA record {current_id!r} has an empty sequence")
        if current_id in seen:
            raise ValueError(f"Duplicate FASTA identifier: {current_id}")
        seen.add(current_id)
        records.append(FastaRecord(current_id, seq))
        current_id = None
        current_parts = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            identifier = line[1:].strip().split()[0]
            if not identifier:
                raise ValueError("Encountered an empty FASTA identifier")
            current_id = identifier
            current_parts = []
        else:
            if current_id is None:
                raise ValueError("FASTA sequence data encountered before the first header")
            current_parts.append(line)

    flush()
    if not records:
        raise ValueError("No FASTA records found")
    return records


def reverse_complement(sequence: str) -> str:
    sequence = normalize_sequence(sequence)
    try:
        return "".join(COMPLEMENT[ch] for ch in reversed(sequence))
    except KeyError as exc:
        raise ValueError(
            "reverse_complement requires a sequence over A, C, G, T only"
        ) from exc


def kmer_position(word: str) -> Tuple[int, int]:
    row = 0
    col = 0
    for char in word:
        try:
            a, b = GAMMA[char]
        except KeyError as exc:
            raise ValueError(f"Invalid DNA symbol in k-mer: {char!r}") from exc
        row = (row << 1) | a
        col = (col << 1) | b
    return row, col


def fcgr_matrix(
    sequence: str,
    k: int,
    min_valid_window_fraction: float = 0.0,
) -> Tuple[np.ndarray, FCGRQualityControl]:
    """
    Return the normalized FCGR matrix and quality-control metadata.

    Windows containing symbols outside A, C, G, T are masked. Such symbols
    are not deleted; therefore no artificial adjacency is introduced.
    """
    sequence = normalize_sequence(sequence)
    if k < 1:
        raise ValueError("k must be positive")
    total_windows = max(0, len(sequence) - k + 1)
    if total_windows == 0:
        raise ValueError(f"Sequence length {len(sequence)} is smaller than k={k}")

    n = 1 << k
    counts = np.zeros((n, n), dtype=float)
    valid_windows = 0

    for start in range(total_windows):
        word = sequence[start : start + k]
        if all(char in DNA_SET for char in word):
            row, col = kmer_position(word)
            counts[row, col] += 1.0
            valid_windows += 1

    invalid_windows = total_windows - valid_windows
    valid_fraction = valid_windows / total_windows
    qc = FCGRQualityControl(
        sequence_length=len(sequence),
        k=k,
        total_windows=total_windows,
        valid_windows=valid_windows,
        invalid_windows=invalid_windows,
        valid_window_fraction=valid_fraction,
    )

    if valid_windows == 0:
        raise ValueError(f"No valid ACGT windows remain for k={k}")
    if valid_fraction < min_valid_window_fraction:
        raise ValueError(
            f"Insufficient valid windows for k={k}: "
            f"fraction={valid_fraction:.4f} < {min_valid_window_fraction:.4f}"
        )

    return counts / valid_windows, qc


def bit_reverse(value: int, width: int) -> int:
    result = 0
    for _ in range(width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def rc_permute_fcgr(matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Apply the FCGR cell permutation corresponding to reverse complementation:
        (r, s) -> (2^k - 1 - rev_k(r), rev_k(s)).
    """
    matrix = np.asarray(matrix, dtype=float)
    n = 1 << k
    if matrix.shape != (n, n):
        raise ValueError(f"Expected shape {(n, n)}, got {matrix.shape}")

    transformed = np.zeros_like(matrix)
    for row in range(n):
        for col in range(n):
            new_row = n - 1 - bit_reverse(row, k)
            new_col = bit_reverse(col, k)
            transformed[new_row, new_col] = matrix[row, col]
    return transformed


# ------------------------------------------------------------
# Orthogonal transforms
# ------------------------------------------------------------

def fwht_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    n = values.size
    assert_power_of_two(n)
    width = 1
    while width < n:
        for start in range(0, n, 2 * width):
            left = values[start : start + width].copy()
            right = values[start + width : start + 2 * width].copy()
            values[start : start + width] = left + right
            values[start + width : start + 2 * width] = left - right
        width *= 2
    return values / math.sqrt(n)


def fwht_2d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("fwht_2d expects a square matrix")
    n = matrix.shape[0]
    assert_power_of_two(n)

    transformed = np.apply_along_axis(fwht_1d, 1, matrix)
    transformed = np.apply_along_axis(fwht_1d, 0, transformed)
    return transformed


def haar_matrix(n: int) -> np.ndarray:
    """
    Construct an orthogonal Haar matrix recursively.
    The first row is the normalized scaling vector.
    """
    assert_power_of_two(n)
    if n == 1:
        return np.ones((1, 1), dtype=float)

    previous = haar_matrix(n // 2)
    top = np.kron(previous, np.array([1.0, 1.0])) / math.sqrt(2.0)
    bottom = np.kron(
        np.eye(n // 2, dtype=float),
        np.array([1.0, -1.0]),
    ) / math.sqrt(2.0)
    return np.vstack([top, bottom])


def haar_2d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("haar_2d expects a square matrix")
    n = matrix.shape[0]
    q = haar_matrix(n)
    return q @ matrix @ q.T


# ------------------------------------------------------------
# Spectral features and entropy
# ------------------------------------------------------------

def rectangular_block_energy(
    spectrum: np.ndarray,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> float:
    spectrum = np.asarray(spectrum, dtype=float)
    block = spectrum[row_start:row_stop, col_start:col_stop]
    return float(np.sum(block * block))


def default_block_energies(spectrum: np.ndarray) -> Dict[str, float]:
    spectrum = np.asarray(spectrum, dtype=float)
    n = spectrum.shape[0]
    if spectrum.shape != (n, n):
        raise ValueError("Expected a square spectrum")

    quarter = max(1, n // 4)
    half = max(1, n // 2)

    return {
        "dc": float(spectrum[0, 0] ** 2),
        "low_square": rectangular_block_energy(spectrum, 0, quarter, 0, quarter),
        "mid_square": rectangular_block_energy(spectrum, 0, half, 0, half),
        "total": float(np.sum(spectrum * spectrum)),
    }


def regularized_probability(vector: np.ndarray, delta: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).ravel()
    if delta <= 0:
        raise ValueError("delta must be positive")
    d = vector.size
    squares = vector * vector
    return (squares + delta) / (float(np.sum(squares)) + d * delta)


def regularized_entropy(vector: np.ndarray, delta: float) -> float:
    probabilities = regularized_probability(vector, delta)
    return float(-np.sum(probabilities * np.log(probabilities)))


def entropy_lipschitz_constant(delta: float, d: int) -> float:
    if delta <= 0:
        raise ValueError("delta must be positive")
    if d < 2:
        raise ValueError("d must be at least 2")
    return (4.0 / (d * delta)) * (
        1.0 + abs(math.log(delta / (1.0 + d * delta)))
    )


# ------------------------------------------------------------
# Torically smoothed Walsh functions
# ------------------------------------------------------------

def walsh_paley(index: int, x: np.ndarray | float) -> np.ndarray:
    """
    Paley-ordered Walsh function on the torus, represented by its
    right-continuous periodic version.
    """
    if index < 0:
        raise ValueError("Walsh index must be nonnegative")

    x_array = np.asarray(x, dtype=float)
    toric_x = np.mod(x_array, 1.0)

    if index == 0:
        return np.ones_like(toric_x, dtype=float)

    parity = np.zeros_like(toric_x, dtype=np.int8)
    for bit_index in range(index.bit_length()):
        if (index >> bit_index) & 1:
            dyadic_digit = np.floor(toric_x * (1 << (bit_index + 1))).astype(np.int64) & 1
            parity ^= dyadic_digit.astype(np.int8)
    return np.where(parity == 0, 1.0, -1.0)


def walsh_jumps_torus(index: int) -> np.ndarray:
    """
    Return all toric jump locations, including 0~1 when applicable.
    """
    if index < 0:
        raise ValueError("Walsh index must be nonnegative")
    if index == 0:
        return np.array([], dtype=float)

    level = index.bit_length()
    grid_size = 1 << level
    midpoints = (np.arange(grid_size, dtype=float) + 0.5) / grid_size
    values = walsh_paley(index, midpoints)

    jumps: List[float] = []
    for j in range(grid_size):
        left = values[(j - 1) % grid_size]
        right = values[j]
        if left != right:
            jumps.append(j / grid_size)
    return np.array(jumps, dtype=float)


def toric_signed_distance(x: np.ndarray | float, center: float) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    return np.mod(x_array - center + 0.5, 1.0) - 0.5


def _minimum_toric_spacing(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if points.size <= 1:
        return 1.0
    sorted_points = np.sort(np.mod(points, 1.0))
    gaps = np.diff(np.concatenate([sorted_points, sorted_points[:1] + 1.0]))
    return float(np.min(gaps))


def smoothed_walsh(index: int, x: np.ndarray | float, epsilon: float) -> np.ndarray:
    """
    Continuous toric smoothing by linear ramps around every Walsh jump.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    x_array = np.asarray(x, dtype=float)
    result = walsh_paley(index, x_array).astype(float)
    jumps = walsh_jumps_torus(index)
    if jumps.size == 0:
        return result

    min_spacing = _minimum_toric_spacing(jumps)
    if 2.0 * epsilon >= min_spacing - 1e-15:
        raise ValueError(
            f"Smoothing arcs overlap for index={index}: "
            f"epsilon={epsilon}, minimum jump spacing={min_spacing}"
        )

    for jump in jumps:
        distance = toric_signed_distance(x_array, jump)
        mask = np.abs(distance) <= epsilon
        if not np.any(mask):
            continue

        left_probe = (jump - epsilon - 1e-12) % 1.0
        right_probe = (jump + epsilon + 1e-12) % 1.0
        left_value = float(walsh_paley(index, left_probe))
        right_value = float(walsh_paley(index, right_probe))

        ramp_parameter = (distance[mask] + epsilon) / (2.0 * epsilon)
        result[mask] = left_value + ramp_parameter * (right_value - left_value)

    return result


def exact_smoothing_l2_squared(index: int, epsilon: float) -> float:
    jumps = walsh_jumps_torus(index)
    if jumps.size == 0:
        return 0.0
    min_spacing = _minimum_toric_spacing(jumps)
    if 2.0 * epsilon >= min_spacing - 1e-15:
        raise ValueError("Smoothing arcs overlap")
    return (2.0 / 3.0) * len(jumps) * epsilon


def numerical_smoothing_l2_squared(
    index: int,
    epsilon: float,
    grid_size: int = 1 << 18,
) -> float:
    points = (np.arange(grid_size, dtype=float) + 0.5) / grid_size
    diff = smoothed_walsh(index, points, epsilon) - walsh_paley(index, points)
    return float(np.mean(diff * diff))


def smoothed_walsh_integral(index: int, epsilon: float, start: float, stop: float) -> float:
    """
    Integrate a torically smoothed Walsh function exactly up to floating-point
    arithmetic by partitioning [start, stop] into intervals on which the
    function is affine.
    """
    if not (0.0 <= start <= stop <= 1.0):
        raise ValueError("Expected 0 <= start <= stop <= 1")
    if start == stop:
        return 0.0

    jumps = walsh_jumps_torus(index)
    level = max(0, index.bit_length())
    grid_size = 1 << level if level > 0 else 1

    breakpoints = {float(start), float(stop)}
    for j in range(grid_size + 1):
        point = j / grid_size
        if start < point < stop:
            breakpoints.add(float(point))

    for jump in jumps:
        for edge in ((jump - epsilon) % 1.0, (jump + epsilon) % 1.0):
            if start < edge < stop:
                breakpoints.add(float(edge))
        if jump == 0.0:
            if start < epsilon < stop:
                breakpoints.add(float(epsilon))
            if start < 1.0 - epsilon < stop:
                breakpoints.add(float(1.0 - epsilon))

    ordered = sorted(breakpoints)
    integral = 0.0
    for left, right in zip(ordered[:-1], ordered[1:]):
        midpoint = 0.5 * (left + right)
        width = right - left
        value_left = float(smoothed_walsh(index, np.array([left]), epsilon)[0])
        value_mid = float(smoothed_walsh(index, np.array([midpoint]), epsilon)[0])
        value_right = float(smoothed_walsh(index, np.array([right]), epsilon)[0])

        # Simpson is exact for affine pieces; midpoint safeguards boundary values.
        integral += width * (value_left + 4.0 * value_mid + value_right) / 6.0

    return float(integral)


def smoothed_cell_averages(index: int, epsilon: float, n: int) -> np.ndarray:
    assert_power_of_two(n)
    averages = np.empty(n, dtype=float)
    for cell in range(n):
        start = cell / n
        stop = (cell + 1) / n
        averages[cell] = n * smoothed_walsh_integral(index, epsilon, start, stop)
    return averages


def smoothed_coefficient(
    matrix: np.ndarray,
    p: int,
    q: int,
    epsilon_p: float,
    epsilon_q: float,
) -> float:
    """
    Exact cell-average formula for the step density F[M].
    x is the column coordinate; y is the row coordinate.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("Expected a square FCGR matrix")
    omega_p = smoothed_cell_averages(p, epsilon_p, n)
    omega_q = smoothed_cell_averages(q, epsilon_q, n)
    return float(np.sum(matrix * np.outer(omega_q, omega_p)))


# ------------------------------------------------------------
# Polar dictionary
# ------------------------------------------------------------

@dataclass(frozen=True)
class PolarDisk:
    center_x: float
    center_y: float
    radius: float

    def validate(self) -> None:
        if self.radius <= 0:
            raise ValueError("Disk radius must be positive")
        if not (
            self.radius <= self.center_x <= 1.0 - self.radius
            and self.radius <= self.center_y <= 1.0 - self.radius
        ):
            raise ValueError(
                "The closed polar dictionary disk must be contained in [0, 1]^2"
            )


def step_density_values(matrix: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("Expected a square matrix")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x < 0.0) or np.any(x > 1.0) or np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError("Sampling points must lie in [0, 1]^2")

    # Use left-closed, right-open cells; clamp the rare endpoint x=1 or y=1.
    cols = np.minimum((x * n).astype(int), n - 1)
    rows = np.minimum((y * n).astype(int), n - 1)
    return (n * n) * matrix[rows, cols]


def polar_walsh_coefficient(
    matrix: np.ndarray,
    disk: PolarDisk,
    m: int,
    n: int,
    radial_nodes: int = 64,
    angular_nodes: int = 128,
) -> float:
    """
    Midpoint quadrature uniform in u=r^2 and v=theta/(2*pi).

    Under u=r^2 and v=theta/(2*pi), one has
        r dr dtheta = pi du dv.
    Therefore,
        c_{m,n}(G) = sqrt(pi) * integral_[0,1]^2 G(u,v) w_m(u) w_n(v) du dv.
    """
    disk.validate()
    if radial_nodes < 1 or angular_nodes < 1:
        raise ValueError("Quadrature sizes must be positive")

    u = (np.arange(radial_nodes, dtype=float) + 0.5) / radial_nodes
    v = (np.arange(angular_nodes, dtype=float) + 0.5) / angular_nodes
    uu, vv = np.meshgrid(u, v, indexing="ij")

    r = np.sqrt(uu)
    theta = 2.0 * math.pi * vv
    x = disk.center_x + disk.radius * r * np.cos(theta)
    y = disk.center_y + disk.radius * r * np.sin(theta)

    density = step_density_values(matrix, x, y)
    basis = walsh_paley(m, uu) * walsh_paley(n, vv)
    return float(math.sqrt(math.pi) * np.mean(density * basis))


def polar_block_energy(
    matrix: np.ndarray,
    disk: PolarDisk,
    m_indices: Sequence[int],
    n_indices: Sequence[int],
    radial_nodes: int = 64,
    angular_nodes: int = 128,
) -> float:
    coefficients = []
    for m in m_indices:
        for n in n_indices:
            coefficients.append(
                polar_walsh_coefficient(
                    matrix=matrix,
                    disk=disk,
                    m=m,
                    n=n,
                    radial_nodes=radial_nodes,
                    angular_nodes=angular_nodes,
                )
            )
    return float(np.sum(np.square(coefficients)))


# ------------------------------------------------------------
# Edit operations and Levenshtein distance
# ------------------------------------------------------------

def levenshtein_distance(source: str, target: str) -> int:
    source = normalize_sequence(source)
    target = normalize_sequence(target)

    if len(source) < len(target):
        source, target = target, source

    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (source_char != target_char)
            current.append(min(insert_cost, delete_cost, substitute_cost))
        previous = current
    return previous[-1]


def substitute_random_base(sequence: str, rng: random.Random) -> str:
    sequence = normalize_sequence(sequence)
    if not sequence:
        raise ValueError("Cannot substitute a base in an empty sequence")
    position = rng.randrange(len(sequence))
    old = sequence[position]
    choices = [base for base in DNA_ALPHABET if base != old]
    new = rng.choice(choices)
    return sequence[:position] + new + sequence[position + 1 :]


def insert_random_base(sequence: str, rng: random.Random) -> str:
    sequence = normalize_sequence(sequence)
    position = rng.randrange(len(sequence) + 1)
    base = rng.choice(DNA_ALPHABET)
    return sequence[:position] + base + sequence[position:]


def delete_random_base(sequence: str, rng: random.Random) -> str:
    sequence = normalize_sequence(sequence)
    if len(sequence) <= 1:
        raise ValueError("Deletion would produce an empty sequence")
    position = rng.randrange(len(sequence))
    return sequence[:position] + sequence[position + 1 :]


def apply_random_edits(
    sequence: str,
    number_of_edits: int,
    mode: str,
    seed: int = 12345,
) -> str:
    if number_of_edits < 0:
        raise ValueError("number_of_edits must be nonnegative")
    rng = random.Random(seed)
    result = normalize_sequence(sequence)

    for _ in range(number_of_edits):
        if mode == "substitution":
            result = substitute_random_base(result, rng)
        elif mode == "insertion":
            result = insert_random_base(result, rng)
        elif mode == "deletion":
            result = delete_random_base(result, rng)
        elif mode == "mixed":
            operation = rng.choice(("substitution", "insertion", "deletion"))
            if operation == "substitution":
                result = substitute_random_base(result, rng)
            elif operation == "insertion":
                result = insert_random_base(result, rng)
            else:
                result = delete_random_base(result, rng)
        else:
            raise ValueError(
                "mode must be substitution, insertion, deletion, or mixed"
            )
    return result


# ------------------------------------------------------------
# Theory-aligned descriptor
# ------------------------------------------------------------

@dataclass(frozen=True)
class DescriptorConfig:
    scales: Tuple[int, ...] = (3, 4)
    entropy_delta: float = 1e-6
    smoothing_c: float = 0.20
    smoothed_indices: Tuple[int, ...] = (0, 1, 2, 3)
    polar_disks: Tuple[PolarDisk, ...] = (
        PolarDisk(0.50, 0.50, 0.20),
        PolarDisk(0.30, 0.30, 0.12),
        PolarDisk(0.70, 0.70, 0.12),
    )
    polar_m_indices: Tuple[int, ...] = (0, 1)
    polar_n_indices: Tuple[int, ...] = (0, 1)
    radial_nodes: int = 32
    angular_nodes: int = 64
    min_valid_window_fraction: float = 0.0


def validate_descriptor_config(config: DescriptorConfig) -> None:
    if not config.scales:
        raise ValueError("At least one FCGR scale is required")
    if any(k < 1 for k in config.scales):
        raise ValueError("All scales must be positive")
    if config.entropy_delta <= 0:
        raise ValueError("entropy_delta must be positive")
    if not (0.0 < config.smoothing_c < 0.5):
        raise ValueError("smoothing_c must satisfy 0 < c < 1/2")
    for disk in config.polar_disks:
        disk.validate()


def theory_aligned_descriptor(
    sequence: str,
    config: DescriptorConfig = DescriptorConfig(),
) -> Tuple[np.ndarray, List[str], List[dict]]:
    validate_descriptor_config(config)

    features: List[float] = []
    labels: List[str] = []
    qc_rows: List[dict] = []

    for k in config.scales:
        matrix, qc = fcgr_matrix(
            sequence,
            k=k,
            min_valid_window_fraction=config.min_valid_window_fraction,
        )
        qc_rows.append(asdict(qc))

        wh = fwht_2d(matrix)
        haar = haar_2d(matrix)

        for transform_name, spectrum in (("wh", wh), ("haar", haar)):
            energies = default_block_energies(spectrum)
            for energy_name, value in energies.items():
                labels.append(f"k{k}_{transform_name}_{energy_name}")
                features.append(value)

            labels.append(f"k{k}_{transform_name}_entropy")
            features.append(
                regularized_entropy(
                    spectrum.ravel(),
                    delta=config.entropy_delta,
                )
            )

        epsilon = config.smoothing_c * (2.0 ** (-k))
        for p in config.smoothed_indices:
            for q in config.smoothed_indices:
                labels.append(f"k{k}_smooth_p{p}_q{q}")
                features.append(
                    smoothed_coefficient(
                        matrix,
                        p=p,
                        q=q,
                        epsilon_p=epsilon,
                        epsilon_q=epsilon,
                    )
                )

        for disk_index, disk in enumerate(config.polar_disks):
            labels.append(f"k{k}_polar_disk{disk_index}_energy")
            features.append(
                polar_block_energy(
                    matrix,
                    disk=disk,
                    m_indices=config.polar_m_indices,
                    n_indices=config.polar_n_indices,
                    radial_nodes=config.radial_nodes,
                    angular_nodes=config.angular_nodes,
                )
            )

    return np.asarray(features, dtype=float), labels, qc_rows


def rc_averaged_descriptor(
    sequence: str,
    config: DescriptorConfig = DescriptorConfig(),
) -> Tuple[np.ndarray, List[str], List[dict]]:
    sequence = normalize_sequence(sequence)
    if any(char not in DNA_SET for char in sequence):
        raise ValueError(
            "RC averaging in this compact demo requires a pure ACGT sequence"
        )

    descriptor, labels, qc_rows = theory_aligned_descriptor(sequence, config)
    rc_descriptor, rc_labels, rc_qc_rows = theory_aligned_descriptor(
        reverse_complement(sequence),
        config,
    )
    if labels != rc_labels:
        raise RuntimeError("Descriptor labels are inconsistent")
    return 0.5 * (descriptor + rc_descriptor), labels, qc_rows + rc_qc_rows


# ------------------------------------------------------------
# Verification suite
# ------------------------------------------------------------

def synthetic_sequence(length: int = 600, seed: int = 20260610) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(DNA_ALPHABET) for _ in range(length))


def fcgr_stability_bound(source: str, target: str, k: int) -> float:
    e = levenshtein_distance(source, target)
    n_min = min(len(source) - k + 1, len(target) - k + 1)
    if n_min <= 0:
        raise ValueError("Sequences must be at least as long as k")
    return ((2 * k + 1) * e) / n_min


def run_verification_suite(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    source = synthetic_sequence(length=640, seed=20260610)
    target = apply_random_edits(source, number_of_edits=5, mode="mixed", seed=17)
    k = 4

    matrix, qc = fcgr_matrix(source, k)
    target_matrix, target_qc = fcgr_matrix(target, k)
    wh = fwht_2d(matrix)
    haar = haar_2d(matrix)

    rc_matrix, _ = fcgr_matrix(reverse_complement(source), k)
    rc_expected = rc_permute_fcgr(matrix, k)

    edit_distance = levenshtein_distance(source, target)
    observed_l1 = vector_l1(matrix - target_matrix)
    observed_frobenius = frobenius_norm(matrix - target_matrix)
    theoretical_bound = fcgr_stability_bound(source, target, k)

    smoothing_index = 5
    smoothing_epsilon = 0.005
    smoothing_theoretical = exact_smoothing_l2_squared(
        smoothing_index,
        smoothing_epsilon,
    )
    smoothing_numerical = numerical_smoothing_l2_squared(
        smoothing_index,
        smoothing_epsilon,
        grid_size=1 << 18,
    )

    n = matrix.shape[0]
    epsilon = 0.20 * (2.0 ** (-k))
    coefficient = smoothed_coefficient(
        matrix,
        p=1,
        q=3,
        epsilon_p=epsilon,
        epsilon_q=epsilon,
    )

    constant_matrix = np.full((16, 16), 1.0 / (16 * 16), dtype=float)
    disk = PolarDisk(0.50, 0.50, 0.20)
    polar_dc = polar_walsh_coefficient(
        constant_matrix,
        disk=disk,
        m=0,
        n=0,
        radial_nodes=64,
        angular_nodes=128,
    )
    polar_non_dc = polar_walsh_coefficient(
        constant_matrix,
        disk=disk,
        m=1,
        n=0,
        radial_nodes=64,
        angular_nodes=128,
    )

    rng = np.random.default_rng(123)
    x = rng.normal(size=16)
    y = rng.normal(size=16)
    x /= max(1.0, float(np.linalg.norm(x)))
    y /= max(1.0, float(np.linalg.norm(y)))
    delta = 1e-3
    entropy_difference = abs(
        regularized_entropy(x, delta) - regularized_entropy(y, delta)
    )
    entropy_bound = entropy_lipschitz_constant(delta, d=x.size) * float(
        np.linalg.norm(x - y)
    )

    checks = {
        "fcgr_normalization": safe_isclose(float(np.sum(matrix)), 1.0),
        "fcgr_frobenius_at_most_one": frobenius_norm(matrix) <= 1.0 + 1e-12,
        "fwht_energy_preservation": safe_isclose(
            frobenius_norm(wh),
            frobenius_norm(matrix),
            atol=1e-12,
            rtol=1e-12,
        ),
        "haar_energy_preservation": safe_isclose(
            frobenius_norm(haar),
            frobenius_norm(matrix),
            atol=1e-12,
            rtol=1e-12,
        ),
        "reverse_complement_equivariance": bool(
            np.allclose(rc_matrix, rc_expected, atol=1e-12, rtol=1e-12)
        ),
        "fcgr_l1_edit_bound": observed_l1 <= theoretical_bound + 1e-12,
        "fcgr_frobenius_edit_bound": observed_frobenius <= theoretical_bound + 1e-12,
        "toric_smoothing_l2_identity": safe_isclose(
            smoothing_numerical,
            smoothing_theoretical,
            atol=2e-5,
            rtol=2e-5,
        ),
        "polar_constant_density_dc": safe_isclose(
            polar_dc,
            math.sqrt(math.pi),
            atol=1e-12,
            rtol=1e-12,
        ),
        "polar_constant_density_non_dc": abs(polar_non_dc) <= 1e-12,
        "entropy_lipschitz_bound": entropy_difference <= entropy_bound + 1e-12,
    }

    report = {
        "all_checks_passed": bool(all(checks.values())),
        "checks": checks,
        "details": {
            "source_length": len(source),
            "target_length": len(target),
            "edit_distance": edit_distance,
            "fcgr_l1_observed": observed_l1,
            "fcgr_frobenius_observed": observed_frobenius,
            "fcgr_theoretical_bound": theoretical_bound,
            "fcgr_norm": frobenius_norm(matrix),
            "fwht_norm": frobenius_norm(wh),
            "haar_norm": frobenius_norm(haar),
            "smoothing_theoretical_l2_squared": smoothing_theoretical,
            "smoothing_numerical_l2_squared": smoothing_numerical,
            "example_smoothed_coefficient": coefficient,
            "polar_dc_for_constant_density": polar_dc,
            "polar_expected_dc": math.sqrt(math.pi),
            "polar_non_dc_for_constant_density": polar_non_dc,
            "entropy_difference": entropy_difference,
            "entropy_lipschitz_upper_bound": entropy_bound,
            "source_qc": asdict(qc),
            "target_qc": asdict(target_qc),
        },
    }

    write_json(output_dir / "verification_report.json", report)
    return report


# ------------------------------------------------------------
# Demonstration
# ------------------------------------------------------------

def run_synthetic_demo(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence = synthetic_sequence(length=800, seed=314159)
    mutated = apply_random_edits(
        sequence,
        number_of_edits=7,
        mode="mixed",
        seed=271828,
    )

    config = DescriptorConfig(
        scales=(3, 4),
        entropy_delta=1e-6,
        smoothing_c=0.20,
        smoothed_indices=(0, 1, 2, 3),
        polar_disks=(
            PolarDisk(0.50, 0.50, 0.20),
            PolarDisk(0.30, 0.30, 0.12),
            PolarDisk(0.70, 0.70, 0.12),
        ),
        polar_m_indices=(0, 1),
        polar_n_indices=(0, 1),
        radial_nodes=32,
        angular_nodes=64,
    )

    descriptor, labels, qc = theory_aligned_descriptor(sequence, config)
    mutated_descriptor, mutated_labels, mutated_qc = theory_aligned_descriptor(
        mutated,
        config,
    )
    if labels != mutated_labels:
        raise RuntimeError("Descriptor labels differ unexpectedly")

    report = {
        "sequence_length": len(sequence),
        "mutated_sequence_length": len(mutated),
        "edit_distance": levenshtein_distance(sequence, mutated),
        "descriptor_dimension": int(descriptor.size),
        "descriptor_l2_norm": float(np.linalg.norm(descriptor)),
        "mutated_descriptor_l2_norm": float(np.linalg.norm(mutated_descriptor)),
        "descriptor_difference_l2": float(np.linalg.norm(descriptor - mutated_descriptor)),
        "config": {
            "scales": list(config.scales),
            "entropy_delta": config.entropy_delta,
            "smoothing_c": config.smoothing_c,
            "smoothed_indices": list(config.smoothed_indices),
            "polar_disks": [asdict(disk) for disk in config.polar_disks],
            "polar_m_indices": list(config.polar_m_indices),
            "polar_n_indices": list(config.polar_n_indices),
            "radial_nodes": config.radial_nodes,
            "angular_nodes": config.angular_nodes,
        },
        "feature_labels": labels,
        "descriptor": descriptor.tolist(),
        "mutated_descriptor": mutated_descriptor.tolist(),
        "source_qc": qc,
        "mutated_qc": mutated_qc,
    }

    write_json(output_dir / "demo_report.json", report)
    return report


# ------------------------------------------------------------
# Optional FASTA analysis in Google Colab
# ------------------------------------------------------------

def analyze_fasta_text(
    fasta_text: str,
    output_dir: Path = OUTPUT_DIR,
    config: DescriptorConfig = DescriptorConfig(),
) -> dict:
    records = parse_fasta(fasta_text)
    results = []

    for record in records:
        descriptor, labels, qc = theory_aligned_descriptor(record.sequence, config)
        results.append(
            {
                "identifier": record.identifier,
                "sequence_length": len(record.sequence),
                "descriptor_dimension": int(descriptor.size),
                "feature_labels": labels,
                "descriptor": descriptor.tolist(),
                "quality_control": qc,
            }
        )

    payload = {
        "number_of_records": len(results),
        "records": results,
    }
    write_json(output_dir / "fasta_analysis.json", payload)
    return payload


def colab_upload_and_analyze_fasta(
    output_dir: Path = OUTPUT_DIR,
    config: DescriptorConfig = DescriptorConfig(),
) -> dict:
    """
    Optional helper for Google Colab.
    Run manually after the main cell:
        result = colab_upload_and_analyze_fasta()
    """
    try:
        from google.colab import files
    except ImportError as exc:
        raise RuntimeError("This helper is intended for Google Colab") from exc

    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Please upload exactly one FASTA file")
    filename, content = next(iter(uploaded.items()))
    text = content.decode("utf-8")
    print(f"Analyzing FASTA file: {filename}")
    return analyze_fasta_text(text, output_dir=output_dir, config=config)


# ------------------------------------------------------------
# Manifest and archive
# ------------------------------------------------------------

def write_run_manifest(output_dir: Path = OUTPUT_DIR) -> dict:
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "working_directory": os.getcwd(),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def zip_output_directory(
    output_dir: Path = OUTPUT_DIR,
    archive_name: str = "walsh_haar_fcgr_colab_results.zip",
) -> Path:
    archive_path = Path(archive_name)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(output_dir.parent))
    return archive_path


# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------

print("=" * 72)
print("Walsh--Haar FCGR Genomics: Colab verification script")
print("=" * 72)

manifest = write_run_manifest()
verification = run_verification_suite()
demo = run_synthetic_demo()
archive = zip_output_directory()

print("\nVerification checks:")
for name, passed in verification["checks"].items():
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

print("\nSummary:")
print(f"  all_checks_passed       = {verification['all_checks_passed']}")
print(f"  demo descriptor size    = {demo['descriptor_dimension']}")
print(f"  demo edit distance      = {demo['edit_distance']}")
print(f"  descriptor L2 change    = {demo['descriptor_difference_l2']:.12f}")
print(f"  output directory        = {OUTPUT_DIR.resolve()}")
print(f"  ZIP archive             = {archive.resolve()}")

if not verification["all_checks_passed"]:
    raise RuntimeError("At least one verification check failed")

print("\nAll mathematical verification checks passed.")
print("\nOptional FASTA analysis:")
print("  result = colab_upload_and_analyze_fasta()")
print("\nOptional download of generated ZIP in Colab:")
print("  from google.colab import files")
print("  files.download(str(archive))")
