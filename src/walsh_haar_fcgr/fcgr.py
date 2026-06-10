"""FCGR construction, masking rules, and reverse-complement equivariance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

ALPHABET = frozenset("ACGT")
COMPLEMENT = str.maketrans({"A": "T", "T": "A", "C": "G", "G": "C"})
ENCODING: dict[str, tuple[int, int]] = {
    "A": (0, 0),
    "C": (0, 1),
    "G": (1, 1),
    "T": (1, 0),
}


@dataclass(frozen=True)
class FCGRResult:
    """Normalized FCGR matrix and quality-control metadata."""

    matrix: np.ndarray
    k: int
    sequence_length: int
    total_windows: int
    valid_windows: int
    masked_windows: int

    @property
    def valid_fraction(self) -> float:
        return self.valid_windows / self.total_windows if self.total_windows else 0.0


def normalize_sequence(sequence: str) -> str:
    """Upper-case a sequence and retain symbols for window-level masking."""

    return "".join(sequence.upper().split())


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence.

    Non-ACGT symbols are preserved by ``str.translate`` and reversed. They remain
    masked by :func:`fcgr_matrix`.
    """

    return normalize_sequence(sequence).translate(COMPLEMENT)[::-1]


def kmer_to_cell(kmer: str) -> tuple[int, int]:
    """Map an ACGT k-mer to its FCGR row and column indices."""

    row = 0
    col = 0
    for symbol in kmer:
        try:
            a, b = ENCODING[symbol]
        except KeyError as exc:
            raise ValueError(f"Invalid DNA symbol in k-mer: {symbol!r}") from exc
        row = (row << 1) | a
        col = (col << 1) | b
    return row, col


def reverse_bits(value: int, width: int) -> int:
    """Reverse exactly ``width`` binary digits."""

    out = 0
    for _ in range(width):
        out = (out << 1) | (value & 1)
        value >>= 1
    return out


def rc_cell(row: int, col: int, k: int) -> tuple[int, int]:
    """Return the FCGR cell corresponding to reverse complementation."""

    max_index = (1 << k) - 1
    return max_index - reverse_bits(row, k), reverse_bits(col, k)


def iter_valid_kmers(sequence: str, k: int) -> Iterable[str]:
    """Yield only windows containing A, C, G, and T."""

    sequence = normalize_sequence(sequence)
    if k <= 0:
        raise ValueError("k must be positive")
    for start in range(max(0, len(sequence) - k + 1)):
        kmer = sequence[start : start + k]
        if all(base in ALPHABET for base in kmer):
            yield kmer


def fcgr_matrix(
    sequence: str,
    k: int,
    *,
    min_valid_fraction: float = 0.0,
) -> FCGRResult:
    """Build a normalized dense FCGR matrix.

    Windows containing symbols outside ``ACGT`` are masked rather than repaired
    or shortened. This avoids artificial adjacencies.
    """

    sequence = normalize_sequence(sequence)
    if k <= 0:
        raise ValueError("k must be positive")
    total_windows = max(0, len(sequence) - k + 1)
    if total_windows == 0:
        raise ValueError(f"Sequence length {len(sequence)} is smaller than k={k}")

    size = 1 << k
    counts = np.zeros((size, size), dtype=np.float64)
    valid_windows = 0
    for kmer in iter_valid_kmers(sequence, k):
        row, col = kmer_to_cell(kmer)
        counts[row, col] += 1.0
        valid_windows += 1

    if valid_windows == 0:
        raise ValueError(f"No valid ACGT windows remain for k={k}")
    valid_fraction = valid_windows / total_windows
    if valid_fraction < min_valid_fraction:
        raise ValueError(
            f"Insufficient valid windows for k={k}: fraction={valid_fraction:.4f}, "
            f"required={min_valid_fraction:.4f}"
        )
    counts /= valid_windows
    return FCGRResult(
        matrix=counts,
        k=k,
        sequence_length=len(sequence),
        total_windows=total_windows,
        valid_windows=valid_windows,
        masked_windows=total_windows - valid_windows,
    )


def verify_rc_equivariance(sequence: str, k: int, *, atol: float = 1e-12) -> bool:
    """Numerically verify FCGR reverse-complement equivariance."""

    original = fcgr_matrix(sequence, k).matrix
    rc = fcgr_matrix(reverse_complement(sequence), k).matrix
    transformed = np.zeros_like(original)
    for row in range(original.shape[0]):
        for col in range(original.shape[1]):
            rr, cc = rc_cell(row, col, k)
            transformed[rr, cc] = original[row, col]
    return bool(np.allclose(rc, transformed, atol=atol, rtol=0.0))
