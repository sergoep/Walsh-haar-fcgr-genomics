"""Orthogonal two-dimensional Walsh--Hadamard and Haar transforms."""

from __future__ import annotations

import math

import numpy as np


def _require_power_of_two(n: int) -> None:
    if n <= 0 or (n & (n - 1)):
        raise ValueError(f"Expected a positive power of two, received {n}")


def fwht1(vector: np.ndarray) -> np.ndarray:
    """Return the orthonormal fast Walsh--Hadamard transform of a vector."""

    x = np.asarray(vector, dtype=np.float64).copy()
    if x.ndim != 1:
        raise ValueError("fwht1 expects a one-dimensional array")
    n = x.size
    _require_power_of_two(n)
    step = 1
    while step < n:
        for start in range(0, n, 2 * step):
            left = x[start : start + step].copy()
            right = x[start + step : start + 2 * step].copy()
            x[start : start + step] = left + right
            x[start + step : start + 2 * step] = left - right
        step *= 2
    return x / math.sqrt(n)


def fwht2(matrix: np.ndarray) -> np.ndarray:
    """Apply the orthonormal Walsh--Hadamard transform along both axes."""

    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("fwht2 expects a square matrix")
    _require_power_of_two(a.shape[0])
    rows = np.vstack([fwht1(row) for row in a])
    return np.vstack([fwht1(col) for col in rows.T]).T


def haar_matrix(n: int) -> np.ndarray:
    """Return an orthonormal Haar matrix of dyadic order ``n``."""

    _require_power_of_two(n)
    if n == 1:
        return np.ones((1, 1), dtype=np.float64)
    half = n // 2
    h_half = haar_matrix(half)
    top = np.kron(h_half, np.array([[1.0, 1.0]]) / math.sqrt(2.0))
    bottom = np.kron(np.eye(half), np.array([[1.0, -1.0]]) / math.sqrt(2.0))
    return np.vstack([top, bottom])


def haar2(matrix: np.ndarray) -> np.ndarray:
    """Apply an orthonormal two-dimensional Haar transform."""

    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("haar2 expects a square matrix")
    h = haar_matrix(a.shape[0])
    return h @ a @ h.T


def block_energy(matrix: np.ndarray, block: tuple[slice, slice]) -> float:
    """Return the squared Frobenius energy inside a fixed coordinate block."""

    part = np.asarray(matrix, dtype=np.float64)[block]
    return float(np.sum(part * part))
