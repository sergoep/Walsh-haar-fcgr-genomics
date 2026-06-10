"""Regularized spectral entropy and explicit Lipschitz constants."""

from __future__ import annotations

import math

import numpy as np


def regularized_probabilities(vector: np.ndarray, delta: float) -> np.ndarray:
    """Return regularized spectral probabilities."""

    if delta <= 0:
        raise ValueError("delta must be positive")
    x = np.asarray(vector, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError("The entropy vector must contain at least two coordinates")
    squares = x * x
    return (squares + delta) / (float(np.sum(squares)) + x.size * delta)


def regularized_entropy(vector: np.ndarray, delta: float) -> float:
    """Compute Shannon entropy of regularized squared coefficients."""

    probabilities = regularized_probabilities(vector, delta)
    return float(-np.sum(probabilities * np.log(probabilities)))


def entropy_lipschitz_constant(dimension: int, delta: float) -> float:
    """Return the explicit global bound from the article."""

    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if delta <= 0:
        raise ValueError("delta must be positive")
    return (4.0 / (dimension * delta)) * (
        1.0 + abs(math.log(delta / (1.0 + dimension * delta)))
    )
