"""Toric smoothing of right-continuous periodic Walsh functions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def walsh_value(index: int, x: float) -> float:
    """Evaluate the right-continuous periodic Paley-ordered Walsh function."""

    if index < 0:
        raise ValueError("index must be nonnegative")
    if index == 0:
        return 1.0
    x = x % 1.0
    level = index.bit_length()
    cell = min(int(math.floor(x * (1 << level))), (1 << level) - 1)
    parity = (index & cell).bit_count() & 1
    return -1.0 if parity else 1.0


def walsh_interval_signs(index: int) -> np.ndarray:
    """Return right-continuous Walsh signs on the minimal dyadic grid."""

    if index < 0:
        raise ValueError("index must be nonnegative")
    if index == 0:
        return np.array([1.0])
    level = index.bit_length()
    size = 1 << level
    return np.array([walsh_value(index, (j + 0.5) / size) for j in range(size)])


def toric_jump_points(index: int) -> tuple[float, ...]:
    """Return jump locations on the torus, including 0 when appropriate."""

    signs = walsh_interval_signs(index)
    size = signs.size
    if size == 1:
        return ()
    points: list[float] = []
    for j in range(size):
        if signs[j - 1] != signs[j]:
            points.append(j / size)
    return tuple(points)


@dataclass(frozen=True)
class SmoothedWalsh:
    """Linear toric smoothing of a Walsh function at all jump points."""

    index: int
    epsilon: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be nonnegative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        jumps = self.jumps
        if len(jumps) > 1:
            sorted_points = sorted(jumps)
            gaps = [
                (sorted_points[(i + 1) % len(sorted_points)] - sorted_points[i]) % 1.0
                for i in range(len(sorted_points))
            ]
            if 2.0 * self.epsilon >= min(gaps) - 1e-15:
                raise ValueError(
                    "Smoothing arcs overlap. Require 2*epsilon smaller than the "
                    "minimal toric spacing between distinct jumps."
                )

    @property
    def jumps(self) -> tuple[float, ...]:
        return toric_jump_points(self.index)

    @property
    def jump_count(self) -> int:
        return len(self.jumps)

    @property
    def lipschitz_constant(self) -> float:
        return 0.0 if self.index == 0 else 1.0 / self.epsilon

    @property
    def theoretical_l2_error_sq(self) -> float:
        return (2.0 / 3.0) * self.jump_count * self.epsilon

    def value(self, x: float) -> float:
        """Evaluate the smoothed periodic function."""

        x = x % 1.0
        if not self.jumps:
            return walsh_value(self.index, x)
        for jump in self.jumps:
            displacement = ((x - jump + 0.5) % 1.0) - 0.5
            if abs(displacement) < self.epsilon or math.isclose(
                abs(displacement), self.epsilon, abs_tol=1e-15
            ):
                left = walsh_value(self.index, (jump - self.epsilon) % 1.0)
                right = walsh_value(self.index, (jump + self.epsilon) % 1.0)
                alpha = (displacement + self.epsilon) / (2.0 * self.epsilon)
                return float((1.0 - alpha) * left + alpha * right)
        return walsh_value(self.index, x)

    def _breakpoints(self) -> tuple[float, ...]:
        points = {0.0, 1.0}
        for jump in self.jumps:
            points.add((jump - self.epsilon) % 1.0)
            points.add((jump + self.epsilon) % 1.0)
        return tuple(sorted(points))

    def integrate(self, start: float, end: float) -> float:
        """Integrate exactly on a non-wrapping interval inside [0, 1]."""

        if not (0.0 <= start <= end <= 1.0):
            raise ValueError("Expected 0 <= start <= end <= 1")
        if math.isclose(start, end):
            return 0.0
        points = [start]
        points.extend(point for point in self._breakpoints() if start < point < end)
        points.append(end)
        total = 0.0
        for left, right in zip(points, points[1:]):
            total += 0.5 * (self.value(left) + self.value(right)) * (right - left)
        return total

    def cell_averages(self, grid_size: int) -> np.ndarray:
        """Return exact cell averages on a uniform grid."""

        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        return np.array(
            [
                grid_size * self.integrate(j / grid_size, (j + 1) / grid_size)
                for j in range(grid_size)
            ],
            dtype=np.float64,
        )


def smoothed_coefficient_exact(
    matrix: np.ndarray,
    p: int,
    q: int,
    epsilon_p: float,
    epsilon_q: float,
) -> float:
    """Compute the exact cell-average coefficient for a step FCGR density."""

    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square")
    n = m.shape[0]
    wp = SmoothedWalsh(p, epsilon_p).cell_averages(n)
    wq = SmoothedWalsh(q, epsilon_q).cell_averages(n)
    return float(np.sum(m * np.outer(wq, wp)))


def numerical_l2_error_sq(smoothed: SmoothedWalsh, samples: int = 200_000) -> float:
    """Numerically approximate the squared L2 smoothing error for tests."""

    xs = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    smooth = np.array([smoothed.value(float(x)) for x in xs])
    original = np.array([walsh_value(smoothed.index, float(x)) for x in xs])
    return float(np.mean((smooth - original) ** 2))
