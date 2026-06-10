"""Fixed polar Walsh dictionary and numerical quadrature."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .smoothing import walsh_value


@dataclass(frozen=True)
class PolarDisk:
    """A fixed closed disk contained in the FCGR unit square."""

    center_x: float
    center_y: float
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if not (0.0 <= self.center_x <= 1.0 and 0.0 <= self.center_y <= 1.0):
            raise ValueError("disk center must lie in [0, 1]^2")
        if (
            self.center_x - self.radius < 0.0
            or self.center_x + self.radius > 1.0
            or self.center_y - self.radius < 0.0
            or self.center_y + self.radius > 1.0
        ):
            raise ValueError("the closed disk must be contained in [0, 1]^2")


def step_density_value(matrix: np.ndarray, x: float, y: float) -> float:
    """Evaluate the step density F[M] = n^2 m_ij on the unit square."""

    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square")
    n = m.shape[0]
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("coordinates must lie in [0, 1]")
    col = min(int(math.floor(x * n)), n - 1)
    row = min(int(math.floor(y * n)), n - 1)
    return float(n * n * m[row, col])


def polar_coefficients(
    matrix: np.ndarray,
    disk: PolarDisk,
    max_radial_index: int,
    max_angular_index: int,
    *,
    radial_nodes: int = 48,
    angular_nodes: int = 96,
) -> np.ndarray:
    """Approximate exact polar coefficients by fixed midpoint quadrature."""

    if max_radial_index < 0 or max_angular_index < 0:
        raise ValueError("polar truncation levels must be nonnegative")
    if radial_nodes <= 0 or angular_nodes <= 0:
        raise ValueError("quadrature node counts must be positive")

    coefficients = np.zeros((max_radial_index + 1, max_angular_index + 1))
    # Use a uniform midpoint grid in u=r^2, not in r. Since
    # r\,dr = du/2, this matches the analytical change of variables used
    # in the orthonormality proof and gives a substantially cleaner
    # quadrature for discontinuous Walsh factors.
    du = 1.0 / radial_nodes
    dtheta = 2.0 * math.pi / angular_nodes
    for radial_bin in range(radial_nodes):
        u = (radial_bin + 0.5) * du
        r = math.sqrt(u)
        for angular_bin in range(angular_nodes):
            theta = (angular_bin + 0.5) * dtheta
            v = theta / (2.0 * math.pi)
            x = disk.center_x + disk.radius * r * math.cos(theta)
            y = disk.center_y + disk.radius * r * math.sin(theta)
            density = step_density_value(matrix, x, y)
            weight = 0.5 * du * dtheta / math.sqrt(math.pi)
            radial_values = [walsh_value(m, u) for m in range(max_radial_index + 1)]
            angular_values = [walsh_value(n, v) for n in range(max_angular_index + 1)]
            for m, radial_value in enumerate(radial_values):
                for n, angular_value in enumerate(angular_values):
                    coefficients[m, n] += density * radial_value * angular_value * weight
    return coefficients


def polar_block_energy(coefficients: np.ndarray, block: tuple[slice, slice]) -> float:
    """Return a fixed polar block energy."""

    selected = np.asarray(coefficients, dtype=np.float64)[block]
    return float(np.sum(selected * selected))
