"""Finite theory-aligned descriptor assembled from fixed components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .entropy import regularized_entropy
from .fcgr import fcgr_matrix, reverse_complement
from .polar import PolarDisk, polar_block_energy, polar_coefficients
from .smoothing import smoothed_coefficient_exact
from .transforms import block_energy, fwht2, haar2


@dataclass(frozen=True)
class DescriptorConfig:
    """Fixed parameters declared before model fitting."""

    scales: tuple[int, ...] = (3, 4)
    entropy_delta: float = 1e-4
    min_valid_window_fraction: float = 0.8
    smoothing_c: float = 0.2
    smoothed_indices: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1), (1, 1))
    polar_disks: tuple[PolarDisk, ...] = (
        PolarDisk(0.5, 0.5, 0.25),
    )
    polar_max_radial_index: int = 2
    polar_max_angular_index: int = 2
    polar_radial_nodes: int = 32
    polar_angular_nodes: int = 64
    reverse_complement_average: bool = True

    def __post_init__(self) -> None:
        if not self.scales or any(k <= 0 for k in self.scales):
            raise ValueError("scales must be a nonempty tuple of positive integers")
        if not (0.0 < self.smoothing_c < 0.5):
            raise ValueError("Require 0 < smoothing_c < 1/2")


def _default_blocks(size: int) -> tuple[tuple[slice, slice], ...]:
    """Return fixed low-frequency, mixed, and high-frequency blocks."""

    half = max(1, size // 2)
    return (
        (slice(0, half), slice(0, half)),
        (slice(0, half), slice(half, size)),
        (slice(half, size), slice(0, half)),
        (slice(half, size), slice(half, size)),
    )


def _descriptor_without_rc(sequence: str, config: DescriptorConfig) -> np.ndarray:
    features: list[float] = []
    for k in config.scales:
        result = fcgr_matrix(
            sequence,
            k,
            min_valid_fraction=config.min_valid_window_fraction,
        )
        matrix = result.matrix
        wh = fwht2(matrix)
        haar = haar2(matrix)
        for block in _default_blocks(matrix.shape[0]):
            features.append(block_energy(wh, block))
        for block in _default_blocks(matrix.shape[0]):
            features.append(block_energy(haar, block))
        features.append(regularized_entropy(wh.ravel(), config.entropy_delta))
        features.append(regularized_entropy(haar.ravel(), config.entropy_delta))

        epsilon = config.smoothing_c * (2.0 ** (-k))
        for p, q in config.smoothed_indices:
            features.append(smoothed_coefficient_exact(matrix, p, q, epsilon, epsilon))

        for disk in config.polar_disks:
            coefficients = polar_coefficients(
                matrix,
                disk,
                config.polar_max_radial_index,
                config.polar_max_angular_index,
                radial_nodes=config.polar_radial_nodes,
                angular_nodes=config.polar_angular_nodes,
            )
            features.append(polar_block_energy(coefficients, (slice(None), slice(None))))
            features.append(float(coefficients[0, 0]))

    return np.asarray(features, dtype=np.float64)


def theory_aligned_descriptor(sequence: str, config: DescriptorConfig) -> np.ndarray:
    """Compute the finite fixed descriptor from the article.

    Quality-control statistics are deliberately excluded from the mathematical
    descriptor and should be stored separately by an experimental pipeline.
    """

    direct = _descriptor_without_rc(sequence, config)
    if not config.reverse_complement_average:
        return direct
    reverse = _descriptor_without_rc(reverse_complement(sequence), config)
    return 0.5 * (direct + reverse)
