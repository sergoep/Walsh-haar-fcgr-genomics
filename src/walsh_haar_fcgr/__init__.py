"""Theory-aligned FCGR, Walsh--Haar, toric smoothing, and polar features."""

from .fcgr import FCGRResult, fcgr_matrix, reverse_complement
from .transforms import fwht2, haar2
from .entropy import regularized_entropy
from .smoothing import SmoothedWalsh
from .polar import PolarDisk, polar_coefficients, polar_block_energy
from .descriptors import DescriptorConfig, theory_aligned_descriptor
from .qc import qc_summary

__all__ = [
    "FCGRResult",
    "fcgr_matrix",
    "reverse_complement",
    "fwht2",
    "haar2",
    "regularized_entropy",
    "SmoothedWalsh",
    "PolarDisk",
    "polar_coefficients",
    "polar_block_energy",
    "DescriptorConfig",
    "theory_aligned_descriptor",
    "qc_summary",
]
