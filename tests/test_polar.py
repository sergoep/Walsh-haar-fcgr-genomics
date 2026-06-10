import math
import numpy as np

from walsh_haar_fcgr.polar import PolarDisk, polar_coefficients


def test_constant_density_polar_dc() -> None:
    matrix = np.full((16, 16), 1.0 / 256.0)
    disk = PolarDisk(0.5, 0.5, 0.25)
    coefficients = polar_coefficients(matrix, disk, 2, 2, radial_nodes=64, angular_nodes=128)
    assert np.isclose(coefficients[0, 0], math.sqrt(math.pi), atol=5e-3)
    off_dc = coefficients.copy()
    off_dc[0, 0] = 0.0
    assert np.max(np.abs(off_dc)) < 5e-3
