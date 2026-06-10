import numpy as np

from walsh_haar_fcgr.smoothing import SmoothedWalsh, numerical_l2_error_sq, smoothed_coefficient_exact


def test_toric_smoothing_l2_identity() -> None:
    smooth = SmoothedWalsh(index=5, epsilon=0.01)
    numerical = numerical_l2_error_sq(smooth, samples=400_000)
    assert np.isclose(numerical, smooth.theoretical_l2_error_sq, atol=3e-4)


def test_constant_density_smoothed_dc() -> None:
    matrix = np.full((8, 8), 1.0 / 64.0)
    dc = smoothed_coefficient_exact(matrix, 0, 0, 0.01, 0.01)
    assert np.isclose(dc, 1.0, atol=1e-12)
