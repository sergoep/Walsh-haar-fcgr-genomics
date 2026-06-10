import numpy as np

from walsh_haar_fcgr.entropy import entropy_lipschitz_constant, regularized_entropy


def test_entropy_lipschitz_bound() -> None:
    x = np.array([0.2, 0.1, -0.3, 0.4])
    y = np.array([0.21, 0.09, -0.28, 0.39])
    delta = 1e-3
    left = abs(regularized_entropy(x, delta) - regularized_entropy(y, delta))
    right = entropy_lipschitz_constant(x.size, delta) * np.linalg.norm(x - y)
    assert left <= right + 1e-12
