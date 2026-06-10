import numpy as np

from walsh_haar_fcgr.fcgr import fcgr_matrix
from walsh_haar_fcgr.transforms import fwht2, haar2, haar_matrix


def test_fwht_energy_preservation() -> None:
    matrix = fcgr_matrix("AACCGGTTAACGTCGATCGATCGT", 4).matrix
    assert np.isclose(np.linalg.norm(fwht2(matrix)), np.linalg.norm(matrix))


def test_haar_energy_preservation() -> None:
    matrix = fcgr_matrix("AACCGGTTAACGTCGATCGATCGT", 4).matrix
    assert np.isclose(np.linalg.norm(haar2(matrix)), np.linalg.norm(matrix))


def test_haar_matrix_is_orthogonal() -> None:
    h = haar_matrix(16)
    assert np.allclose(h @ h.T, np.eye(16), atol=1e-12)
