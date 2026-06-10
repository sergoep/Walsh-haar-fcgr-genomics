import numpy as np

from walsh_haar_fcgr.fcgr import fcgr_matrix, reverse_complement, verify_rc_equivariance


def test_fcgr_normalization_and_masking() -> None:
    result = fcgr_matrix("AACGTNACGT", 3)
    assert np.isclose(result.matrix.sum(), 1.0)
    assert result.masked_windows > 0
    assert result.valid_windows + result.masked_windows == result.total_windows


def test_reverse_complement_equivariance() -> None:
    sequence = "AACCGGTTAACGTCGATCGATCGT"
    assert verify_rc_equivariance(sequence, 4)
    assert reverse_complement(reverse_complement(sequence)) == sequence
