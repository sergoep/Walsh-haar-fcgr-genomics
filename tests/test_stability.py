import numpy as np

from walsh_haar_fcgr.fcgr import fcgr_matrix
from walsh_haar_fcgr.robustness import delete, insert, substitute


def _check_bound(sequence: str, modified: str, k: int, edit_count: int) -> None:
    left = np.sum(np.abs(fcgr_matrix(sequence, k).matrix - fcgr_matrix(modified, k).matrix))
    n_min = min(len(sequence) - k + 1, len(modified) - k + 1)
    right = (2 * k + 1) * edit_count / n_min
    assert left <= right + 1e-12


def test_fcgr_edit_bound_for_substitutions_insertions_and_deletions() -> None:
    sequence = "ACGT" * 30
    _check_bound(sequence, substitute(sequence, 3, seed=1), 4, 3)
    _check_bound(sequence, insert(sequence, 3, seed=1), 4, 3)
    _check_bound(sequence, delete(sequence, 3, seed=1), 4, 3)
