"""
Pytest tests.

Tests:
    - test_eigen: test if eigen function returns eigenvalues and eigenvectors
"""
from mltool.linalg.eigen import eigen
import numpy as np


def test_eigen():
    """
    Test if eigenvectors and eigenvalues are calculated correctly.

    Function is given the diagonal matric with values 1, 2 and 3
    Correct eigenvalues: 1, 2, 3
    Correct eigenvector: [1, 0, 0], [0, 1, 0], [0, 0, 1]
    """
    vals, vec = eigen(np.diag((1, 2, 3)))
    assert all(vals == [1, 2, 3])
    assert all(vec[:, 0] == [1, 0, 0])
    assert all(vec[:, 1] == [0, 1, 0])
    assert all(vec[:, 2] == [0, 0, 1])
