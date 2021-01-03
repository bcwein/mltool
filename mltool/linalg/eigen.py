"""
Python code for linear algebra operations.

Funcions:
    - eigen: returns eigenvectors and eigenvalues
"""
import numpy as np


def eigen(mat):
    """Return eigenvectors (unit vectors) and eigen values.

    Args:
        mat (numpy array): 2D array of linear transformation

    Returns:
        vals: Numpy array of eigenvalues
        vec: 2D numpy array of eigenvectors as unit vectors

    """
    vals, vec = np.linalg.eig(mat)
    return vals, vec
