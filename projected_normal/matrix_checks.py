import torch


__all__ = [
  "is_symmetric", "is_positive_definite", "is_positive_semidefinite"
]


def __dir__():
    return __all__


def is_symmetric(matrix):
    """Check if a matrix is symmetric.

    Parameters:
    ----------------
      - matrix : Matrix to check. (n_dim x n_dim)

    Returns:
    ----------------
      True if B is symmetric, False otherwise
    """
    return torch.allclose(matrix, matrix.t(), atol=5e-6)


def is_positive_definite(matrix):
    """Check if a matrix is positive definite.

    Parameters:
    ----------------
      - matrix : Matrix to check. (n_dim x n_dim)

    Returns:
    ----------------
      True if B is positive definite, False otherwise
    """
    return torch.all(torch.linalg.eigh(matrix)[0] > 0)


def is_positive_semidefinite(matrix):
    """Check if a matrix is positive definite.

    Parameters:
    ----------------
      - matrix : Matrix to check. (n_dim x n_dim)

    Returns:
    ----------------
      True if B is positive definite, False otherwise
    """
    return torch.all(torch.linalg.eigh(matrix)[0] >= 0)
