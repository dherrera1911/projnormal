"""Functions to check properties of matrices."""
import torch

__all__ = [
  "is_symmetric", "is_positive_definite", "is_positive_semidefinite"
]


def __dir__():
    return __all__


def is_symmetric(matrix, atol=5e-6):
    """Check if a matrix is symmetric.

    Parameters
    ----------
      matrix : torch.Tensor, shape (n_dim, n_dim)
          Matrix to check for symmetry.

      atol : float, optional
          Absolute tolerance for the check. Default is 5e-6.

    Returns
    -------
      bool
          True if B is symmetric, False otherwise
    """
    return torch.allclose(matrix, matrix.t(), atol=atol)


def is_positive_definite(matrix):
    """Check if a matrix is positive definite.

    Parameters
    ----------
      matrix : torch.Tensor, shape (n_dim, n_dim)
          Matrix to check for positive definiteness.

    Returns
    -------
      bool
          True if B is positive definite, False otherwise
    """
    return torch.all(torch.linalg.eigh(matrix)[0] > 0)


def is_positive_semidefinite(matrix):
    """Check if a matrix is positive definite.

    Parameters
    ----------
      matrix : torch.Tensor, shape (n_dim, n_dim)
          Matrix to check for positive semidefiniteness.

    Returns
    -------
      bool
          True if B is positive definite, False otherwise
    """
    return torch.all(torch.linalg.eigh(matrix)[0] >= 0)
