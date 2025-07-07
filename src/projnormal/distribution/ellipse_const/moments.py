"""Approximation to the moments of the general projected normal distribution projected onto ellipse given by matrix B."""
import torch

from ..const import moments as _pnc_moments

__all__ = ["mean", "second_moment"]


def __dir__():
    return __all__


def mean(mean_x, covariance_x, const, B=None, B_chol=None):
    """
    Compute the Taylor approximation to the expected value of the variable
    Y = X/(X'BX + C)^0.5, where X~N(mean_x, covariance_x) and B is a symmetric positive
    defininte matrix. Y is an elliptical version of the general projected normal
    distribution.

    Either the matrix B or its square root and its inverse can be provided.
    If B is provided, the square root and inverse are computed internally.

    Parameters
    ----------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

      const : torch.Tensor, shape ()
        Constant added to the denominator.

      B : torch.Tensor, shape (n_dim, n_dim), optional
          Symmetric positive definite matrix defining the ellipse.

      B_chol : torch.Tensor, shape (n_dim, n_dim), optional
        Cholesky decomposition of B. Can be provided to avoid recomputing it.

    Returns
    -------
      torch.Tensor, shape (n_dim,)
          Expected value for the projected normal on ellipse.
    """
    if B_chol is None:
        if B is None:
            raise ValueError("Either B or B_chol must be provided.")
        B_chol = torch.linalg.cholesky(B)

    # Change basis to make B the identity
    mean_z = B_chol.T @ mean_x
    covariance_z = B_chol.T @ covariance_x @ B_chol

    # Compute the mean in the new basis
    gamma_z = _pnc_moments.mean(
      mean_x=mean_z,
      covariance_x=covariance_z,
      const=const
    )

    # Change back to the original basis
    gamma = torch.linalg.solve_triangular(B_chol.T, gamma_z.unsqueeze(1), upper=True).squeeze()
    return gamma


def second_moment(mean_x, covariance_x, const, B=None, B_chol=None):
    """
    Compute the Taylor approximation to the second moment matrix of the
    variable Y = X/(X'X)^0.5, where X~N(mean_x, covariance_x). Y has a
    general projected normal distribution.

    The approximation is based on the Taylor expansion of the
    function f(n,d) = n/d, where n = X_i*X_j and d = X'X.

    Parameters
    ----------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

      const : torch.Tensor, shape ()
        Constant added to the denominator.

      B : torch.Tensor, shape (n_dim, n_dim), optional
          Symmetric positive definite matrix defining the ellipse.

      B_chol : torch.Tensor, shape (n_dim, n_dim), optional
        Cholesky decomposition of B. Can be provided to avoid recomputing it.

    Returns
    -------
      torch.Tensor, shape (n_dim, n_dim)
          Second moment matrix of Y
    """
    if B_chol is None:
        if B is None:
            raise ValueError("Either B or B_chol must be provided.")
        B_chol = torch.linalg.cholesky(B)

    mean_z = B_chol.T @ mean_x
    covariance_z = B_chol.T @ covariance_x @ B_chol

    # Compute the second moment in the new basis
    sm_z = _pnc_moments.second_moment(
      mean_x=mean_z,
      covariance_x=covariance_z,
      const=const
    )

    # Change back to the original basis
    B_chol_inv = torch.linalg.solve_triangular(B_chol, torch.eye(B_chol.shape[0]), upper=False)
    sm = B_chol_inv.T @ sm_z @ B_chol_inv

    return sm
