"""Sampling functions for the general projected normal distribution."""
import torch
from projnormal.linalg import spd_sqrt

from ..const import sampling as _pnc_sampling

__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, n_samples, const, B=None, B_chol=None):
    """
    Sample from the variable Y = X/(X'BX)^0.5, where X~N(mean_x, covariance_x)
    and B is a symmetric positive definite matrix.
    The variable Y has a general projected normal distribution with
    projection to the ellipsoid defined by B.

    Parameters:
    -----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
          Covariance matrix of X elements.

      n_samples : int
          Number of samples.

      const : torch.Tensor, shape ()
        Constant added to the denominator.

      B : torch.Tensor, shape (n_dim, n_dim), optional
        Symmetric positive definite matrix defining the ellipse.

      B_chol : torch.Tensor, shape (n_dim, n_dim), optional
        Cholesky decomposition of B. Can be provided to avoid recomputing it.

    Returns:
    -----------------
      torch.Tensor, shape (n_samples, n_dim)
          Samples from the projected normal.
    """
    if B_chol is None:
        if B is None:
            raise ValueError("Either B or B_chol must be provided.")
        B_chol = torch.linalg.cholesky(B)

    # Change basis to make B the identity
    mean_z = B_chol.T @ mean_x
    covariance_z = B_chol.T @ covariance_x @ B_chol

    # Sample from the standard projected normal
    samples_prnorm_z = _pnc_sampling.sample(
      mean_x=mean_z,
      covariance_x=covariance_z,
      n_samples=n_samples,
      const=const
    )

    # Change basis back to the original space
    samples_prnorm = torch.linalg.solve_triangular(B_chol.T, samples_prnorm_z.T, upper=True).T
    return samples_prnorm


def empirical_moments(mean_x, covariance_x, const, n_samples, B=None, B_chol=None):
    """
    Compute the mean, covariance and second moment of the variable
    Y = X/(X'X)^0.5, where X~N(mean_x, covariance_x), by sampling from the
    distribution. The variable Y has a general projected normal distribution.

    Parameters:
    -----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
          Covariance matrix of X elements.

      const : torch.Tensor, shape ()
        Constant added to the denominator.

      n_samples : int
          Number of samples.

      B : torch.Tensor, shape (n_dim, n_dim), optional
        Symmetric positive definite matrix defining the ellipse.

      B_chol : torch.Tensor, shape (n_dim, n_dim), optional
        Cholesky decomposition of B. Can be provided to avoid recomputing it.

    Returns:
    -----------------
      dict
          Dictionary with the following keys and values
            'mean' : torch.Tensor, shape (n_dim,)
                Mean of the projected normal.
            'covariance' : torch.Tensor, shape (n_dim, n_dim)
                Covariance of the projected normal.
            'second_moment' : torch.Tensor, shape (n_dim, n_dim)
                Second moment of the projected normal.
    """
    if B_chol is None:
        if B is None:
            raise ValueError("Either B or B_chol must be provided.")
        B_chol = torch.linalg.cholesky(B)

    # Change basis to make B the identity
    mean_z = B_chol.T @ mean_x
    covariance_z = B_chol.T @ covariance_x @ B_chol

    moment_dict_z = _pnc_sampling.empirical_moments(
      mean_x=mean_z,
      covariance_x=covariance_z,
      n_samples=n_samples,
      const=const
    )

    # Change basis back to the original space
    B_chol_inv = torch.linalg.solve_triangular(B_chol, torch.eye(B_chol.shape[0]), upper=False)
    moment_dict = {}
    moment_dict["mean"] = B_chol_inv.T @ moment_dict_z["mean"]
    moment_dict["covariance"] = B_chol_inv.T @ moment_dict_z["covariance"] @ B_chol_inv
    moment_dict["second_moment"] = (
      B_chol_inv.T @ moment_dict_z["second_moment"] @ B_chol_inv
    )
    return moment_dict
