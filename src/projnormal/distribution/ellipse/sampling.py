"""Sampling functions for the general projected normal distribution."""
import torch
from projnormal.linalg import spd_sqrt

from ..ellipse_const import sampling as _pnec_sampling

__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, n_samples, B=None, B_chol=None):
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

      B : torch.Tensor, shape (n_dim, n_dim), optional
          Symmetric positive definite matrix defining the ellipse.

      B_chol : torch.Tensor, shape (n_dim, n_dim), optional
          Cholesky decomposition matrix L, such that B = LL'.
          Can be provided to avoid recomputing it.

    Returns:
    -----------------
      torch.Tensor, shape (n_samples, n_dim)
          Samples from the projected normal.
    """
    return _pnec_sampling.sample(mean_x=mean_x, covariance_x=covariance_x,
                                 n_samples=n_samples, const=0, B=B, B_chol=B_chol)


def empirical_moments(mean_x, covariance_x, n_samples, B=None, B_chol=None):
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

      n_samples : int
          Number of samples.

      B : torch.Tensor, shape (n_dim, n_dim), optional
          Symmetric positive definite matrix defining the ellipse.

      B_chol : torch.Tensor, shape (n_dim, n_dim), optional
          Cholesky decomposition matrix L, such that B = LL'.
          Can be provided to avoid recomputing it.

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
    return _pnec_sampling.empirical_moments(mean_x=mean_x, covariance_x=covariance_x,
                                           n_samples=n_samples, const=0, B=B, B_chol=B_chol)
