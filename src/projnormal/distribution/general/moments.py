"""Approximation to the moments of the general projected normal distribution."""
import torch

from ..const import moments as const_moments

__all__ = ["mean", "second_moment"]


def __dir__():
    return __all__


def mean(mean_x, covariance_x):
    """
    Compute the Taylor approximation to the expected value of the variable
    Y = X/(X'X)^0.5, where X~N(mean_x, covariance_x). Y has a general
    projected normal distribution.

    The approximation is based on the function
    f(u,v) = u/sqrt(u^2 + v), where u=X_i and v = (X'X - X_i^2).

    Parameters:
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

    Returns:
    ----------------
      torch.Tensor, shape (n_dim,)
          Expected value for the projected normal.
    """
    gamma = const_moments.mean(mean_x, covariance_x)
    return gamma


def second_moment(mean_x, covariance_x):
    """
    Compute the Taylor approximation to the second moment matrix of the
    variable Y = X/(X'X)^0.5, where X~N(mean_x, covariance_x). Y has a
    general projected normal distribution.

    The approximation is based on the Taylor expansion of the
    function f(n,d) = n/d, where n = X_i*X_j and d = X'X.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

    Returns
    ----------------
      torch.Tensor, shape (n_dim, n_dim)
          Second moment matrix of Y
    """
    sm = const_moments.second_moment(mean_x, covariance_x)
    return sm
