"""Approximation to the moments of the general projected normal distribution."""
import torch

from ..c50 import moments as c50_moments

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
      - mean_x : Means of normal distributions X. (n_dim)
      - covariance_x : covariance of X. (n_dim x n_dim)

    Returns:
    ----------------
      Expected mean value for each projected normal. Shape (n_dim)
    """
    gamma = c50_moments.mean(mean_x, covariance_x)
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
      - mean_x : Means of normal distributions X. (n_dim)
      - covariance_x : Covariance of the normal distributions (n_dim x n_dim)

    Returns
    ----------------
      Second moment matrix of Y
    """
    sm = c50_moments.second_moment(mean_x, covariance_x)
    return sm
