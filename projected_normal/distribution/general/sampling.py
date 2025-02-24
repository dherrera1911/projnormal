"""Sampling functions for the general projected normal distribution."""
import torch

from ..c50 import sampling as _c50_sampling

__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, n_samples):
    """
    Sample from the variable Y = X/(X'X)^0.5, where X~N(mean_x, covariance_x).
    The variable Y has a general projected normal distribution.

    Parameters:
    -----------------
      - mean_x : Mean of X. (n_dim)
      - covariance_x : Covariance matrix of X. (n_dim x n_dim)
      - n_samples : Number of samples.

    Returns:
    -----------------
      Samples from the general projected normal. (n_samples x n_dim)
    """
    samples_prnorm = _c50_sampling.sample(
      mean_x=mean_x, covariance_x=covariance_x, n_samples=n_samples, c50=0
    )
    return samples_prnorm


def empirical_moments(mean_x, covariance_x, n_samples):
    """
    Compute the mean, covariance and second moment of the variable
    Y = X/(X'X)^0.5, where X~N(mean_x, covariance_x), by sampling from the
    distribution. The variable Y has a general projected normal distribution.

    Parameters:
    -----------------
      - mean_x : Mean. (n_dim)
      - covariance _x: Covariance matrix. (n_dim x n_dim)
      - n_samples : Number of samples.

    Returns:
    -----------------
      Dictionary with the following keys:
      - gamma : Mean of the projected normal. (n_dim)
      - psi : Covariance of the projected normal. (n_dim x n_dim)
      - second_moment : Second moment of the projected normal. (n_dim x n_dim)
    """
    moment_dict = _c50_sampling.empirical_moments(
      mean_x=mean_x, covariance_x=covariance_x, n_samples=n_samples, c50=0
    )
    return moment_dict
