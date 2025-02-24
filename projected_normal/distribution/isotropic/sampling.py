"""Sampling functions for the isotropic projected normal distribution."""

import torch

from ..c50 import sampling as c50_sampling

__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, var_x, n_samples):
    """
    Sample from the variable Y = X/(X'X)^0.5, where X~N(mean_x, var_x*I).
    Y has a projected normal distribution.

    Parameters:
    -----------------
      - mean_x : Mean of X. (n_dim)
      - var_x : Variance of X elements (Scalar)
      - n_samples : Number of samples.

    Returns:
    -----------------
      Samples from the projected normal. (n_samples x n_dim)
    """
    covariance_x = var_x * torch.eye(
      len(mean_x), device=mean_x.device, dtype=mean_x.dtype
    )
    samples_prnorm = c50_sampling.sample(
      mean_x=mean_x, covariance_x=covariance_x, n_samples=n_samples, c50=0
    )
    return samples_prnorm


def empirical_moments(mean_x, var_x, n_samples):
    """
    Compute the mean, covariance and second moment of the variable
    Y = X/(X'X)^0.5, where X~N(mean_x, var_x*I). The mean and covariance are
    obtained by sampling from the distribution.

    Parameters:
    -----------------
      - mean_x : Mean. (n_dim)
      - var_x : Variance of X elements (Scalar)
      - n_samples : Number of samples.

    Returns:
    -----------------
      Dictionary with the following keys:
      - gamma : Mean of the projected normal. (n_dim)
      - psi : Covariance of the projected normal. (n_dim x n_dim)
      - second_moment : Second moment of the projected normal. (n_dim x n_dim)
    """
    covariance_x = var_x * torch.eye(
      len(mean_x), device=mean_x.device, dtype=mean_x.dtype
    )
    moment_dict = c50_sampling.empirical_moments(
      mean_x=mean_x, covariance_x=covariance_x, n_samples=n_samples, c50=0
    )
    return moment_dict
