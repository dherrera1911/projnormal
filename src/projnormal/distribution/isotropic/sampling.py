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
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      var_x : torch.Tensor, shape ()
          Variance of X elements.

      n_samples : int
          Number of samples.

    Returns:
    -----------------
      torch.Tensor, shape (n_samples, n_dim)
          Samples from the projected normal.
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
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      var_x : torch.Tensor, shape ()
          Variance of X elements.

      n_samples : int
          Number of samples.

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
    covariance_x = var_x * torch.eye(
      len(mean_x), device=mean_x.device, dtype=mean_x.dtype
    )
    moment_dict = c50_sampling.empirical_moments(
      mean_x=mean_x, covariance_x=covariance_x, n_samples=n_samples, c50=0
    )
    return moment_dict
