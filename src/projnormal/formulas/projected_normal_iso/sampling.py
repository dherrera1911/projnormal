"""Sampling functions for the isotropic projected normal distribution."""

import torch

from .. import projected_normal as pn_formulas

__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, var_x, n_samples):
    r"""
    Sample the variable :math:`y = x/\sqrt{x^T x}`
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`\Sigma_x = \sigma^2 I`.


    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      var_x : ``torch.tensor``
          variance of `x`. shape is ``()``.

      n_samples : ``int``
          Number of samples.

    Returns
    -------
      ``torch.Tensor``
          Samples from the projected normal. Shape is ``(n_samples, n_dim)``.
    """
    covariance_x = var_x * torch.eye(
      len(mean_x), device=mean_x.device, dtype=mean_x.dtype
    )
    samples_prnorm = pn_formulas.sample(
      mean_x=mean_x,
      covariance_x=covariance_x,
      n_samples=n_samples,
    )
    return samples_prnorm


def empirical_moments(mean_x, var_x, n_samples):
    r"""
    Compute the mean, covariance and second moment of the variable
    :math:`y = x/\sqrt{x^T x}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`\Sigma_x = \sigma^2 I`, by sampling from the distribution.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      var_x : ``torch.Tensor``
          Variance of `x`. Shape is ``()``.

      n_samples : ``int``
          Number of samples.

    Returns
    -------
      ``dict``
          Dictionary with the keys ``mean``, ``covariance``, and ``second_moment``,
          containing the empirical moments of the projected normal distribution.
    """
    covariance_x = var_x * torch.eye(
      len(mean_x), device=mean_x.device, dtype=mean_x.dtype
    )
    moment_dict = pn_formulas.empirical_moments(
      mean_x=mean_x,
      covariance_x=covariance_x,
      n_samples=n_samples,
    )
    return moment_dict
