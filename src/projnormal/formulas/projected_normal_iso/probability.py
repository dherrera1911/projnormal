"""Probability density function (PDF) for the projected normal distribution with isotropic covariance of the unprojected Gaussian."""

import torch

from .. import projected_normal as pn_formulas

__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mean_x, var_x, y):
    r"""
    Compute the pdf at points y for the distribution of the variable
    :math:`y = x/\sqrt{x^T x}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`\Sigma_x = \sigma^2 I` (isotropic covariance matrix).


    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      var_x : ``torch.tensor``
          variance of `x`. shape is ``()``.

      y : ``torch.Tensor``
          Points where to evaluate the PDF. Shape is ``(n_points, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          PDF evaluated at each y. Shape is ``(n_points,)``.
    """
    lpdf = log_pdf(mean_x, var_x, y)
    pdf = torch.exp(lpdf)
    return pdf


def log_pdf(mean_x, var_x, y):
    r"""
    Compute the log-pdf at points y for the distribution of the variable
    :math:`y = x/\sqrt{x^T x}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`\Sigma_x = \sigma^2 I` (isotropic covariance matrix).


    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      var_x : ``torch.tensor``
          variance of `x`. shape is ``()``.

      y : ``torch.Tensor``
          Points where to evaluate the PDF. Shape is ``(n_points, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          Log-PDF evaluated at each y. Shape is ``(n_points,)``.
    """
    iso_cov = torch.eye(
      mean_x.shape[0], device=var_x.device, dtype=var_x.dtype
    ) * var_x
    lpdf = pn_formulas.log_pdf(mean_x, iso_cov, y)
    return lpdf
