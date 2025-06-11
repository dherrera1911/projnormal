"""Probability density function (PDF) for the projected normal distribution with isotropic covariance of the unprojected Gaussian."""

import torch

from ..projected_normal import pdf as gen_pdf


__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mean_x, var_x, y):
    """
    Compute the probability density function of a projected Gaussian distribution
    with parameters mean_x and covariance var_x*eye(n_dim) at points y.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      var_x : torch.Tensor, shape ()
          Variance of X elements.

      y : torch.Tensor, shape (n_points, n_dim)
          Points where to evaluate the PDF.

    Returns
    ----------------
      torch.Tensor, shape (n_points)
          PDF evaluated at each y.
    """
    lpdf = log_pdf(mean_x, var_x, y)
    pdf = torch.exp(lpdf)
    return pdf


def log_pdf(mean_x, var_x, y):
    """
    Compute the log probability density function of a projected
    normal distribution with parameters mean_x and covariance var_x*eye(n_dim) at points y.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      var_x : torch.Tensor, shape ()
          Variance of X elements.

      y : torch.Tensor, shape (n_points, n_dim)
          Points where to evaluate the PDF.

    Returns
    ----------------
      torch.Tensor, shape (n_points)
          log-PDF evaluated at each y.
    """
    iso_cov = torch.eye(
      mean_x.shape[0], device=var_x.device, dtype=var_x.dtype
    ) * var_x
    lpdf = gen_pdf.log_pdf(mean_x, iso_cov, y)
    return lpdf
