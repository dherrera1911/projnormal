"""Probability density function (PDF) for the projected normal distribution with isotropic covariance of the unprojected Gaussian."""

import torch

from ..general import pdf as gen_pdf


__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mean_x, var_x, y):
    """
    Compute the probability density function of a projected Gaussian distribution
    with parameters mean_x and covariance var_x*eye(n_dim) at points y.

    Parameters
    ----------------
      - mean_x : Mean of the non-projected Gaussian. Shape (n_dim).
      - var_x : Variance of the isotropic Gaussian. Shape (n_dim x n_dim).
      - y : Points where to evaluate the PDF. Shape (n_points x n_dim).

    Returns
    ----------------
      PDF evaluated at y. Shape (n_points).
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
      - mean_x : Mean of the non-projected Gaussian. Shape (n_dim).
      - var_x : Variance of the isotropic Gaussian. Shape (n_dim x n_dim).
      - y : Points where to evaluate the PDF. Shape (n_points x n_dim).

    Returns
    ----------------
      log-PDF evaluated at y. Shape (n_points).
    """
    iso_cov = torch.eye(
      mean_x.shape[0], device=var_x.device, dtype=var_x.dtype
    ) * var_x
    lpdf = gen_pdf.log_pdf(mean_x, iso_cov, y)
    return lpdf
