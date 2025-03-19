"""Probability density function (PDF) for the general projected normal distribution."""
import torch
from projnormal.ellipse_linalg import spd_sqrt

from ..general import pdf as _png_pdf


__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mean_x, covariance_x, y, const, B=None, B_sqrt=None, B_sqrt_ldet=None):
    """
    Compute the probability density function at points y for the variable
    Y = X/(X'BX)^0.5 where X ~ N(mean_x, covariance_x) and B is a symmetric
    positive definite matrix . Y has a general projected normal distribution
    on an ellipsoid.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

      const : torch.Tensor, shape ()
        Constant added to the denominator. Must be positive.

      y : torch.Tensor, shape (n_points, n_dim)
          Points where to evaluate the PDF.

      B : torch.Tensor, shape (n_dim, n_dim), optional
        Symmetric positive definite matrix defining the ellipse.

      B_sqrt : torch.Tensor, shape (n_dim, n_dim), optional
        Square root of B.

      B_sqrt_ldet : torch.Tensor, shape (), optional
        Log-Determinant of B_sqrt.

    Returns
    ----------------
      torch.Tensor, shape (n_points)
          PDF evaluated at each y.
    """
    lpdf = log_pdf(
      mean_x=mean_x,
      covariance_x=covariance_x,
      y=y,
      const=const,
      B=B,
      B_sqrt=B_sqrt,
      B_sqrt_ldet=B_sqrt_ldet
    )
    pdf = torch.exp(lpdf)
    return pdf


def log_pdf(mean_x, covariance_x, y, const, B_sqrt=None, B_sqrt_ldet=None):
    """
    Compute the log probability density function at points y for the variable
    Y = X/(X'X)^0.5 where X ~ N(mean_x, covariance_x). Y has a general projected
    normal distribution.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

      y : torch.Tensor, shape (n_points, n_dim)
          Points where to evaluate the PDF.

      const : torch.Tensor, shape ()
        Constant added to the denominator. Must be positive.

      B : torch.Tensor, shape (n_dim, n_dim), optional
        Symmetric positive definite matrix defining the ellipse.

      B_sqrt : torch.Tensor, shape (n_dim, n_dim), optional
        Square root of B.

      B_sqrt_ldet : torch.Tensor, shape (), optional
        Determinant of B_sqrt.

    Returns
    ----------------
      torch.Tensor, shape (n_points)
          Log-PDF evaluated at each y.
    """
    if B_sqrt is None or B_sqrt_ldet is None:
        if B is None:
            raise ValueError(
              "Either B or B_sqrt and B_sqrt_ldet must be provided."
            )
        B_sqrt = spd_sqrt(B, return_inverse=False)
        B_sqrt_ldet = torch.logdet(B_sqrt)
    elif B is None:
        B = B_sqrt @ B_sqrt

    # Change basis to make B the identity
    mean_z = B_sqrt @ mean_x
    covariance_z = B_sqrt @ covariance_x @ B_sqrt
    y_z = y @ B_sqrt

    # Compute the PDF of the transformed variable
    lpdf = _pnc_pdf.log_pdf(
      mean_x=mean_z,
      covariance_x=covariance_z,
      y=y_z,
      const=const
    ) + B_sqrt_ldet

    return lpdf
