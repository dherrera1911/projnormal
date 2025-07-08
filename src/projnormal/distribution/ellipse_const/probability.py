"""Probability density function (PDF) for the general projected normal distribution."""
import torch

from .. import const as const_dist

__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mean_x, covariance_x, y, const, B=None, B_chol=None):
    """
    Compute the probability density function at points y for the variable
    Y = X/(X'BX)^0.5 where X ~ N(mean_x, covariance_x) and B is a symmetric
    positive definite matrix . Y has a general projected normal distribution
    on an ellipsoid.

    Parameters
    ----------
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

      B_chol : torch.Tensor, shape (n_dim, n_dim), optional
          Cholesky decomposition matrix L, such that B = LL'.
          Can be provided to avoid recomputing it.

    Returns
    -------
      torch.Tensor, shape (n_points)
          PDF evaluated at each y.
    """
    lpdf = log_pdf(
      mean_x=mean_x,
      covariance_x=covariance_x,
      y=y,
      const=const,
      B=B,
      B_chol=B_chol,
    )
    pdf = torch.exp(lpdf)
    return pdf


def log_pdf(mean_x, covariance_x, y, const, B=None, B_chol=None):
    """
    Compute the log probability density function at points y for the variable
    Y = X/(X'X)^0.5 where X ~ N(mean_x, covariance_x). Y has a general projected
    normal distribution.

    Parameters
    ----------
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

      B_chol : torch.Tensor, shape (n_dim, n_dim), optional
          Cholesky decomposition matrix L, such that B = LL'.
          Can be provided to avoid recomputing it.

    Returns
    -------
      torch.Tensor, shape (n_points)
          Log-PDF evaluated at each y.
    """
    if B_chol is None:
        if B is None:
            raise ValueError("Either B or B_chol must be provided.")
        B_chol = torch.linalg.cholesky(B)

    # Change basis to make B the identity
    mean_z = B_chol.T @ mean_x
    covariance_z = B_chol.T @ covariance_x @ B_chol
    y_z = y @ B_chol

    # Compute the PDF of the transformed variable
    B_chol_ldet = torch.sum(torch.log(torch.diag(B_chol)))
    lpdf = const_dist.log_pdf(
      mean_x=mean_z,
      covariance_x=covariance_z,
      y=y_z,
      const=const
    ) + B_chol_ldet

    return lpdf
