"""Probability density function (PDF) for the general projected normal distribution."""
import torch

from .. import projected_normal_c as pnc_formulas

__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mean_x, covariance_x, y, const, B=None, B_chol=None):
    r"""
    Compute the pdf at points y for the distribution of the variable
    :math:`y = x/\sqrt{x^T B x + c}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`,
    :math:`B` is a symmetric positive definite matrix and :math:`c` is a positive constant.


    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
        Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      y : ``torch.Tensor``
          Points where to evaluate the PDF. Shape is ``(n_points, n_dim)``.

      const : ``torch.Tensor``
          Constant added to the denominator. Must be positive. Shape is ``()``.

      B : ``torch.Tensor``, optional
          Matrix B used in the denominator of the projection. If not provided,
          the identity matrix is used. Shape is ``(n_dim, n_dim)``.

      B_chol : ``torch.Tensor``, optional
          Cholesky decomposition of B. Can be provided to avoid recomputing it. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          PDF evaluated at each y. Shape is ``(n_points,)``.
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
    r"""
    Compute the log-pdf at points y for the distribution of the variable
    :math:`y = x/\sqrt{x^T B x + c}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`,
    :math:`B` is a symmetric positive definite matrix and :math:`c` is a positive constant.
    (:math:`y` has a projected normal distribution.).


    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
        Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      y : ``torch.Tensor``
          Points where to evaluate the PDF. Shape is ``(n_points, n_dim)``.

      const : ``torch.Tensor``
          Constant added to the denominator. Must be positive. Shape is ``()``.

      B : ``torch.Tensor``, optional
          Matrix B used in the denominator of the projection. If not provided,
          the identity matrix is used. Shape is ``(n_dim, n_dim)``.

      B_chol : ``torch.Tensor``, optional
          Cholesky decomposition of B. Can be provided to avoid recomputing it. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          Log-PDF evaluated at each y. Shape is ``(n_points,)``.
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
    lpdf = pnc_formulas.log_pdf(
      mean_x=mean_z,
      covariance_x=covariance_z,
      y=y_z,
      const=const
    ) + B_chol_ldet

    return lpdf
