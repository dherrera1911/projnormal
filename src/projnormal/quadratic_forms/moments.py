"""Moments of quadratic forms of multidimensional Gaussian distributions."""

from __future__ import annotations

import torch

__all__ = [
  "mean",
  "variance",
  "qf_covariance",
  "qf_linear_covariance",
]


def __dir__():
    return __all__


def mean(mean_x: torch.Tensor, covariance_x: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Compute the mean of :math:`x^T M x`, where :math:`x`
    follows a multivariate normal distribution
    :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape ``(n_dim)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape ``(n_dim, n_dim)`` or scalar (isotropic covariance).

      M : ``torch.Tensor``, optional
          Matrix in quadratic form.  If a vector is provided, it is used as the diagonal of `M`.
          Default is the identity matrix. Shape ``(n_dim, n_dim)`` or ``(n_dim,)``.

    Returns
    -------
      ``torch.Tensor``
          Expected value of the quadratic form. Shape ``()``.
    """
    if M is None:
        M = torch.ones(
          len(mean_x), dtype=mean_x.dtype, device=mean_x.device
        )
    if M.dim() == 1:
        mean_quadratic = _mean_diagonal(mean_x, covariance_x, M)
    else:
        term1 = _product_trace(M, covariance_x)
        term2 = torch.einsum("d,db,b->", mean_x, M, mean_x)
        mean_quadratic = term1 + term2
    return mean_quadratic


def _mean_diagonal(mean_x, covariance_x, M_diagonal):
    r"""
    Compute the mean of :math:`x^T M x`, where :math:`x`
    follows a multivariate normal distribution
    :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)` and :math:`M` is diagonal.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape ``(n_dim)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape ``(n_dim, n_dim)``.

      M_diagonal : ``torch.Tensor``
          Diagonal elements of the matrix `M`. Shape ``(n_dim,)``.

    Returns
    -------
      ``torch.Tensor``
          Expected value of the quadratic form. Shape ``()``.
    """
    term1 = torch.einsum("ii,i->", covariance_x, M_diagonal)
    term2 = torch.einsum("i,i,i->", mean_x, M_diagonal, mean_x)
    mean_quadratic = term1 + term2
    return mean_quadratic


def variance(mean_x, covariance_x, M=None):
    r"""
    Compute the variance of :math:`x^T M x`, where :math:`x`
    follows a multivariate normal distribution
    :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape ``(n_dim)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape ``(n_dim, n_dim)``.

      M : ``torch.Tensor``
          Matrix in the quadratic form. If a vector is provided, it is used as the diagonal of `M`.
          Default is the identity matrix. Shape ``(n_dim, n_dim)`` or ``(n_dim,)``.

    Returns
    -------
      ``torch.Tensor``
          Variance of the quadratic form. Shape ``()``.
    """
    if M is None:
        M = torch.ones(
            len(mean_x), dtype=mean_x.dtype, device=mean_x.device
        )
    if M.dim() == 1:
        psi_qf = _variance_diagonal(mean_x, covariance_x, M)
    else:
        # Compute the trace of M*covariance_x*M*covariance_x
        trace = _product_trace4(A=M, B=covariance_x, C=M, D=covariance_x)
        # Compute the quadratic form term
        mean_qf = torch.einsum(
            "d,db,bk,km,m->", mean_x, M, covariance_x, M, mean_x
        )
        psi_qf = 2 * trace + 4 * mean_qf
    return psi_qf


def _variance_diagonal(mean_x, covariance_x, M_diagonal):
    r"""
    Compute the variance of :math:`x^T M x`, where :math:`x`
    follows a multivariate normal distribution
    :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)` and :math:`M` is diagonal.

    Parameters
    ----------
      mean_x : ``torch.Tensor``, shape ``torch.Size([n_dim])``
          Mean of `x`.

      covariance_x : ``torch.Tensor``, shape ``torch.Size([n_dim, n_dim])``
          Covariance of `x`.

      M_diagonal : ``torch.Tensor``, shape ``torch.Size([n_dim])``
          Diagonal elements of the matrix `M`.

    Returns
    -------
      ``torch.Tensor``, shape ``torch.Size([])``
          Variance of the quadratic form.
    """
    trace = torch.einsum("i,ij,j,ji->", M_diagonal, covariance_x, M_diagonal, covariance_x)
    mean_qf = torch.einsum(
        "d,d,dk,k,k->", mean_x, M_diagonal, covariance_x, M_diagonal, mean_x
    )
    psi_qf = 2 * trace + 4 * mean_qf
    return psi_qf


def qf_covariance(mean_x, covariance_x, M, M2):
    r"""
    Compute the covariance between :math:`x^T M x` and :math:`x^T M_2 x`,
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. A vector of shape ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. A scalar implies isotropic covariance. Shape ``(n_dim, n_dim)`` or scalar.

      M : ``torch.Tensor``
          Matrix of the first quadratic form. Shape ``(n_dim, n_dim)``

      M2 : ``torch.Tensor``
          Matrix of the second quadratic form. Shape ``(n_dim, n_dim)``

    Returns
    -------
      ``torch.Tensor``
          Covariance of the two quadratic forms. Shape ``()``.
    """
    if covariance_x.dim() == 2:
        trace = _product_trace4(A=M, B=covariance_x, C=M2, D=covariance_x)
    else:  # scalar covariance (isotropic)
        trace = _product_trace(A=M, B=M2) * covariance_x**2
    mean_term = torch.einsum("d,db,bk,km,m->", mean_x, M, covariance_x, M2, mean_x)
    cov_quadratic = 2 * trace + 4 * mean_term
    return cov_quadratic


def qf_linear_covariance(mean_x, covariance_x, M, b):
    r"""
    Compute the covariance between :math:`x^T M x` and the linear form :math:`x^T b`,
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. A vector of shape ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape ``(n_dim, n_dim)``.

      M : ``torch.Tensor``
          Matrix for the quadratic form. Shape ``(n_dim, n_dim)``.

      b : ``torch.Tensor``
          Vector for the linear form. Shape ``(n_dim,)``.

    Returns
    -------
      ``torch.Tensor``
          Covariance between the quadratic and linear forms.
    """
    cov_quadratic = 2 * torch.einsum("i,ij,jk,k->", mean_x, M, covariance_x, b)
    return cov_quadratic


def _product_trace(A, B):
    """
    Efficiently compute the trace of a matrix product.

    Parameters
    ----------
      A : ``torch.Tensor``, shape ``torch.Size([n_dim, n_dim])``
          First matrix.

      B : ``torch.Tensor``, shape ``torch.Size([n_dim, n_dim])``
          Second matrix.

    Returns
    -------
      ``torch.Tensor``, shape ``torch.Size([])``
          Trace of ``A @ B``.
    """
    return torch.einsum("ij,ji->", A, B)


def _product_trace4(A, B, C, D):
    """
    Efficiently compute the trace of four matrix products.

    Parameters
    ----------
      A, B, C, D : ``torch.Tensor``, each shape ``torch.Size([n_dim, n_dim])``
          Matrices to multiply in order.

    Returns
    -------
      ``torch.Tensor``, shape ``torch.Size([])``
          Trace of ``A @ B @ C @ D``.
    """
    return torch.einsum("ij,jk,kl,li->", A, B, C, D)
