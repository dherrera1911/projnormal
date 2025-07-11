"""Approximation to the moments of the general projected normal distribution projected onto ellipse given by matrix B."""

from .. import projected_normal_Bc as pnbc_formulas

__all__ = ["mean", "second_moment"]


def __dir__():
    return __all__


def mean(mean_x, covariance_x, B=None, B_chol=None):
    r"""
    Compute the mean of :math:`y = x/\sqrt{x^T B x}`,
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)` and
    :math:`B` is a symmetric positive definite matrix.
    Uses a Taylor approximation. (:math:`y` is distributed on the
    ellipse defined by :math:`B`.).

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      B : ``torch.Tensor``, optional
          Symmetric positive definite matrix defining the ellipse. Shape is ``(n_dim, n_dim)``.

      B_chol : ``torch.Tensor``, optional
          Cholesky decomposition of B. Can be provided to avoid recomputing it. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          Expected value for the projected normal on ellipse. Shape is ``(n_dim,)``.
    """
    return pnbc_formulas.mean(mean_x=mean_x, covariance_x=covariance_x,
                              const=0, B=B, B_chol=B_chol)


def second_moment(mean_x, covariance_x, B=None, B_chol=None):
    """
    Compute the Taylor approximation to the second moment matrix of the
    variable Y = X/(X'BX)^0.5, where X~N(mean_x, covariance_x). Y has a
    general projected normal distribution.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      B : ``torch.Tensor``, optional
          Symmetric positive definite matrix defining the ellipse. Shape is ``(n_dim, n_dim)``.

      B_chol : ``torch.Tensor``, optional
          Cholesky decomposition of B. Can be provided to avoid recomputing it. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          Second moment matrix of :math:`y`. Shape is ``(n_dim, n_dim)``.
    """
    return pnbc_formulas.second_moment(mean_x=mean_x, covariance_x=covariance_x,
                                       const=0, B=B, B_chol=B_chol)
