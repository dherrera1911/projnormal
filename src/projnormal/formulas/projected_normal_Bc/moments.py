"""Approximation to the moments of the general projected normal distribution projected onto ellipse given by matrix B."""
import torch

from .. import projected_normal_c as pnc_formulas

__all__ = ["mean", "second_moment"]


def __dir__():
    return __all__


def mean(mean_x, covariance_x, const, B=None, B_chol=None):
    r"""
    Compute the mean of :math:`y = x/\sqrt{x^T B x + c}`,
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`,
    :math:`B` is a symmetric positive definite matrix and
    :math:`c` is a positive constant.
    Uses a Taylor approximation. (:math:`y` is distributed on the
    ellipse defined by :math:`B`.).

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      const : ``torch.Tensor``
          Constant added to the denominator. Shape is ``()``.

      B : ``torch.Tensor``, optional
          Symmetric positive definite matrix defining the ellipse. Shape is ``(n_dim, n_dim)``.

      B_chol : ``torch.Tensor``, optional
          Cholesky decomposition of B. Can be provided to avoid recomputing it. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          Expected value for the projected normal on ellipse. Shape is ``(n_dim,)``.
    """
    if B_chol is None:
        if B is None:
            raise ValueError("Either B or B_chol must be provided.")
        B_chol = torch.linalg.cholesky(B)

    # Change basis to make B the identity
    mean_z = B_chol.T @ mean_x
    covariance_z = B_chol.T @ covariance_x @ B_chol

    # Compute the mean in the new basis
    gamma_z = pnc_formulas.mean(
      mean_x=mean_z,
      covariance_x=covariance_z,
      const=const
    )

    # Change back to the original basis
    gamma = torch.linalg.solve_triangular(B_chol.T, gamma_z.unsqueeze(1), upper=True).squeeze()
    return gamma


def second_moment(mean_x, covariance_x, const, B=None, B_chol=None):
    r"""
    Compute the second moment matrix of :math:`y = x/\sqrt{x^T B x + c}`,
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`,
    :math:`B` is a symmetric positive definite matrix and
    :math:`c` is a positive constant. Uses a Taylor approximation.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      const : ``torch.Tensor``
          Constant added to the denominator. Shape is ``()``.

      B : ``torch.Tensor``, optional
          Symmetric positive definite matrix defining the ellipse. Shape is ``(n_dim, n_dim)``.

      B_chol : ``torch.Tensor``, optional
          Cholesky decomposition of B. Can be provided to avoid recomputing it. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          Second moment matrix of :math:`y`. Shape is ``(n_dim, n_dim)``.
    """
    if B_chol is None:
        if B is None:
            raise ValueError("Either B or B_chol must be provided.")
        B_chol = torch.linalg.cholesky(B)

    mean_z = B_chol.T @ mean_x
    covariance_z = B_chol.T @ covariance_x @ B_chol

    # Compute the second moment in the new basis
    sm_z = pnc_formulas.second_moment(
      mean_x=mean_z,
      covariance_x=covariance_z,
      const=const
    )

    # Change back to the original basis
    B_chol_inv = torch.linalg.solve_triangular(B_chol, torch.eye(B_chol.shape[0]), upper=False)
    sm = B_chol_inv.T @ sm_z @ B_chol_inv

    return sm
