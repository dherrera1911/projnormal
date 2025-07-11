"""Sampling functions for the general projected normal distribution."""
import torch

from .. import projected_normal_c as pnc_formulas

__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, n_samples, const, B=None, B_chol=None):
    r"""
    Sample the variable :math:`y = x/\sqrt{x^T B x + c}`
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`,
    :math:`B` is a symmetric positive definite matrix
    and :math:`c` is a positive constant added to the denominator.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      n_samples : ``int``
          Number of samples to draw.

      const : ``torch.Tensor``
          Constant added to the denominator. Shape is ``()``.

      B : ``torch.Tensor``, optional
          Matrix B used in the denominator of the projection. If not provided,
          the identity matrix is used. Shape is ``(n_dim, n_dim)``.

      B_chol : ``torch.Tensor``, optional
          Cholesky decomposition matrix L, such that B = LL'.
          Can be provided to avoid recomputing it. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          Samples from the projected normal. Shape is ``(n_samples, n_dim)``.
    """
    if B_chol is None:
        if B is None:
            raise ValueError("Either B or B_chol must be provided.")
        B_chol = torch.linalg.cholesky(B)

    # Change basis to make B the identity
    mean_z = B_chol.T @ mean_x
    covariance_z = B_chol.T @ covariance_x @ B_chol

    # Sample from the standard projected normal
    samples_prnorm_z = pnc_formulas.sample(
      mean_x=mean_z,
      covariance_x=covariance_z,
      n_samples=n_samples,
      const=const
    )

    # Change basis back to the original space
    samples_prnorm = torch.linalg.solve_triangular(B_chol.T, samples_prnorm_z.T, upper=True).T
    return samples_prnorm


def empirical_moments(mean_x, covariance_x, const, n_samples, B=None, B_chol=None):
    r"""
    Compute the mean, covariance and second moment of the variable
    :math:`y = x/\sqrt{x^T B x + c}`
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`,
    :math:`B` is a symmetric positive definite matrix, and
    :math:`c` is a positive constant added to the denominator,
    by sampling from the distribution.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      n_samples : ``int``
          Number of samples to draw.

      const : ``torch.Tensor``
          Constant added to the denominator. Shape is ``()``.

      B : ``torch.Tensor``, optional
          Matrix B used in the denominator of the projection. If not provided,
          the identity matrix is used. Shape is ``(n_dim, n_dim)``.

      B_chol : ``torch.Tensor``, optional
          Cholesky decomposition matrix L, such that B = LL'.
          Can be provided to avoid recomputing it. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``dict``
          Dictionary with the keys ``mean``, ``covariance``, and ``second_moment``,
          containing the empirical moments of the projected normal distribution.
    """
    if B_chol is None:
        if B is None:
            raise ValueError("Either B or B_chol must be provided.")
        B_chol = torch.linalg.cholesky(B)

    # Change basis to make B the identity
    mean_z = B_chol.T @ mean_x
    covariance_z = B_chol.T @ covariance_x @ B_chol

    moment_dict_z = pnc_formulas.empirical_moments(
      mean_x=mean_z,
      covariance_x=covariance_z,
      n_samples=n_samples,
      const=const
    )

    # Change basis back to the original space
    B_chol_inv = torch.linalg.solve_triangular(B_chol, torch.eye(B_chol.shape[0]), upper=False)
    moment_dict = {}
    moment_dict["mean"] = B_chol_inv.T @ moment_dict_z["mean"]
    moment_dict["covariance"] = B_chol_inv.T @ moment_dict_z["covariance"] @ B_chol_inv
    moment_dict["second_moment"] = (
      B_chol_inv.T @ moment_dict_z["second_moment"] @ B_chol_inv
    )
    return moment_dict
