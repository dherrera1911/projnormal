"""Sampling functions for the general projected normal distribution."""

from .. import projected_normal_Bc as pnbc_formulas

__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, n_samples, B=None, B_chol=None):
    r"""
    Sample the variable :math:`y = x/\sqrt{x^T B x}`
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`B` is a symmetric positive definite matrix.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      n_samples : ``int``
          Number of samples to draw.

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
    return pnbc_formulas.sample(mean_x=mean_x, covariance_x=covariance_x,
                                 n_samples=n_samples, const=0, B=B, B_chol=B_chol)


def empirical_moments(mean_x, covariance_x, n_samples, B=None, B_chol=None):
    r"""
    Compute the mean, covariance and second moment of the variable
    :math:`y = x/\sqrt{x^T B x}`
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`B` is a symmetric positive definite matrix, by sampling from the
    distribution.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      n_samples : ``int``
          Number of samples to draw.

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
    return pnbc_formulas.empirical_moments(
      mean_x=mean_x, covariance_x=covariance_x, n_samples=n_samples,
      const=0, B=B, B_chol=B_chol
    )
