"""Sampling functions for the general projected normal distribution."""
from .. import projected_normal_c as pnc_formulas

__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, n_samples):
    r"""
    Sample the variable :math:`y = x/\sqrt{x^T x}`
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      n_samples : ``int``
          Number of samples.

    Returns
    -------
      ``torch.Tensor``
          Samples from the projected normal. Shape is ``(n_samples, n_dim)``.
    """
    return pnc_formulas.sample(
      mean_x=mean_x,
      covariance_x=covariance_x,
      n_samples=n_samples,
      const=0
    )


def empirical_moments(mean_x, covariance_x, n_samples):
    r"""
    Compute the mean, covariance and second moment of the variable
    :math:`y = x/\sqrt{x^T x}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`,
    by sampling from the distribution.
    The variable :math:`y` has a general projected normal distribution.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      n_samples : ``int``
          Number of samples.

    Returns
    -------
      ``dict``
          Dictionary with the keys ``mean``, ``covariance``, and ``second_moment``,
          containing the empirical moments of the projected normal distribution.
    """
    return pnc_formulas.empirical_moments(
      mean_x=mean_x,
      covariance_x=covariance_x,
      n_samples=n_samples,
      const=0
    )
