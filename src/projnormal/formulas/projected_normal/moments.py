"""Approximation to the moments of the general projected normal distribution."""
from .. import projected_normal_c as pnc_formulas

__all__ = ["mean", "second_moment"]


def __dir__():
    return __all__


def mean(mean_x, covariance_x):
    r"""
    Compute the mean of :math:`y = x/\sqrt{x^T x}`,
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)` via
    Taylor approximation. (:math:`y` has a projected normal distribution.).

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          Expected value for the projected normal. Shape is ``(n_dim,)``.
    """
    return pnc_formulas.mean(mean_x, covariance_x, const=0)


def second_moment(mean_x, covariance_x):
    r"""
    Compute the second moment matrix of :math:`y = x/\sqrt{x^T x}`,
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)` via Taylor approximation.
    (:math:`y` has a projected normal distribution.).

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          Second moment matrix of :math:`y`. Shape is ``(n_dim, n_dim)``.
    """
    return pnc_formulas.second_moment(mean_x, covariance_x, const=0)
