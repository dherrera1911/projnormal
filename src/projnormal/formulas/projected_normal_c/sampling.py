"""Sampling functions for the general projected normal distribution with an additive constant const in the denominator."""
import torch
import torch.distributions.multivariate_normal as mvn

__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, const, n_samples):
    r"""
    Sample the variable :math:`y = x/\sqrt{x^T x + c}`
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`c` is a constant added to the denominator.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      const : ``torch.Tensor``
          Constant added to the denominator. Shape is ``()``.

      n_samples : ``int``
          Number of samples to draw.

    Returns
    -------
      ``torch.Tensor``
          Samples from the distribution. Shape is ``(n_samples, n_dim)``.
    """
    # Initialize Gaussian distribution to sample from
    dist = mvn.MultivariateNormal(loc=mean_x, covariance_matrix=covariance_x)
    # Take n_samples
    X = dist.sample([n_samples])
    q = torch.sqrt(torch.einsum("ni,in->n", X, X.t()) + const)
    # Normalize normal distribution samples
    samples_prnorm = torch.einsum("ni,n->ni", X, 1 / q)
    return samples_prnorm


def empirical_moments(mean_x, covariance_x, const, n_samples):
    r"""
    Compute the mean, covariance and second moment of the variable
    :math:`y = x/\sqrt{x^T x + c}`
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)` and
    :math:`c` is a positive constant added to the denominator,
    by sampling from the distribution.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      const : ``torch.Tensor``
          Constant added to the denominator. Shape is ``()``.

      n_samples : ``int``
          Number of samples to draw.

    Returns
    -------
      ``dict``
          Dictionary with the keys ``mean``, ``covariance``, and ``second_moment``,
          containing the empirical moments of the projected normal distribution.
    """
    samples = sample(mean_x, covariance_x, n_samples=n_samples, const=const)
    gamma = torch.mean(samples, dim=0)
    psi = torch.cov(samples.t())
    second_moment = psi + torch.outer(gamma, gamma)
    return {"mean": gamma, "covariance": psi, "second_moment": second_moment}
