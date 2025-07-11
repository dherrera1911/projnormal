"""Sampling of quadratic forms of multivariate Gaussian random variables."""
import torch
import torch.distributions.multivariate_normal as mvn

__all__ = [
  "sample",
  "empirical_moments",
  "empirical_covariance",
]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, M, n_samples):
    r"""
    Sample from the quadratic form :math:`x^T M x`,
    where :math:`x` follows a multivariate normal distribution
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

      n_samples : ``int``
          Number of samples to generate.

    Returns
    -------
      ``torch.Tensor``
          Samples from the quadratic form. Shape is ``(n_samples, n_dim)``.
    """
    dist = mvn.MultivariateNormal(
      loc=mean_x, covariance_matrix=covariance_x
    )
    X = dist.sample([n_samples])
    if M.dim() == 1:
        samples_qf = torch.einsum("ni,i,in->n", X, M, X.t())
    else:
        samples_qf = torch.einsum("ni,ij,jn->n", X, M, X.t())
    return samples_qf


def empirical_moments(mean_x, covariance_x, M, n_samples):
    r"""
    Compute an empirical approximation to the moments of the quadratic form
    :math:`x^T M x`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape ``(n_dim)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape ``(n_dim, n_dim)`` or scalar (isotropic covariance).

      M : ``torch.Tensor``, optional
          Matrix in quadratic form.  If a vector is provided, it is used as the diagonal of `M`.
          Default is the identity matrix. Shape ``(n_dim, n_dim)`` or ``(n_dim,)``.

      n_samples: ``int``
          Number of samples to use.

    Returns
    -------
      ``dict``
          Dictionary with fields ``mean``, ``var``, and ``second_moment``.
    """
    samples_qf = sample(mean_x, covariance_x, M, n_samples)
    mean = torch.mean(samples_qf)
    var = torch.var(samples_qf)
    second_moment = torch.mean(samples_qf**2)
    return {"mean": mean, "var": var, "second_moment": second_moment}


def empirical_covariance(mean_x, covariance_x, M1, M2, n_samples):
    """
    Compute an empirical approximation to the covariance between
    two quadratic forms X'MX and X'MX, where X~N(mean_x, covariance_x).

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape ``(n_dim)``.

      covariance_x : ``torch.Tensor``
          Covariance of `x`. Shape ``(n_dim, n_dim)`` or scalar (isotropic covariance).

      M1 : ``torch.Tensor``, optional
          Matrix in quadratic form.  If a vector is provided, it is used as the diagonal of `M`.
          Default is the identity matrix. Shape ``(n_dim, n_dim)``.

      M2 : ``torch.Tensor``, optional
          Matrix in quadratic form.  If a vector is provided, it is used as the diagonal of `M`.
          Default is the identity matrix. Shape ``(n_dim, n_dim)``.

      n_samples: ``int``
          Number of samples to generate use.

    Returns
    -------
      ``torch.Tensor``
        Covariance between the two quadratic forms. Shape is ``(1,)``.
    """
    dist = mvn.MultivariateNormal(loc=mean_x, covariance_matrix=covariance_x)
    X = dist.sample([n_samples])
    qf1 = torch.einsum("ni,ij,jn->n", X, M1, X.t())
    qf2 = torch.einsum("ni,ij,jn->n", X, M2, X.t())
    cov = torch.cov(torch.cat((qf1.unsqueeze(0), qf2.unsqueeze(0))))[0, 1]
    return cov
