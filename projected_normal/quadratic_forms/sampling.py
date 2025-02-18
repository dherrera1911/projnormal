"""Moments of quadratic forms of multidimensional Gaussian distributions."""
import torch
import torch.distributions.multivariate_normal as mvn

__all__ = [
  "sample",
  "empirical_moments",
  "empirical_covariance",
]


def __dir__():
    return __all__


def sample(mu, covariance, M, n_samples):
    """
    Sample from the quadratic form X'MX, where X~N(mu, covariance).

    Parameters
    -----------------
      - mu: Mean. (n_dim)
      - covariance: Covariance matrix. (n_dim x n_dim)
      - M: Matrix of the quadratic form. (n_dim x n_dim)
      - n_samples: Number of samples.

    Returns
    -----------------
      Samples from the quadratic form. (n_samples)
    """
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    X = dist.sample([n_samples])
    if M.dim() == 1:
        samples_qf = torch.einsum("ni,i,in->n", X, M, X.t())
    else:
        samples_qf = torch.einsum("ni,ij,jn->n", X, M, X.t())
    return samples_qf


def empirical_moments(mu, covariance, M, n_samples):
    """
    Compute an empirical approximation to the moments of X'MX for X~N(mu, covariance).

    Parameters
    -----------------
      - mu: Mean. (n_dim)
      - covariance: Covariance matrix. (n_dim x n_dim)
      - M: Matrix of the quadratic form. (n_dim x n_dim)
      - n_samples: Number of samples.

    Returns
    -----------------
      - statsDict: Dictionary with the mean, variance and second
          moment of the quadratic form
    """
    samples_qf = sample(mu, covariance, M, n_samples)
    mean = torch.mean(samples_qf)
    var = torch.var(samples_qf)
    second_moment = torch.mean(samples_qf**2)
    return {"mean": mean, "var": var, "second_moment": second_moment}


def empirical_covariance(mu, covariance, M1, M2, n_samples):
    """
    Compute an empirical approximation to the covariance between two quadratic forms
    X'MX and X'MX, where X~N(mu, covariance).

    Parameters
    -----------------
      - mu: Mean vector of the Gaussian. (n_dim)
      - covariance: Covariance matrix of the Gaussian. (n_dim x n_dim)
      - M1: Matrix of the first quadratic form. (n_dim x n_dim)
      - M2: Matrix of the second quadratic form. (n_dim x n_dim)
      - n_samples: Number of samples to use to compute the moments.

    Returns
    -----------------
      - Covariance between the two quadratic forms. (Scalar)
    """
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    X = dist.sample([n_samples])
    qf1 = torch.einsum("ni,ij,jn->n", X, M1, X.t())
    qf2 = torch.einsum("ni,ij,jn->n", X, M2, X.t())
    cov = torch.cov(torch.cat((qf1.unsqueeze(0), qf2.unsqueeze(0))))[0, 1]
    return cov
