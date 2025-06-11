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


def sample(mean_x, covariance_x, M, n_samples):
    """
    Sample from the quadratic form X'MX, where X~N(mean_x, covariance_x).

    Parameters
    -----------------
      mean_x : torch.Tensor, shape (n_dim,)
        Mean of normal distribution X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance of the normal distribution.

      M: torch.Tensor, shape (n_dim, n_dim).
        Matrix of the quadratic form.

      n_samples: int
        Number of samples to generate.

    Returns
    -----------------
      torch.Tensor, shape (n_samples,)
          Samples from the quadratic form.
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
    """
    Compute an empirical approximation to the moments of X'MX for X~N(mean_x, covariance_x).

    Parameters
    -----------------
      mean_x : torch.Tensor, shape (n_dim,)
        Mean of normal distribution X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance of the normal distribution.

      M: torch.Tensor, shape (n_dim, n_dim).
        Matrix of the quadratic form.

      n_samples: int
        Number of samples to use.

    Returns
    -----------------
      dict
        Dictionary with fields
          - "mean": torch.Tensor, shape ()
          - "var": torch.Tensor, shape ()
          - "second_moment": torch.Tensor, shape ()
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
    -----------------
      mean_x : torch.Tensor, shape (n_dim,)
        Mean of normal distribution X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance of the normal distribution.

      M1: torch.Tensor, shape (n_dim, n_dim).
        Matrix of the first quadratic form.

      M2: torch.Tensor, shape (n_dim, n_dim).
        Matrix of the second quadratic form.

      n_samples: int
        Number of samples to generate use.

    Returns
    -----------------
      torch.Tensor, shape ()
        Covariance between the two quadratic forms.
    """
    dist = mvn.MultivariateNormal(loc=mean_x, covariance_matrix=covariance_x)
    X = dist.sample([n_samples])
    qf1 = torch.einsum("ni,ij,jn->n", X, M1, X.t())
    qf2 = torch.einsum("ni,ij,jn->n", X, M2, X.t())
    cov = torch.cov(torch.cat((qf1.unsqueeze(0), qf2.unsqueeze(0))))[0, 1]
    return cov
