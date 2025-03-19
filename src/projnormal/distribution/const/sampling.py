"""Sampling functions for the general projected normal distribution with an additive constant const in the denominator."""
import torch
import torch.distributions.multivariate_normal as mvn


__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, const, n_samples):
    """
    Sample from the variable Y = X/(X'X + const)^0.5, where X~N(mean_x, covariance_x).
    The variable Y has a general projected normal distribution with an extra
    constant const added to the denominator.

    Parameters:
    -----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
          Covariance matrix of X elements.

      const : torch.Tensor, shape ()
          Constant added to the denominator. Must be >=0.

      n_samples : int
          Number of samples.

    Returns:
    -----------------
      torch.Tensor, shape (n_samples, n_dim)
          Samples from the projected normal.
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
    """
    Compute the mean, covariance and second moment of the variable
    Y = X/(X'X + const)^0.5, where X~N(mean_x, covariance_x), by sampling from the
    distribution. The variable Y has a projected normal distribution with an extra
    constant const added to the denominator.

    Parameters:
    -----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
          Covariance matrix of X elements.

      const : torch.Tensor, shape ()
          Constant added to the denominator. Must be >=0.

      n_samples : int
          Number of samples.

    Returns:
    -----------------
      dict
          Dictionary with the following keys and values
            'mean' : torch.Tensor, shape (n_dim,)
                Mean of the projected normal.
            'covariance' : torch.Tensor, shape (n_dim, n_dim)
                Covariance of the projected normal.
            'second_moment' : torch.Tensor, shape (n_dim, n_dim)
                Second moment of the projected normal.
    """
    samples = sample(mean_x, covariance_x, n_samples=n_samples, const=const)
    gamma = torch.mean(samples, dim=0)
    psi = torch.cov(samples.t())
    second_moment = psi + torch.outer(gamma, gamma)
    return {"mean": gamma, "covariance": psi, "second_moment": second_moment}
