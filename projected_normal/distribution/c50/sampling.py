"""Sampling functions for the general projected normal distribution with an additive constant c50 in the denominator."""
import torch
import torch.distributions.multivariate_normal as mvn


__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, n_samples, c50=0):
    """
    Sample from the variable Y = X/(X'X + c50)^0.5, where X~N(mean_x, covariance_x).
    The variable Y has a general projected normal distribution with an extra
    constant c50 added to the denominator.

    Parameters:
    -----------------
      - mean_x : Mean of X. (n_dim)
      - covariance_x : Covariance matrix of X. (n_dim x n_dim)
      - n_samples : Number of samples.
      - c50 : Constant added to the denominator. Scalar

    Returns:
    -----------------
      Samples from the general projected normal. (n_samples x n_dim)
    """
    # Initialize Gaussian distribution to sample from
    dist = mvn.MultivariateNormal(loc=mean_x, covariance_matrix=covariance_x)
    # Take n_samples
    X = dist.sample([n_samples])
    q = torch.sqrt(torch.einsum("ni,in->n", X, X.t()) + c50)
    # Normalize normal distribution samples
    samples_prnorm = torch.einsum("ni,n->ni", X, 1 / q)
    return samples_prnorm


def empirical_moments(mean_x, covariance_x, n_samples, c50=0):
    """
    Compute the mean, covariance and second moment of the variable
    Y = X/(X'X + c50)^0.5, where X~N(mean_x, covariance_x), by sampling from the
    distribution. The variable Y has a projected normal distribution with an extra
    constant c50 added to the denominator.

    Parameters:
    -----------------
      - mean_x : Mean. (n_dim)
      - covariance_x : Covariance matrix. (n_dim x n_dim)
      - n_samples : Number of samples.
      - c50 : Constant added to the denominator. Scalar

    Returns:
    -----------------
      Dictionary with the following keys:
      - gamma : Mean of the projected normal. (n_dim)
      - psi : Covariance of the projected normal. (n_dim x n_dim)
      - second_moment : Second moment of the projected normal. (n_dim x n_dim)
    """
    samples = sample(mean_x, covariance_x, n_samples=n_samples, c50=c50)
    gamma = torch.mean(samples, dim=0)
    second_moment = torch.einsum("in,nj->ij", samples.t(), samples) / n_samples
    psi = second_moment - torch.einsum("i,j->ij", gamma, gamma)
    return {"mean": gamma, "covariance": psi, "second_moment": second_moment}
