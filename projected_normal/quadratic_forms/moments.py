"""Moments of quadratic forms of multidimensional Gaussian distributions."""

import torch
import torch.special as spt
import scipy.special as sps
from projected_normal.auxiliary import (
    product_trace,
    product_trace4,
)

__all__ = [
  "mean",
  "variance",
  "qf_covariance",
  "qf_linear_covariance",
]


def __dir__():
    return __all__


def mean(mu, covariance, M):
    """
    Compute the mean of X'MX where X~N(mu, covariance).

    Parameters
    ----------------
      - mu: Mean of normal distribution X. (n_dim)
      - covariance: Covariance of the normal distribution (n_dim x n_dim)
      - M: Matrix to multiply by. (n_dim x n_dim). If None, M=I.

    Returns:
    ----------------
      Expected value of the quadratic form (scalar)
    """
    if M.dim() == 1:
        gamma = _mean_diagonal(mu, covariance, M)
    else:
        trace = product_trace(M, covariance)
        mu_quadratic = torch.einsum("d,db,b->", mu, M, mu)
        gamma = trace + mu_quadratic
    return gamma


def _mean_diagonal(mu, covariance, M_diagonal):
    """
    Compute the mean of X'MX where X~N(mu, covariance) and M is diagonal.

    Parameters
    ----------------
      - mu: Mean of normal distribution X. (n_dim)
      - covariance: Covariance of the normal distribution (n_dim x n_dim)
      - M_diagonal: Diagonal elements of M. (n_dim)

    Returns:
    ----------------
      Expected value of the quadratic form.
    """
    trace = torch.einsum("ii,i->", covariance, M_diagonal)
    mu_quadratic = torch.einsum("i,i,i->", mu, M_diagonal, mu)
    gamma_quadratic = trace + mu_quadratic
    return gamma_quadratic


def variance(mu, covariance, M):
    """
    Compute the variance of X'MX, where X~N(mu, covariance).

    Parameters
    ----------------
      - mu: Mean of normally distributed X. (n_dim)
      - covariance: Covariance of the normal distribution (n_dim x n_dim)
      - M: Matrix to multiply by. (n_dim x n_dim)

    Returns
    ----------------
      Variance of quadratic form.
    """
    if M.dim() == 1:
        psi_qf = _variance_diagonal(mu, covariance, M)
    else:
        # Compute the trace of M*covariance*M*covariance
        trace = product_trace4(A=M, B=covariance, C=M, D=covariance)
        # Compute the quadratic form term
        mean_qf = torch.einsum("d,db,bk,km,m->", mu, M, covariance, M, mu)
        # Add terms
        psi_qf = 2 * trace + 4 * mean_qf
    return psi_qf


def _variance_diagonal(mu, covariance, M_diagonal):
    """
    Computes the variance of the quadratic form given
    by Gaussian variable X and matrix M, where X~N(mu, covariance)
    and M is diagonal.

    Parameters
    ----------------
      - mu: Mean of normally distributed X. (n_dim)
      - covariance: Covariance of the normal distribution (n_dim x n_dim)
      - M_diagonal: Diagonal elements of M. (n_dim)

    Returns
    ----------------
      Variance of quadratic form. Scalar
    """
    trace = torch.einsum("i,ij,j,ji->", M_diagonal, covariance, M_diagonal, covariance)
    mean_qf = torch.einsum(
        "d,d,dk,k,k->", mu, M_diagonal, covariance, M_diagonal, mu
    )
    psi_qf = 2 * trace + 4 * mean_qf
    return psi_qf


def qf_covariance(mu, covariance, M, M2):
    """
    Compute the covariance of X'MX and X'M2X, where X ~ N(mu, covariance).

    Parameters
    ----------------
      - mu: Mean of normal distributions X. (n_dim)
      - covariance: Covariance of the normal distributions (n_dim x n_dim)
      - M: Matrix of quadratic form 1. (n_dim x n_dim)
      - M2: Matrix of quadratic form 2. (n_dim x n_dim)

    Returns
    ----------------
      Covariance of X'MX and X'M2X. Scalar
    """
    covariance = torch.as_tensor(covariance)
    # Compute the trace of M*covariance*M2*covariance
    if covariance.dim() == 2:
        trace = product_trace4(A=M, B=covariance, C=M2, D=covariance)
    elif covariance.dim() == 0:  # Isotropic case
        trace = product_trace(A=M, B=M2) * covariance**2
    # Compute mean term
    mean_term = torch.einsum("d,db,bk,km,m->", mu, M, covariance, M2, mu)
    # Add terms
    cov_quadratic = 2 * trace + 4 * mean_term
    return cov_quadratic


def qf_linear_covariance(mu, covariance, M, b):
    """
    Compute the covariance of X'MX and X'b, where X ~ N(mu, covariance).

    Parameters
    ----------------
      - mu: Means of normal distribution X. (n_dim)
      - covariance: Covariance of the normal distribution (n_dim x n_dim)
      - M: Matrix to multiply by. (n_dim x n_dim)
      - b: Vector for linear form. (n_dim)

    Returns
    ----------------
      Covariance of X'MX and X'b. Scalar
    """
    cov_quadratic = 2 * torch.einsum("i,ij,jk,k->", mu, M, covariance, b)
    return cov_quadratic


def non_central_x2_moments(mu, sigma, s):
    """
    Compute the s-th moment of y=X'X where X ~ N(mu, sigma^2 I) (y is a non-central chi-square distribution).

    Parameters
    ----------------
      - mu: Multidimensional mean of the gaussian. (n_dim)
      - sigma: Standard deviation of isotropic noise. (Scalar)
      - s: Order of the moment to compute.

    Returns
    ----------------
      s-th moment of X'X. (Scalar)
    """
    n_dim = torch.as_tensor(len(mu))
    non_centrality = torch.norm(mu / sigma, dim=-1) ** 2
    if s == 1:
        moment = (non_centrality + n_dim) * sigma**2
    elif s == 2:
        moment = (
          n_dim**2 + 2 * n_dim + 4 * non_centrality + non_centrality**2 + 2 * n_dim * non_centrality
        ) * sigma**4
    else:
        # Get gamma and hyp1f1 values
        hyp_val = sps.hyp1f1(n_dim / 2 + s, n_dim / 2, non_centrality / 2)
        gammaln1 = spt.gammaln(n_dim / 2 + s)
        gammaln2 = spt.gammaln(n_dim / 2)
        gamma_ratio = (
          2**s / torch.exp(non_centrality / 2)
        ) * torch.exp(gammaln1 - gammaln2)
        moment = (gamma_ratio * hyp_val) * (sigma ** (s * 2))
    return moment


def inverse_non_central_x_mean(mu, sigma):
    """
    Compute the expected value of 1/||X|| where X ~ N(mu, sigma^2 I) (||X|| has a non-chentral chi distribution).

    Parameters
    ----------------
      - mu: Mean of the gaussian. (n_dim)
      - sigma: Standard deviation of isotropic noise. (Scalar)

    Returns
    ----------------
      Expected value of 1/||X||
    """
    n_dim = torch.as_tensor(len(mu))
    non_centrality = torch.norm(mu / sigma, dim=-1) ** 2
    # Corresponding hypergeometric function values
    hyp_val = sps.hyp1f1(1 / 2, n_dim / 2, - non_centrality / 2)
    gammaln1 = spt.gammaln((n_dim - 1) / 2)
    gammaln2 = spt.gammaln(n_dim / 2)
    gamma_ratio = (
      1 / torch.sqrt(torch.as_tensor(2))
    ) * torch.exp(gammaln1 - gammaln2)
    gamma_invncx = (gamma_ratio * hyp_val) / sigma  # This is a torch tensor
    return gamma_invncx

