"""Moments of quadratic forms of multidimensional Gaussian distributions."""

import torch
import torch.special as spt
import scipy.special as sps

__all__ = [
  "mean",
  "variance",
  "qf_covariance",
  "qf_linear_covariance",
]


def __dir__():
    return __all__


def mean(mean_x, covariance_x, M=None):
    """
    Compute the mean of X'MX where X~N(mean_x, covariance_x).

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
        Mean of normal distribution X. 

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance of the normal distribution. 

      M: torch.Tensor, shape (n_dim, n_dim) or (n_dim,), optional
        Matrix to multiply by. Can either be a matrix or a vector,
        in which case it is assumed to be diagonal. If None, M=I.

    Returns:
    ----------------
      torch.Tensor, shape ()
        Expected value of the quadratic form
    """
    if M is None:
        M = torch.ones(
          len(mean_x), dtype=mean_x.dtype, device=mean_x.device
        )
    if M.dim() == 1:
        mean_quadratic = _mean_diagonal(mean_x, covariance_x, M)
    else:
        term1 = _product_trace(M, covariance_x)
        term2 = torch.einsum("d,db,b->", mean_x, M, mean_x)
        mean_quadratic = term1 + term2
    return mean_quadratic


def _mean_diagonal(mean_x, covariance_x, M_diagonal):
    """
    Compute the mean of X'MX where X~N(mean_x, covariance_x) and M is diagonal.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
        Mean of normal distribution X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance of the normal distribution.

      M: torch.Tensor, shape (n_dim,), optional
        Diagonal elements of the diagonal matrix to multiply by.

    Returns:
    ----------------
      torch.Tensor, shape ()
        Expected value of the quadratic form.
    """
    term1 = torch.einsum("ii,i->", covariance_x, M_diagonal)
    term2 = torch.einsum("i,i,i->", mean_x, M_diagonal, mean_x)
    mean_quadratic = term1 + term2
    return mean_quadratic


def variance(mean_x, covariance_x, M=None):
    """
    Compute the variance of X'MX, where X~N(mean_x, covariance_x).

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
        Mean of normal distribution X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance of the normal distribution.

      M: torch.Tensor, shape (n_dim, n_dim) or (n_dim,), optional
        Matrix to multiply by. Can either be a matrix or a vector,
        in which case it is assumed to be diagonal. If None, M=I.

    Returns
    ----------------
      torch.Tensor, shape ()
        Variance of quadratic form.
    """
    if M is None:
        M = torch.ones(
          len(mean_x), dtype=mean_x.dtype, device=mean_x.device
        )
    if M.dim() == 1:
        psi_qf = _variance_diagonal(mean_x, covariance_x, M)
    else:
        # Compute the trace of M*covariance_x*M*covariance_x
        trace = _product_trace4(A=M, B=covariance_x, C=M, D=covariance_x)
        # Compute the quadratic form term
        mean_qf = torch.einsum(
          "d,db,bk,km,m->", mean_x, M, covariance_x, M, mean_x
        )
        # Add terms
        psi_qf = 2 * trace + 4 * mean_qf
    return psi_qf


def _variance_diagonal(mean_x, covariance_x, M_diagonal):
    """
    Computes the variance of the quadratic form given
    by Gaussian variable X and matrix M, where X~N(mean_x, covariance_x)
    and M is diagonal.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
        Mean of normal distribution X. 

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance of the normal distribution. 

      M_diagonal: torch.Tensor, shape (n_dim,).
        Diagonal elements of the diagonal matrix to multiply by.

    Returns
    ----------------
      torch.Tensor, shape ()
        Variance of quadratic form.
    """
    trace = torch.einsum("i,ij,j,ji->", M_diagonal, covariance_x, M_diagonal, covariance_x)
    mean_qf = torch.einsum(
        "d,d,dk,k,k->", mean_x, M_diagonal, covariance_x, M_diagonal, mean_x
    )
    psi_qf = 2 * trace + 4 * mean_qf
    return psi_qf


def qf_covariance(mean_x, covariance_x, M, M2):
    """
    Compute the covariance of X'MX and X'M2X, where X ~ N(mean_x, covariance_x).

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
        Mean of normal distribution X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance of the normal distribution.

      M: torch.Tensor, shape (n_dim, n_dim)
        Matrix of first quadratic form.

      M2: torch.Tensor, shape (n_dim, n_dim)
        Matrix of second quadratic form.

    Returns
    ----------------
      torch.Tensor, shape ()
        Covariance of X'MX and X'M2X. Scalar
    """
    # Compute the trace of M*covariance*M2*covariance
    if covariance_x.dim() == 2:
        trace = _product_trace4(A=M, B=covariance_x, C=M2, D=covariance_x)
    elif covariance_x.dim() == 0:  # Isotropic case
        trace = _product_trace(A=M, B=M2) * covariance_x**2
    # Compute mean term
    mean_term = torch.einsum("d,db,bk,km,m->", mean_x, M, covariance_x, M2, mean_x)
    # Add terms
    cov_quadratic = 2 * trace + 4 * mean_term
    return cov_quadratic


def qf_linear_covariance(mean_x, covariance_x, M, b):
    """
    Compute the covariance of X'MX and X'b, where X ~ N(mean_x, covariance_x).

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
        Mean of normal distribution X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance of the normal distribution.

      M: torch.Tensor, shape (n_dim, n_dim)
        Matrix of first quadratic form.

      b: torch.Tensor, shape (n_dim,)
        Vector for linear form.

    Returns
    ----------------
      torch.Tensor, shape ()
        Covariance of X'MX and X'b.
    """
    cov_quadratic = 2 * torch.einsum("i,ij,jk,k->", mean_x, M, covariance_x, b)
    return cov_quadratic


def non_central_x2_moments(mean_x, sigma_x, s):
    """
    Compute the s-th moment of y=X'X where X ~ N(mean_x, sigma_X^2 I) (y is a non-central chi-square distribution).

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
        Mean of normal distribution X.

      sigma_x : torch.Tensor, shape ()
        Standard deviation of isotropic noise.

      s : int
        Order of the moment to compute.

    Returns
    ----------------
      torch.Tensor, shape ()
          s-th moment of X'X.
    """
    n_dim = torch.as_tensor(len(mean_x))
    non_centrality = torch.norm(mean_x / sigma_x, dim=-1) ** 2
    if s == 1:
        moment = (non_centrality + n_dim) * sigma_x**2
    elif s == 2:
        moment = (
          n_dim**2 + 2 * n_dim + 4 * non_centrality + non_centrality**2 + 2 * n_dim * non_centrality
        ) * sigma_x**4
    else:
        # Get gamma and hyp1f1 values
        hyp_val = sps.hyp1f1(n_dim / 2 + s, n_dim / 2, non_centrality / 2)
        gammaln1 = spt.gammaln(n_dim / 2 + s)
        gammaln2 = spt.gammaln(n_dim / 2)
        gamma_ratio = (
          2**s / torch.exp(non_centrality / 2)
        ) * torch.exp(gammaln1 - gammaln2)
        moment = (gamma_ratio * hyp_val) * (sigma_x ** (s * 2))
    return moment


def inverse_non_central_x_mean(mean_x, sigma_x):
    """
    Compute the expected value of 1/||X|| where X ~ N(mean_x, sigma_x^2 I) (||X|| has a non-chentral chi distribution).

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
        Mean of normal distribution X.

      sigma_x : torch.Tensor, shape ()
        Standard deviation of isotropic noise.

    Returns
    ----------------
      torch.Tensor, shape ()
        Expected value of 1/||X||
    """
    n_dim = torch.as_tensor(len(mean_x))
    non_centrality = torch.norm(mean_x / sigma_x, dim=-1) ** 2
    # Corresponding hypergeometric function values
    hyp_val = sps.hyp1f1(1 / 2, n_dim / 2, - non_centrality / 2)
    gammaln1 = spt.gammaln((n_dim - 1) / 2)
    gammaln2 = spt.gammaln(n_dim / 2)
    gamma_ratio = (
      1 / torch.sqrt(torch.as_tensor(2))
    ) * torch.exp(gammaln1 - gammaln2)
    gamma_invncx = (gamma_ratio * hyp_val) / sigma_x  # This is a torch tensor
    return gamma_invncx


def _product_trace(A, B):
    """
    Efficiently compute tr(A*B).
    """
    return torch.einsum("ij,ji->", A, B)


def _product_trace4(A, B, C, D):
    """
    Efficiently compute tr(A*B*C*D).
    """
    return torch.einsum("ij,jk,kl,li->", A, B, C, D)
