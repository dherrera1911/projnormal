##################################
##################################
#
## QUADRATIC FORMS MOMENTS AND SAMPLING
#
##################################
##################################

#### Terminology:
# X: Random variable with multidimensional Gaussian distribution
# mu: Mean of X
# sigma: Standard deviation of X (if isotropic)
# covariance: Covariance of X
# nc: non-centrality parameter
# n_dim: Dimensions of X
# gamma: Expected value of a function of X
# psi: Variance of a function of X


import numpy as np
import torch
from torch.special import gammaln
import scipy.special as sps
import torch.distributions.multivariate_normal as mvn
from projected_normal.auxiliary import (
    is_diagonal,
    product_trace,
    product_trace4,
    non_centrality,
)


##################################
##################################
#
## QUADRATIC FORMS MOMENTS
#
##################################
##################################


def quadratic_form_mean(mu, covariance, M):
    """
    Computes the mean of the quadratic form given
    by Gaussian variable X and matrix M, where X~N(mu, covariance).
    ----------------
    Arguments:
    ----------------
      - mu: Mean of normal distribution X. (n_dim)
      - covariance: Covariance of the normal distribution (n_dim x n_dim)
      - M: Matrix to multiply by. (n_dim x n_dim). If None, M=I.
    ----------------
    Outputs:
    ----------------
      - gamma_quadratic: Expected value of the quadratic form.
    """
    trace = product_trace(M, covariance)
    mu_quadratic = torch.einsum("d,db,b->", mu, M, mu)

    # Add terms
    gamma_quadratic = trace + mu_quadratic
    return gamma_quadratic


def quadratic_form_mean_diagonal(mu, covariance, M_diagonal):
    """
    Computes the mean of the quadratic form in an efficient
    way for diagonal matrices.
    ----------------
    Arguments:
    ----------------
      - mu: Mean of normal distribution X. (n_dim)
      - covariance: Covariance of the normal distribution (n_dim x n_dim)
      - diagonal_weights: Diagonal elements of M. (n_dim)
    ----------------
    Outputs:
    ----------------
      - gamma_quadratic: Expected value of the quadratic form.
    """
    trace = torch.einsum("ii,i->", covariance, M_diagonal)
    mu_quadratic = torch.einsum("i,i,i->", mu, M_diagonal, mu)

    # Add terms
    gamma_quadratic = trace + mu_quadratic
    return gamma_quadratic


def quadratic_form_var(mu, covariance, M):
    """
    Computes the variance of the quadratic form given
    by Gaussian variable X and matrix M, where X~N(mu, covariance).
    ----------------
    Arguments:
    ----------------
      - mu: Mean of normally distributed X. (n_dim)
      - covariance: Covariance of the normal distribution (n_dim x n_dim)
      - M: Matrix to multiply by. (n_dim x n_dim)
    ----------------
    Outputs:
    ----------------
      - psi_quadratic: Variance of quadratic form.
    """
    # Compute the trace of M*covariance*M*covariance
    trace = product_trace4(A=M, B=covariance, C=M, D=covariance)

    # Compute the quadratic form term
    mu_quadratic = torch.einsum("d,db,bk,km,m->", mu, M, covariance, M, mu)

    # Add terms
    psi_quadratic = 2 * trace + 4 * mu_quadratic
    return psi_quadratic


def quadratic_form_var_diagonal(mu, covariance, M_diagonal):
    """
    Same as quadratic_form_var but for diagonal matrix M.
    ----------------
    Arguments:
    ----------------
      - mu: Mean of normally distributed X. (n_dim)
      - covariance: Covariance of the normal distribution (n_dim x n_dim)
      - M_diagonal: Diagonal elements of M. (n_dim)
    ----------------
    Outputs:
    ----------------
      - psi_quadratic: Variance of quadratic form.
    """
    # Compute the trace of M*covariance*M*covariance
    covariance_scaled_columns = M_diagonal * covariance
    # trace = product_trace(covariance_scaled_columns,
    #                      covariance_scaled_columns)
    trace = torch.einsum("i,ij,j,ji->", M_diagonal, covariance, M_diagonal, covariance)

    # Compute the quadratic form term
    mu_quadratic = torch.einsum(
        "d,d,dk,k,k->", mu, M_diagonal, covariance, M_diagonal, mu
    )
    # Add terms
    psi_quadratic = 2 * trace + 4 * mu_quadratic
    return psi_quadratic


def quadratic_form_cov(mu, covariance, M, M2):
    """
    Computes the covariance of the quadratic forms of
    random variable X given by M and M2, where X ~ N(mu, covariance).
    ----------------
    Arguments:
    ----------------
      - mu: Mean of normal distributions X. (n_dim)
      - covariance: Covariance of the normal distributions (n_dim x n_dim)
      - M: Matrix of quadratic form 1. (n_dim x n_dim)
      - M2: Matrix of quadratic form 2. (n_dim x n_dim)
    ----------------
    Outputs:
    ----------------
      - cov_quadratic: Covariance of quadratic forms of random
          variable X with M and X with M2. Scalar
    """
    covariance = torch.as_tensor(covariance)
    # Compute the trace of M*covariance*M2*covariance
    if covariance.dim() == 2:
        trace = product_trace4(A=M, B=covariance, C=M2, D=covariance)
    elif covariance.dim() == 0:  # Isotropic case
        trace = product_trace(A=M, B=M2) * covariance**2
    # Compute mean term
    mu_quadratic = torch.einsum("d,db,bk,km,m->", mu, M, covariance, M2, mu)
    # Add terms
    cov_quadratic = 2 * trace + 4 * mu_quadratic
    return cov_quadratic


def quadratic_linear_cov(mu, covariance, M, b):
    """
    Computes the covariance of the quadratic form of random
    variable X with matrix M and the linear form given by b,
    where X ~ N(mu, covariance).
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distribution X. (n_dim)
      - covariance: Covariance of the normal distribution (n_dim x n_dim)
      - M: Matrix to multiply by. (n_dim x n_dim)
      - b: Vector for linear form. (n_dim)
    ----------------
    Outputs:
    ----------------
      - cov_quadratic: Covariance of quadratic form and linear form of random
          variable X. Scalar
    """
    cov_quadratic = 2 * torch.einsum("i,ij,jk,k->", mu, M, covariance, b)
    return cov_quadratic


def nc_X2_moments(mu, sigma, s):
    """
    Computes the s-th moment of the non-central chi squared
    distribution with mean vector mu and standard deviation sigma
    for each element.
    ----------------
    Arguments:
    ----------------
      - mu: Multidimensional mean of the gaussian. (n_dim)
      - sigma: Standard deviation of isotropic noise. (Scalar)
      - s: Order of the moment to compute.
    ----------------
    Outputs:
    ----------------
      - moment: s-th moment.
    """
    n_dim = torch.as_tensor(len(mu))
    # lambda parameter of non-central chi distribution, squared
    nc = non_centrality(mu=mu, sigma=sigma)
    if s == 1:
        out = (nc + n_dim) * sigma**2
    elif s == 2:
        out = (n_dim**2 + 2 * n_dim + 4 * nc + nc**2 + 2 * n_dim * nc) * sigma**4
    else:
        # Get gamma and hyp1f1 values
        hyp_val = sps.hyp1f1(n_dim / 2 + s, n_dim / 2, nc / 2)
        gammaln1 = gammaln(n_dim / 2 + s)
        gammaln2 = gammaln(n_dim / 2)
        gamma_ratio = (2**s / torch.exp(nc / 2)) * torch.exp(gammaln1 - gammaln2)
        moment = (gamma_ratio * hyp_val) * (sigma ** (s * 2))  # This is a torch tensor
    return moment


def inv_ncx_mean(mu, sigma):
    """
    Compute the expected value of 1/||X|| where X is
    a non-centered gaussian with mean mu and standard deviation sigma.
    ----------------
    Arguments:
    ----------------
      - mu: Mean of the gaussian. (n_dim)
      - sigma: Standard deviation of isotropic noise. (Scalar)
    ----------------
    Outputs:
    ----------------
      - gamma_invncx: Expected value of 1/||X||
    """
    n_dim = torch.as_tensor(len(mu))
    # lambda parameter of non-central chi distribution, squared
    nc = non_centrality(mu=mu, sigma=sigma)
    # Corresponding hypergeometric function values
    hyp_val = sps.hyp1f1(1 / 2, n_dim / 2, -nc / 2)
    gammaln1 = gammaln((n_dim - 1) / 2)
    gammaln2 = gammaln(n_dim / 2)
    gamma_ratio = (1 / np.sqrt(2)) * torch.exp(gammaln1 - gammaln2)
    gamma_invncx = (gamma_ratio * hyp_val) / sigma  # This is a torch tensor
    return gamma_invncx


def inv_ncx2_mean(mu, sigma):
    """
    Compute the expected value of 1/(||X||^2) where X is
    a non-centered gaussian with mean mu and standard deviation sigma.
    ----------------
    Arguments:
    ----------------
      - mu: Multidimensional mean of the gaussian. (n_dim)
      - sigma: Standard deviation of isotropic noise. (Scalar)
    ----------------
    Outputs:
    ----------------
      - gamma_invncx2: Expected value of 1/||X||^2.
    """
    n_dim = torch.as_tensor(len(mu))
    nc = non_centrality(mu=mu, sigma=sigma)
    gammaln1 = gammaln(n_dim / 2 - 1)
    gammaln2 = gammaln(n_dim / 2)
    gamma_ratio = 0.5 * torch.exp(gammaln1 - gammaln2)
    hyp_val = sps.hyp1f1(1, n_dim / 2, -nc / 2)
    gamma_invncx2 = (gamma_ratio * hyp_val) / sigma**2
    return gamma_invncx2


##################################
##################################
#
## EMPIRICAL MOMENTS
#
##################################
##################################


def sample_quadratic_form(mu, covariance, M, n_samples):
    """
    Sample from the quadratic form X^T M X, where X~N(mu, covariance).
    -----------------
    Arguments:
    -----------------
      - mu: Mean. (n_dim)
      - covariance: Covariance matrix. (n_dim x n_dim)
      - M: Matrix of the quadratic form. (n_dim x n_dim)
      - n_samples: Number of samples.
    -----------------
    Output:
    -----------------
      - samples_qf: Samples from the quadratic form. (n_samples)
    """
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    X = dist.sample([n_samples])
    if is_diagonal(M):
        D = torch.diagonal(M)
        samples_qf = torch.einsum("ni,i,in->n", X, D, X.t())
    else:
        samples_qf = torch.einsum("ni,ij,jn->n", X, M, X.t())
    return samples_qf


def empirical_moments_quadratic_form(mu, covariance, M, n_samples):
    """Compute the mean and variance of the quadratic form
    qf = X^T M X for X~N(mu, covariance).
    -----------------
    Arguments:
    -----------------
      - mu: Mean. (n_dim)
      - covariance: Covariance matrix. (n_dim x n_dim)
      - M: Matrix of the quadratic form. (n_dim x n_dim)
      - n_samples: Number of samples.
    -----------------
    Output:
    -----------------
      - statsDict: Dictionary with the mean, variance and second
          moment of the quadratic form
    """
    samples_qf = sample_quadratic_form(mu, covariance, M, n_samples)
    mean = torch.mean(samples_qf)
    var = torch.var(samples_qf)
    second_moment = torch.mean(samples_qf**2)
    return {"mean": mean, "var": var, "second_moment": second_moment}


def empirical_covariance_quadratic_form(mu, covariance, M1, M2, n_samples):
    """Compute the covariance between the quadratic forms
    qf1 = X^T M1 X and qf2 = X^T M2 X, where X~N(mu, covariance).
    -----------------
    Arguments:
    -----------------
      - mu: Mean vector of the Gaussian. (n_dim)
      - covariance: Covariance matrix of the Gaussian. (n_dim x n_dim)
      - M1: Matrix of the first quadratic form. (n_dim x n_dim)
      - M2: Matrix of the second quadratic form. (n_dim x n_dim)
      - n_samples: Number of samples to use to compute the moments.
    -----------------
    Output:
    -----------------
      - cov: Covariance between the two quadratic forms.
    """
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    X = dist.sample([n_samples])
    qf1 = torch.einsum("ni,ij,jn->n", X, M1, X.t())
    qf2 = torch.einsum("ni,ij,jn->n", X, M2, X.t())
    cov = torch.cov(torch.cat((qf1.unsqueeze(0), qf2.unsqueeze(0))))[0, 1]
    return cov
