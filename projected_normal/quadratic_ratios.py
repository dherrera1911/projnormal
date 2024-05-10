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
# nDim: Dimensions of X


import numpy as np
import torch
from torch.special import gammaln
import scipy.special as sps
import torch.distributions.multivariate_normal as mvn


##################################
##################################
#
## QUADRATIC FORMS MOMENTS
#
##################################
##################################


def quadratic_form_mean(mu, covariance, M=None):
    """ Compute the mean of the quadratic form
    X^T * M * X, where X~N(mu, covariance).
    ----------------
    Arguments:
    ----------------
      - mu: Mean of normal distribution X. (nDim)
      - covariance: Covariance of the normal distribution (nDim x nDim)
      - M: Matrix to multiply by. (nDim x nDim). If None, M=I.
    ----------------
    Outputs:
    ----------------
      - qfMean: Expected value of quadratic form of random
          variable X with M. Scalar
    """
    covariance = torch.as_tensor(covariance)
    if M is None:
        trace = torch.trace(covariance)
        muQuadratic = torch.einsum('d,d->', mu, mu)
    else:
        trace = product_trace(M, covariance)
        muQuadratic = torch.einsum('d,db,b->', mu, M, mu)
    # Add terms
    qfMean = trace + muQuadratic
    return qfMean


def quadratic_form_var(mu, covariance, M=None):
    """ Compute the variance of the quadratic form
    X^T * M * X, where X~N(mu, covariance), and M is
    positive definite.
    ----------------
    Arguments:
    ----------------
      - mu: Mean of normal distribution X. (nDim)
      - covariance: Covariance of the normal distribution (nDim x nDim)
      - M: Matrix to multiply by. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - qfVar: Variance of quadratic form of random
          variable X with M. Scalar
    """
    covariance = torch.as_tensor(covariance)
    # Compute the trace of M*covariance*M*covariance
    if M is None:
        trace = product_trace(covariance, covariance)
        muQuadratic = torch.einsum('b,bk,k->', mu, covariance, mu)
    else:
        trace = product_trace4(A=M, B=covariance, C=M, D=covariance)
        muQuadratic = torch.einsum('d,db,bk,km,m->', mu, M, covariance, M, mu)
    # Add terms
    qfVar = 2*trace + 4*muQuadratic
    return qfVar


def quadratic_form_cov(mu, covariance, M, M2):
    """ Compute the covariance of the quadratic forms
    given by random variable X with M and X with M2.
    X ~ N(mu, covariance).
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nDim)
      - covariance: Covariance of the normal distributions (nDim x nDim)
      - M: Matrix to multiply by. (nDim x nDim)
      - M2: Matrix to multiply by. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - qfCov: Covariance of quadratic forms of random
          variable X with M and X with M2. Scalar
    """
    covariance = torch.as_tensor(covariance)
    # Compute the trace of M*covariance*M2*covariance
    if covariance.dim() == 2:
        trace = product_trace4(A=M, B=covariance, C=M2, D=covariance)
    elif covariance.dim() == 0:  # Isotropic case
        trace = product_trace(A=M, B=M2) * covariance**2
    # Compute mean term
    muQuadratic = torch.einsum('d,db,bk,km,m->', mu, M, covariance, M2, mu)
    # Add terms
    qfCov = 2*trace + 4*muQuadratic
    return qfCov


def quadratic_linear_cov(mu, covariance, M, b):
    """ Compute the covariance of the quadratic form
    given by X'MX and the linear form given by b'X.
    X ~ N(mu, covariance).
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nDim)
      - covariance: Covariance of the normal distributions (nDim x nDim)
      - M: Matrix to multiply by. (nDim x nDim)
      - b: Vector for linear form. (nDim)
    ----------------
    Outputs:
    ----------------
      - qlCov: Covariance of quadratic form and linear form of random
          variable X. Scalar
    """
    qlCov = 2 * torch.einsum('i,ij,jk,k->', mu, M, covariance, b)
    return qlCov


# Moments of non-central chi squared distribution
def nc_X2_moments(mu, sigma, s):
    """ Get the s-th moment of the non-central chi squared
    distribution, for a normal distribution with mean mu and
    standard deviation sigma for each element.
    ----------------
    Arguments:
    ----------------
      - mu: Multidimensional mean of the gaussian. (nDim)
      - sigma: Standard deviation of isotropic noise. (Scalar)
      - s: Order of the moment to compute
    ----------------
    Outputs:
    ----------------
      - out: Expected value of the s-th moment of the non-central
          chi squared distribution. Scalar
    """
    nDim = torch.as_tensor(len(mu))
    # lambda parameter of non-central chi distribution, squared
    nc = non_centrality(mu=mu, sigma=sigma)
    if s == 1:
        out = (nc + nDim) * sigma**2
    elif s == 2:
        out = (nDim**2 + 2*nDim + 4*nc + nc**2 + 2*nDim*nc) * sigma**4
    else:
        # Get gamma and hyp1f1 values
        hypGeomVal = sps.hyp1f1(nDim/2+s, nDim/2, nc/2)
        gammaln1 = gammaln(nDim/2+s)
        gammaln2 = gammaln(nDim/2)
        gammaQRes = (2**s/torch.exp(nc/2)) * torch.exp(gammaln1 - gammaln2)
        out = (gammaQRes * hypGeomVal) * (sigma**(s*2))  # This is a torch tensor
    return out


# Inverse non-centered chi expectation.
def inv_ncx_mean(mu, sigma):
    """ Get the expected value of the inverse of the norm
    of a multivariate gaussian X with mean mu and isotropic noise
    standard deviation sigma.
    ----------------
    Arguments:
    ----------------
      - mu: Multidimensional mean of the gaussian. (nDim)
      - sigma: Standard deviation of isotropic noise. (Scalar)
    ----------------
    Outputs:
    ----------------
      - expectedValue: Expected value of 1/||x|| with x~N(mu, sigma).
      Scalar
    """
    nDim = torch.as_tensor(len(mu))
    # lambda parameter of non-central chi distribution, squared
    nc = non_centrality(mu=mu, sigma=sigma)
    # Corresponding hypergeometric function values
    hypGeomVal = sps.hyp1f1(1/2, nDim/2, -nc/2)
    gammaln1 = gammaln((nDim-1)/2)
    gammaln2 = gammaln(nDim/2)
    gammaQRes = (1/np.sqrt(2)) * torch.exp(gammaln1 - gammaln2)
    expectedValue = (gammaQRes * hypGeomVal) / sigma  # This is a torch tensor
    return expectedValue


# Inverse non-centered chi square expectation.
def inv_ncx2_mean(mu, sigma):
    """ Get the expected value of the inverse of the
    squared norm of a non-centered gaussian
    distribution, with degrees of freedom nDim, and non-centrality
    parameter nc (||\mu||^2).
    ----------------
    Arguments:
    ----------------
      - mu: Multidimensional mean of the gaussian. (nDim)
      - sigma: Standard deviation of isotropic noise. (Scalar)
    ----------------
    Outputs:
    ----------------
      - expectedValue: Expected value of 1/||x||^2 with x~N(mu, sigma).
    """
    nDim = torch.as_tensor(len(mu))
    nc = non_centrality(mu=mu, sigma=sigma)
    gammaln1 = gammaln(nDim/2-1)
    gammaln2 = gammaln(nDim/2)
    gammaQRes = 0.5 * torch.exp(gammaln1 - gammaln2)
    hypFunRes = sps.hyp1f1(1, nDim/2, -nc/2)
    expectedValue = (gammaQRes * hypFunRes) / sigma**2
    return expectedValue


##################################
##################################
#
## HELPER FUNCTIONS
#
##################################
##################################


def non_centrality(mu, sigma):
    """ Compute the non-centrality parameter of each row
    in mu, given isotropic noise with sigma standard deviation.
    ----------------
    Arguments:
    ----------------
      - mu: Means of gaussians. Shape (nX x nDim)
      - sigma: standard deviation of noise. Scalar, or a vector
          (vector size nX or nDim?)
    ----------------
    Outputs:
    ----------------
      - nc: Non-centrality parameter of each random variable, which is
            ||muNorm||^2, where muNorm = mu/sigma. Shape (nX)
    """
    # If mu has dim 1, make it size 2 with 1 row
    muNorm = mu/sigma
    # non-centrality parameter, ||\mu||^2
    nc = muNorm.norm()**2
    return nc


def product_trace(A, B):
    """ Compute the trace of the product of two matrices
    A and B. """
    return torch.einsum('ij,ji->', A, B)


def product_trace4(A, B, C, D):
    """ Compute the trace of the product of  matrices
    A and B. """
    return torch.einsum('ij,jk,kl,li->', A, B, C, D)


#### CONVERT BETWEEN SECOND MOMENT MATRICES ####

def secondM_2_cov(secondM, mean):
    """Convert matrices of second moments to covariances, by
    subtracting the product of the mean with itself.
    ----------------
    Arguments:
    ----------------
      - secondM: Second moment matrix. E.g. computed with
           'isotropic_ctg_resp_secondM'. (nClasses x nFilt x nFilt,
           or nFilt x nFilt)
      - mean: mean matrix (nClasses x nFilt, or nFilt)
    ----------------
    Outputs:
    ----------------
      - covariance: Covariance matrices. (nClasses x nFilt x nFilt)
    """
    # Get the multiplying factor to make covariance unbiased
    covariance = secondM - torch.einsum('d,b->db', mean, mean)
    return covariance


def cov_2_corr(covariance):
    """ Convert covariance matrices to correlation matrices
    ----------------
    Arguments:
    ----------------
      - covariance: Covariance matrices. (nClasses x nFilt x nFilt)
    ----------------
    Outputs:
    ----------------
      - correlation: Correlation matrices. (nClasses x nFilt x nFilt)
    """
    std = torch.sqrt(torch.diagonal(covariance))
    correlation = covariance / torch.einsum('a,b->ab', std, std)
    return correlation


#### MATRIX CHECKS ####

def check_diagonal(B):
    """ Check if a matrix is diagonal
    ----------------
    Arguments:
    ----------------
      - B: Matrix to check. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - isDiagonal: True if B is diagonal, False otherwise
    """
    isDiagonal = torch.allclose(B, torch.diag(torch.diagonal(B)))
    return isDiagonal


##################################
##################################
#
## EMPIRICAL MOMENTS
#
##################################
##################################


def sample_quadratic_form(mu, covariance, M, nSamples):
    """ Sample from the random variable Y = X^T M X, where X~N(mu, covariance).
    -----------------
    Arguments:
    -----------------
      - mu: Mean. (nDim)
      - covariance: Covariance matrix. (nDim x nDim)
      - M: Matrix of the quadratic form. (nDim x nDim)
      - nSamples: Number of samples.
    -----------------
    Output:
    -----------------
      - qf: Samples from the quadratic form. (nSamples)
    """
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    X = dist.sample([nSamples])
    if check_diagonal(M):
        D = torch.diagonal(M)
        qf = torch.einsum('ni,i,in->n', X, D, X.t())
    else:
        qf = torch.einsum('ni,ij,jn->n', X, M, X.t())
    return qf


def empirical_moments_quadratic_form(mu, covariance, M, nSamples):
    """ Compute the mean and variance of the quadratic form
    qf = X^T M X for X~N(mu, covariance).
    -----------------
    Arguments:
    -----------------
      - mu: Mean. (nDim)
      - covariance: Covariance matrix. (nDim x nDim)
      - M: Matrix of the quadratic form. (nDim x nDim)
      - nSamples: Number of samples.
    -----------------
    Output:
    -----------------
      - statsDict: Dictionary with the mean, variance and second
          moment of the quadratic form
    """
    qfSamples = sample_quadratic_form(mu, covariance, M, nSamples)
    mean = torch.mean(qfSamples)
    var = torch.var(qfSamples)
    secondM = torch.mean(qfSamples**2)
    return {'mean': mean, 'var':var, 'secondM': secondM}


def empirical_covariance_quadratic_form(mu, covariance, M1, M2, nSamples):
    """ Compute the covariance between the quadratic forms
    qf1 = X^T M1 X and qf2 = X^T M2 X, where X~N(mu, covariance).
    -----------------
    Arguments:
    -----------------
      - mu: Mean vector of the Gaussian. (nDim)
      - covariance: Covariance matrix of the Gaussian. (nDim x nDim)
      - M1: Matrix of the first quadratic form. (nDim x nDim)
      - M2: Matrix of the second quadratic form. (nDim x nDim)
      - nSamples: Number of samples to use to compute the moments.
    -----------------
    Output:
    -----------------
      - cov: Covariance between the two quadratic forms.
    """
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    X = dist.sample([nSamples])
    qf1 = torch.einsum('ni,ij,jn->n', X, M1, X.t())
    qf2 = torch.einsum('ni,ij,jn->n', X, M2, X.t())
    cov = torch.cov(torch.cat((qf1.unsqueeze(0), qf2.unsqueeze(0))))[0,1]
    return cov

