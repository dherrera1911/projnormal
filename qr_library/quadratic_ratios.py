import numpy as np
import torch
from torch.special import gammaln
from torch import fft as fft
import mpmath as mpm  # Important to use mpmath for hyp1f1, scipy blows up

##################################
##################################
#
## QUADRATIC MOMENTS FUNCTIONS
#
##################################
##################################

#### Terminology:
# X: Random variable with multidimensional Gaussian distribution
# mu: Mean of X
# sigma: Standard deviation of X
# nc: non-centrality parameter
# df: Degrees of freedom of X, or nDim
# nX: Number of random variables with different mu or sigma

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
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    muNorm = mu/sigma
    # non-centrality parameter, ||\mu||^2
    nc = muNorm.norm(dim=1)**2
    return nc


# Get the values of the hypergeometric function given a and b, for each
# value of non-centrality parameter, which is random-variable  dependent
def hyp1f1(a, b, nc):
    """ For each element in nc (which is the non-centrality parameter
    given by ||mu/sigma||**2 for random variable X), compute the
    confluent hypergeometric function hyp1f1(a, b, -nc/2).
    Acts as a wrapper of mpm.hyp1f1 for pytorch tensors.
    ----------------
    Arguments:
    ----------------
      - a: First parameter of hyp1f1 (usually 1 or 1/2). (Scalar)
      - b: Second parameter of hyp1f1 (usually df/2+k, k being an integer). (Scalar)
      - nc: Non-centrality parameter. (Vector length df)
    ----------------
    Outputs:
    ----------------
      - hypVal: Value of hyp1f1 for each nc. (Vector length df)
    """
    # If nc is a pytorch tensor, convert to numpy array
    isTensor = isinstance(nc, torch.Tensor)
    if isTensor:
        device = nc.device
        if nc.is_cuda:
            nc = nc.cpu()
        nc = nc.numpy()
    nX = len(nc)  # Get number of dimensions
    # Calculate hypergeometric functions
    hypVal = torch.zeros(nX)
    for i in range(nX):
        hypVal[i] = torch.tensor(float(mpm.hyp1f1(a, b, -nc[i]/2)))
    if isTensor:
        hypVal = hypVal.to(device)
    return hypVal


def isotropic_sm_weights(mu, sigma):
    """ For a set of X, assuming isotropic Gaussian noise,
    compute and the of the mean outer product and the identity
    matrix in the second moment estimation formula.
    ----------------
    Arguments:
    ----------------
      - mu: Means of different X. (nX x nDim)
      - sigma: Standard deviation of the noise
    ----------------
    Outputs:
    ----------------
      - nc: Non centrality parameter of each X (nX)
      - smMeanW: Weigths for the outer products of the means for
          each random variable. (nX)
      - smNoiseW: Weights for the identity matrices. (nX)
    """
    df = mu.shape[1]
    # Square mean of stimuli divided by standard deviation
    nc = non_centrality(mu, sigma)
    # Weighting factors for the mean outer product in second moment estimation
    smMeanW = hyp1f1(a=1, b=df/2+2, nc=nc) * (1/(df+2))
    # Weighting factors for the identity matrix in second moment estimation
    smNoiseW = hyp1f1(a=1, b=df/2+1, nc=nc) * (1/df)
    return torch.tensor(nc), smMeanW, smNoiseW


# Inverse non-centered chi expectation.
def inv_ncx_mean(mu, sigma):
    """ Get the expected value of the inverse of the norm
    of a multivariate gaussian X with mean mu and isotropic noise
    standard deviation sigma, for each different value of mu.
    ----------------
    Arguments:
    ----------------
      - mu: Multidimensional mean of the gaussian. (nX x df)
      - sigma: Standard deviation of isotropic noise. (Scalar)
    ----------------
    Outputs:
    ----------------
      - expectedValue: Expected value of 1/||x|| with x~N(mu, sigma).
          (Vector length nX)
    """
    df = mu.shape[1]
    # lambda parameter of non-central chi distribution, squared
    lam = non_centrality(mu=mu, sigma=sigma)
    # Corresponding hypergeometric function values
    hypGeomVal = hyp1f1(1/2, df/2, lam)
    gammaQRes = (1/np.sqrt(2)) * torch.exp(gammaln(torch.tensor((df-1)/2))
                                           - gammaln(torch.tensor(df/2)))
    expectedValue = (gammaQRes * hypGeomVal) / sigma  # This is a torch tensor
    return expectedValue


# Inverse non-centered chi square expectation.
##### Check that description is ok, regarding non-centrality
##### parameter and what distribution is actually obtained
def inv_ncx2_mean(mu, sigma):
    """ Get the expected value of the inverse of the
    squared norm of a non-centered gaussian
    distribution, with degrees of freedom df, and non-centrality
    parameter nc (||\mu||^2).
    ----------------
    Arguments:
    ----------------
      - df: degrees of freedom
      - nc: non-centrality parameter
    ----------------
    Outputs:
    ----------------
      - expectedValue: Expected value of 1/||x||^2 with x~N(mu, sigma).
    """
    df = mu.shape[1]
    nc = non_centrality(mu=mu, sigma=sigma)
    gammaQRes = 0.5 * torch.exp((gammaln(torch.tensor(df)/2-1) -
                                 gammaln(torch.tensor(df)/2)))
    hypFunRes = hyp1f1(1, df/2, nc)
    expectedValue = (gammaQRes * hypFunRes) / sigma**2
    return expectedValue


##################################
##################################
#
## RATIOS OF QUADRATIC FORMS
#
##################################
##################################
#
# NOTE: POINT THE SPECIFIC FUNCTIONS TO THE ACCOMPANYING DOCUMENT

def projected_normal_mean(mu, sigma=0.1, invNorm=None, acrossVars=True):
    """ Compute the approximated expected value of each
    projected gaussian Yi = Xi/||Xi||, where Xi~N(mu[i,:], I*sigma^2).
    The expected value is approximated, by taking the expected values
    to be mu * E(1/||X||).
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nX x nDim)
      - sigma: Standard deviation of the normal distributions (isotropic)
      - invNorm: If E(1/||X||) is already known, it can be passed here
          to avoid computing again. Shape (nX)
      - acrossVars: If True, compute the expected value across
          stimuli, instead of the expected value of each stimulus.
    ----------------
    Outputs:
    ----------------
      - YExpected: Expected mean value across projected
          normals, or for each projected normal, as indicated in
          acrossVars. Shape (nDim) or (nX x nDim), respectively.
    """
    nX = mu.shape[0]
    # Get expected value of inverse norm
    if invNorm is None:
        invNorm = inv_ncx_mean(mu, sigma)
    # Compute expected value
    if acrossVars:
        YExpected = torch.einsum('nb,n->b', mu, invNorm/nX)
    else:
        YExpected = torch.einsum('nd,n->nd', mu, invNorm)
    return YExpected


#############
#### SECOND MOMENT
#############

def compute_missing_weights(mu, sigma, noiseW=None, meanW=None):
    """ Auxiliary function, if weights of noise and mean outer
    products are not given, compute them"""
    # If precomputed weights are not given, compute them here
    df = mu.shape[1]
    if (noiseW is None) or (meanW is None):
        nc = non_centrality(mu=mu, sigma=sigma)
    if noiseW is None:
        hypFunNoise = hyp1f1(a=1, b=df/2+1, nc=nc)
        noiseW = hypFunNoise * (1/df)
        noiseW = noiseW
    if meanW is None:
        hypFunMean = hyp1f1(a=1, b=df/2+2, nc=nc)
        meanW = hypFunMean * (1/(df+2))
        meanW = meanW
    return noiseW, meanW


# Apply the isotropic covariance formula to get the covariance
# for each stimulus
def projected_normal_sm_iso(mu, sigma, noiseW=None, meanW=None, acrossVars=True):
    """ Compute the second moment of each projected gaussian
    Yi = Xi/||Xi||, where Xi~N(mu[i,:], sigma*I). Note that
    Xi has isotropic noise.
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nX x nDim)
      - sigma: Standard deviation of the normal distributions (isotropic)
      - noiseW: If the weighting term for the identity matrix is already
          computed, it can be passed here to avoid computing again. (nX)
      - meanW: If the weighting term for the mean outer product is already
          computed, it can be passed here to avoid computing again. (nX)
      - acrossVars: If True, compute the expected value across
          stimuli, instead of the expected value of each stimulus.
    ----------------
    Outputs:
    ----------------
      - YSM: Second moment across the projected gaussians, or of each projected
          gaussian, as indicated in acrossVars.
          Shape (nDim x nDim) or (nX x nDim x nDim), respectively.
    """
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    nX = mu.shape[0]
    df = mu.shape[1]
    # If precomputed weights are not given, compute them here
    noiseW, meanW = compute_missing_weights(mu=mu, sigma=sigma,
                                            noiseW=noiseW, meanW=meanW)
    if acrossVars:
        # Get the total weight of the identity in the across stim SM
        noiseWMean = noiseW.mean()
        # Scale each stimulus by the sqrt of the outer prods weights
        stimScales = torch.sqrt(meanW/(nX))/sigma
        #if normFactor is not None:
        #    stimScales = stimScales * torch.sqrt(normFactor[mask])
        scaledStim = torch.einsum('nd,n->nd', mu, stimScales)
        YSM = torch.einsum('nd,nb->db', scaledStim, scaledStim) + \
                torch.eye(df, device=mu.device) * noiseWMean
    else:
        # Compute the second moment of each stimulus
        muNorm = mu/sigma
        # Get the outer product of the normalized stimulus, and multiply by weight
        meanTerm = torch.einsum('nd,nb,n->ndb', muNorm, muNorm, meanW)
        # Multiply the identity matrix by weighting term
        eye = torch.eye(df, device=mu.device)
        noiseTerm = torch.einsum('db,n->ndb', eye, device=mu.device), noiseW)
        # Add the two terms
        YSM = torch.tensor(meanTerm + noiseTerm)
    return YSM


def product_trace(A, B):
    """ Compute the trace of the product of two matrices
    A and B. """
    return torch.einsum('ij,ji->', A, B)


def product_trace4(A, B, C, D):
    """ Compute the trace of the product of  matrices
    A and B. """
    return torch.einsum('ij,jk,kl,li->', A, B, C, D)


def quadratic_form_mean(mu, covariance, M):
    """ Compute the mean of the quadratic form
    X^T * matrix * X, where X~N(mu, covariance).
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nX x nDim)
      - covariance: Covariance of the normal distributions (nX x nDim x nDim)
      - matrix: Matrix to multiply by. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - qfMean: Expected value of quadratic form of random
          variable X with M. Shape (nX)
    """
    covariance = torch.tensor(covariance)
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    # Compute the trace of M*covariance
    if covariance.dim() == 2:
        trace = product_trace(M, covariance)
    elif covariance.dim() == 0:
        trace = torch.trace(M)*covariance
    # Compute mean
    muQuadratic = torch.einsum('nd,db,nb->n', mu, M, mu)
    # Add terms
    qfMean = trace + muQuadratic
    return qfMean


def quadratic_form_var(mu, covariance, M):
    """ Compute the variance of the quadratic form
    X^T * matrix * X, where X~N(mu, covariance), and M is
    positive definite.
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nX x nDim)
      - covariance: Covariance of the normal distributions (nX x nDim x nDim)
      - matrix: Matrix to multiply by. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - qfVar: Variance of quadratic form of random
          variable X with M. Shape (nX)
    """
    covariance = torch.tensor(covariance)
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    # Compute the trace of M*covariance*M*covariance
    if covariance.dim() == 2:
        trace = product_trace4(A=M, B=covariance, C=M, D=covariance)
    elif covariance.dim() == 0:  # Isotropic case
        trace = product_trace(A=M, B=M) * covariance**2
    # Compute mean term
    muQuadratic = torch.einsum('nd,db,bk,km,nm->n', mu, M, covariance, M, mu)
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
      - mu: Means of normal distributions X. (nX x nDim)
      - covariance: Covariance of the normal distributions (nX x nDim x nDim)
      - M: Matrix to multiply by. (nDim x nDim)
      - M2: Matrix to multiply by. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - qfCov: Covariance of quadratic forms of random
          variable X with M and X with M2. Shape (nX)
    """
    covariance = torch.tensor(covariance)
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    # Compute the trace of M*covariance*M2*covariance
    if covariance.dim() == 2:
        trace = product_trace4(A=M, B=covariance, C=M2, D=covariance)
    elif covariance.dim() == 0:  # Isotropic case
        trace = product_trace(A=M, B=M2) * covariance**2
    # Compute mean term
    muQuadratic = torch.einsum('nd,db,bk,km,nm->n', mu, M, covariance, M2, mu)
    # Add terms
    qfCov = 2*trace + 4*muQuadratic
    return qfCov


def projected_normal_sm(mu, covariance, B):
    """ Compute the second moment of the projected normal distribution
    Yi = Xi/||Xi||, where Xi~N(mu[i,:], sigma*I)
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nX x nDim)
      - covariance: Covariance of the normal distributions (nX x nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - YSM: Second moment of each projected gaussian.
          Shape (nX x nDim x nDim)
    """
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
        # Compute the mean of numerator for each matrix A^{ij}
    muN = covariance + torch.einsum('nd,nb->ndb', mu, mu)
    # Compute denominator terms
    muD = quadratic_form_mean(mu=mu, covariance=covariance, M=B)
    varD = quadratic_form_var(mu=mu, covariance=covariance, M=B)
    # Compute covariance between numerator and denominator for each
    # matrix A^{ij}
    covB = torch.einsum('ij,jk->ik', covariance, B)
    term1 = torch.einsum('ij,jk->ik', covB, covariance)
    term2 = torch.einsum('ni,nj,kj->nik', mu, mu, covB)
    covND = 2 * (term1 + term2 + term2.transpose(1, 2))
    # Compute second moment of projected normal
    YSM = torch.zeros_like(muN)
    for i in range(muN.shape[0]):
        YSM[i,:,:] = muN[i,:,:]/muD[i] * \
            (1 - covND[i,:,:]/muN[i,:,:]*(1/muD[i]) + varD[i]/muD[i]**2)
    return YSM


#dim = 10
#B = torch.randn(dim, dim) 
#B = B + B.T
#covariance = make_spdm(dim=dim)
#i = 7
#j = 3
#A = make_Aij(dim=dim, i=i, j=j)
#lala = quadratic_form_cov(mu=mu, covariance=covariance, M=A, M2=B)
#print(f'Estimate 1: {lala}, estimate 2: {covND[i,j]}')

##################################
##################################
#
## UTILITY FUNCTIONS
#
##################################
##################################


def secondM_2_cov(secondM, mean, nX=None):
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
    if secondM.dim() == 2:
        secondM = secondM.unsqueeze(0)
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)
    # Get the multiplying factor to make covariance unbiased
    if nX is None:
        unbiasingFactor = 1
    else:
        unbiasingFactor = nX/(nX-1)
    covariance = (secondM - torch.einsum('cd,cb->cdb', mean, mean)) * unbiasingFactor
    return covariance


def make_spdm(dim):
    """ Make a random symmetric positive definite matrix
    ----------------
    Arguments:
    ----------------
      - dim: Dimension of matrix
    ----------------
    Outputs:
    ----------------
      - M: Random symmetric positive definite matrix
    """
    M = torch.randn(dim, dim)
    M = torch.einsum('ij,ik->jk', M, M) / dim**2
    return M


def make_Aij(dim, i, j):
  A = torch.zeros(dim, dim)
  A[i,j] = 0.5
  A[j,i] = A[j,i] + 0.5
  return A



