import numpy as np
import torch
from torch.special import gammaln
import scipy.special as sps

#### Terminology:
# X: Random variable with multidimensional Gaussian distribution
# mu: Mean of X
# sigma: Standard deviation of X (if isotropic)
# covariance: Covariance of X
# nc: non-centrality parameter
# df: Degrees of freedom of X, or nDim
# nX: Number of random variables with different mu or sigma


##################################
##################################
#
## QUADRATIC FORMS MOMENTS
#
##################################
##################################


def quadratic_form_mean(mu, covariance, M):
    """ Compute the mean of the quadratic form
    X^T * M * X, where X~N(mu, covariance).
    ----------------
    Arguments:
    ----------------
      - mu: Mean of normal distribution X. (nDim)
      - covariance: Covariance of the normal distribution (nDim x nDim)
      - M: Matrix to multiply by. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - qfMean: Expected value of quadratic form of random
          variable X with M. Scalar
    """
    covariance = torch.as_tensor(covariance)
    # Compute the trace of M*covariance
    if covariance.dim() == 2:
        trace = product_trace(M, covariance)
    elif covariance.dim() == 0:
        trace = torch.trace(M)*covariance
    # Compute mean
    muQuadratic = torch.einsum('d,db,b->', mu, M, mu)
    # Add terms
    qfMean = trace + muQuadratic
    return qfMean


def quadratic_form_var(mu, covariance, M):
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
    if covariance.dim() == 2:
        trace = product_trace4(A=M, B=covariance, C=M, D=covariance)
    elif covariance.dim() == 0:  # Isotropic case
        trace = product_trace(A=M, B=M) * covariance**2
    # Compute mean term
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
def nc_X2_moments(mu, sigma, s, exact=True):
    """ Get the s-th moment of the non-central chi squared
    distribution, for a normal distribution with mean mu and
    standard deviation sigma for each element.
    ----------------
    Arguments:
    ----------------
      - mu: Multidimensional mean of the gaussian. (df)
      - sigma: Standard deviation of isotropic noise. (Scalar)
      - s: Order of the moment to compute
    ----------------
    Outputs:
    ----------------
      - out: Expected value of the s-th moment of the non-central
          chi squared distribution. Scalar
    """
    df = torch.as_tensor(len(mu))
    # lambda parameter of non-central chi distribution, squared
    nc = non_centrality(mu=mu, sigma=sigma)
    if s == 1:
        out = (nc + df) * sigma**2
    elif s == 2:
        out = (df**2 + 2*df + 4*nc + nc**2 + 2*df*nc) * sigma**4
    else:
        # Get gamma and hyp1f1 values
        hypGeomVal = hyp1f1(df/2+s, df/2, nc/2, exact=exact)
        gammaln1 = gammaln(df/2+s)
        gammaln2 = gammaln(df/2)
        gammaQRes = (2**s/torch.exp(nc/2)) * torch.exp(gammaln1 - gammaln2)
        out = (gammaQRes * hypGeomVal) * (sigma**(s*2))  # This is a torch tensor
    return out


# Inverse non-centered chi expectation.
def inv_ncx_mean(mu, sigma, exact=True):
    """ Get the expected value of the inverse of the norm
    of a multivariate gaussian X with mean mu and isotropic noise
    standard deviation sigma.
    ----------------
    Arguments:
    ----------------
      - mu: Multidimensional mean of the gaussian. (df)
      - sigma: Standard deviation of isotropic noise. (Scalar)
    ----------------
    Outputs:
    ----------------
      - expectedValue: Expected value of 1/||x|| with x~N(mu, sigma).
      Scalar
    """
    df = torch.as_tensor(len(mu))
    # lambda parameter of non-central chi distribution, squared
    nc = non_centrality(mu=mu, sigma=sigma)
    # Corresponding hypergeometric function values
    hypGeomVal = hyp1f1(1/2, df/2, -nc/2, exact=exact)
    gammaln1 = gammaln((df-1)/2)
    gammaln2 = gammaln(df/2)
    gammaQRes = (1/np.sqrt(2)) * torch.exp(gammaln1 - gammaln2)
    expectedValue = (gammaQRes * hypGeomVal) / sigma  # This is a torch tensor
    return expectedValue


# Inverse non-centered chi square expectation.
def inv_ncx2_mean(mu, sigma, exact=True):
    """ Get the expected value of the inverse of the
    squared norm of a non-centered gaussian
    distribution, with degrees of freedom df, and non-centrality
    parameter nc (||\mu||^2).
    ----------------
    Arguments:
    ----------------
      - mu: Multidimensional mean of the gaussian. (df)
      - sigma: Standard deviation of isotropic noise. (Scalar)
    ----------------
    Outputs:
    ----------------
      - expectedValue: Expected value of 1/||x||^2 with x~N(mu, sigma).
    """
    df = torch.as_tensor(len(mu))
    nc = non_centrality(mu=mu, sigma=sigma)
    gammaln1 = gammaln(df/2-1)
    gammaln2 = gammaln(df/2)
    gammaQRes = 0.5 * torch.exp(gammaln1 - gammaln2)
    hypFunRes = hyp1f1(1, df/2, -nc/2, exact=exact)
    expectedValue = (gammaQRes * hypFunRes) / sigma**2
    return expectedValue


##################################
##################################
#
## PROJECTED NORMAL MOMENTS
#
##################################
##################################
#
# NOTE: POINT THE SPECIFIC FUNCTIONS TO THE ACCOMPANYING DOCUMENT

#############
#### ISOTROPIC, EXACT
#############

def prnorm_mean_iso(mu, sigma, exact=True):
    """ Compute the expected value of each projected gaussian Yi = Xi/||Xi||,
    where Xi~N(mu[i,:], I*sigma^2).
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nX x nDim)
      - sigma: Standard deviation of the normal distributions (isotropic)
      - exact: If True, compute the exact value of the hypergeometric function.
    ----------------
    Outputs:
    ----------------
      - YExpected: Expected mean value of the projected
          normals. Shape (nX x nDim), respectively.
    """
    df = torch.as_tensor(len(mu))
    nc = non_centrality(mu, sigma)
    gammaln1 = gammaln((df+1)/2)
    gammaln2 = gammaln(df/2+1)
    gammaRatio = 1/(np.sqrt(2)*sigma) * torch.exp(gammaln1 - gammaln2)
    hypFunRes = hyp1f1(1/2, df/2+1, -nc/2, exact=exact)
    YExpected = gammaRatio * hypFunRes * mu
    return YExpected


# Apply the isotropic covariance formula to get the covariance
# for each stimulus
def prnorm_sm_iso(mu, sigma, exact=True):
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
    ----------------
    Outputs:
    ----------------
      - YSM: Second moment of each projected gaussian. Shape (nX x nDim x nDim)
    """
    df = torch.as_tensor(len(mu))
    # If precomputed weights are not given, compute them here
    noiseW, meanW = iso_sm_weights(mu=mu, sigma=sigma, exact=exact)
    # Compute the second moment of each stimulus
    muNorm = mu/sigma
    # Get the outer product of the normalized stimulus, and multiply by weight
    YSM = torch.einsum('d,b->db', muNorm, muNorm) * meanW
    # Add noise term to the diagonal
    diag_idx = torch.arange(df)
    YSM[diag_idx, diag_idx] += noiseW
    return YSM


def prnorm_sm_iso_batch(mu, sigma):
    """
    To get the SM of a set of projected normals, this saves a lot
    of computation time by not adding the indentity matrix to each SM.
    """
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    nX = mu.shape[0]
    df = mu.shape[1]
    # Compute mean SM
    noiseW = torch.zeros(nX, device=mu.device)
    meanW = torch.zeros(nX, device=mu.device)
    for i in range(nX):
        noiseW[i], meanW[i] = iso_sm_weights(mu=mu[i,:], sigma=sigma)
    # Compute the second moment of each stimulus
    muNorm = mu/sigma
    # Get the total weight of the identity in the across stim SM
    noiseWMean = noiseW.mean()
    # Scale each stimulus by the sqrt of the outer prods weights
    stimScales = torch.sqrt(meanW/(nX))/sigma
    scaledStim = torch.einsum('nd,n->nd', mu, stimScales)
    YSM = torch.einsum('nd,nb->db', scaledStim, scaledStim) + \
            torch.eye(df, device=mu.device) * noiseWMean
    return YSM


#############
#### GENERAL CASE, TAYLOR APPROXIMATION
#############


# Taylor approximation uses a variable v = ||X||^2 - X_i^2 for each
# X_i. That is, v is the sum of squares of all elements of X,
# except the i-th element.
# We use some functions to efficiently compute the statistics of v.
def v_mean(mu, covariance):
    """ For random variable X~N(mu,\Sigma), compute the expected value
    of ||X||^2 - X_i^2 for each X_i.
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nDim)
      - covariance: covariance of X. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - meanV: Expected value of ||X||^2 - X_i^2 for each different i (nDim)
    """
    # Compute the expected value of ||X||^2
    meanX2 = torch.trace(covariance) + torch.einsum('i,i->', mu, mu)
    # Compute E(X_i^2)
    meanXi2 = mu**2 + torch.diagonal(covariance)
    # Subtract to get E(||X||^2 - X_i^2)
    meanV = meanX2 - meanXi2
    return meanV


def v_var(mu, covariance):
    """ For random variable X~N(mu,\Sigma), compute the variance
    of ||X||^2 - X_i^2 for each X_i.
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nDim)
      - covariance: covariance of X. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - varV: Variance of ||X||^2 - X_i^2 for each different i (nDim)
    """
    # Compute the variance of ||X||^2
    varX2 = 2 * product_trace(covariance, covariance) + \
        4 * torch.einsum('i,ij,j->', mu, covariance, mu)
    # Compute the term to subtract for each X_i
    term1 = 2 * torch.einsum('ij,ij->i', covariance, covariance) - \
        covariance.diagonal()**2  # Repeated terms in the trace
    term2 = 2 * torch.einsum('i,ij,j->i', mu, covariance, mu) - \
        mu**2 * torch.diag(covariance) # Repeated terms in (mu'Cov mu)
    # Subtract to get variance
    varV = varX2 - 2*term1 - 4*term2
    return varV


def v_cov(mu, covariance):
    """ For random variable X~N(mu,\Sigma), compute the covariance
    between (||X||^2 - X_i^2) and X_i for each X_i.
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nDim)
      - covariance: covariance of X. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - covV: Covariance between (||X||^2 - X_i^2) and X_i for each
        different i (nDim)
    """
    covV = 2 * (torch.einsum('i,ij->j', mu, covariance) - \
        mu * torch.diagonal(covariance))
    return covV


def prnorm_mean_taylor(mu, covariance):
    """ Compute the approximated expected value of each
    projected gaussian Yi = Xi/||Xi||, where Xi~N(mu[i,:], \Sigma),
    and \Sigma is a diagonal matrix with variances on the diagonal.
    Uses a Taylor approximation where Xi/||Xi|| = u/sqrt(u^2 + v) = f(u,v)
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nDim)
      - covariance: covariance of X. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - YExpected: Expected mean value for each projected normal.
          Shape (nDim)
    """
    # NEED TO FIX THE DIMENSIONS, SINCE IT DOESNT WORK WITH
    # MORE THAN ONE VARIANCE VECTOR
    nDim = torch.as_tensor(len(mu))
    variances = torch.diagonal(covariance)
    ### Get moments required for the formula
    meanV = v_mean(mu=mu, covariance=covariance)
    # Compute the expected value and variance of ||X||^2
    meanX2 = quadratic_form_mean(mu=mu, covariance=covariance, M=torch.eye(nDim))
    varX2 = quadratic_form_var(mu=mu, covariance=covariance, M=torch.eye(nDim))
    # Compute E(X_i^2)
    meanXi2 = mu**2 + variances # Vector with E(Xi^2)
    # Get statistics of varible v in formulas (||X||^2 - X_i^2)
    # Get E(||X||^2 - X_i^2)
    meanX2 = meanX2.repeat(nDim)
    meanV = meanX2 - meanXi2
    # Get Var(||X||^2 - X_i^2)
    varSubtract = 2 * variances**2 + 4 * mu**2 * variances
    varX2 = varX2.repeat(nDim)
    varV = varX2 - varSubtract
    ### Get the derivatives for the taylor approximation
    dfdu2 = prnorm_du2(u=mu, v=meanV)
    dfdv2 = prnorm_dv2(u=mu, v=meanV)
    ### 0th order approximation
    term0 = mu / torch.sqrt(mu**2 + meanV)
    ### Compute Taylor approximation
    YExpected = term0 + 0.5*dfdu2*variances + 0.5*dfdv2*varV
    return YExpected


def prnorm_sm_taylor(mu, covariance, B):
    """ Compute the second moment of the projected normal distribution
    Yi = Xi/(X'BX), where Xi~N(mu[i,:], sigma*I).
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nX x nDim)
      - covariance: Covariance of the normal distributions (nX x nDim x nDim)
      - B: Matrix in the denominator. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - YSM: Second moment of each projected gaussian.
          Shape (nX x nDim x nDim)
    """
    # Compute the mean of numerator for each matrix A^{ij}
    muN = covariance + torch.einsum('d,b->db', mu, mu)
    # Compute denominator terms
    muD = quadratic_form_mean(mu=mu, covariance=covariance, M=B)
    varD = quadratic_form_var(mu=mu, covariance=covariance, M=B)
    # Compute covariance between numerator and denominator for each
    # matrix A^{ij}
    covB = torch.einsum('ij,jk->ik', covariance, B)
    term1 = torch.einsum('ij,jk->ik', covB, covariance)
    term2 = torch.einsum('i,j,kj->ik', mu, mu, covB)
    covND = 2 * (term1 + term2 + term2.transpose(0,1))
    # Compute second moment of projected normal
    YSM = muN/muD * (1 - covND/muN*(1/muD) + varD/muD**2)
    return YSM


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


## Get the values of the hypergeometric function given a and b, for each
## value of non-centrality parameter, which is random-variable  dependent
#def hyp1f1(a, b, c, exact=True):
#    """ For each element in c, compute the
#    confluent hypergeometric function hyp1f1(a, b, c).
#    Acts as a wrapper of mpm.hyp1f1 for pytorch tensors.
#    ----------------
#    Arguments:
#    ----------------
#      - a: First parameter of hyp1f1 (usually 1 or 1/2). (Scalar)
#      - b: Second parameter of hyp1f1 (usually df/2+k, k being an integer). (Scalar)
#      - c: Vector, usually function of non-centrality parameters (Vector length df)
#      - exact: Whether to use exact formula to compute hyp1f1 or approximation
#          (1+2*nc/b)**(-a). (Boolean)
#    ----------------
#    Outputs:
#    ----------------
#      - hypVal: Value of hyp1f1 for each nc. (Vector length df)
#    """
#    # If nc is a pytorch tensor, convert to numpy array
#    isTensor = isinstance(c, torch.Tensor)
#    b = float(b)
#    if exact:
#        if isTensor:
#            device = c.device
#            if c.is_cuda:
#                c = c.cpu()
#            c = c.numpy()
#        if isinstance(b, torch.Tensor) or isinstance(a, torch.Tensor):
#            b = float(b)
#            a = float(a)
#        nX = len(c)  # Get number of dimensions
#        # Calculate hypergeometric functions
#        hypVal = torch.zeros(nX)
#        for i in range(nX):
#            hypVal[i] = torch.tensor(float(mpm.hyp1f1(a, b, c[i])))
#        if isTensor:
#            hypVal = hypVal.to(device)
#    else:
#        hypVal = (1 + c/(b*2))**(-a)
#    return hypVal


# Get the values of the hypergeometric function given a and b, for each
# value of non-centrality parameter, which is random-variable  dependent
def hyp1f1(a, b, c, exact=True):
    """ For each element in c, compute the
    confluent hypergeometric function hyp1f1(a, b, c).
    ----------------
    Arguments:
    ----------------
      - a: First parameter of hyp1f1 (usually 1 or 1/2). (Scalar)
      - b: Second parameter of hyp1f1 (usually df/2+k, k being an integer). (Scalar)
      - c: Vector, usually function of non-centrality parameters (Vector length df)
      - exact: Whether to use exact formula to compute hyp1f1 or approximation
          (1+2*nc/b)**(-a). (Boolean)
    ----------------
    Outputs:
    ----------------
      - hypVal: Value of hyp1f1 for each nc. (Vector length df)
    """
    # If nc is a pytorch tensor, convert to numpy array
    if exact:
        hypVal = sps.hyp1f1(a, b, c)
    else:
        hypVal = (1 + c/(b*2))**(-a)
    return hypVal


def product_trace(A, B):
    """ Compute the trace of the product of two matrices
    A and B. """
    return torch.einsum('ij,ji->', A, B)


def product_trace4(A, B, C, D):
    """ Compute the trace of the product of  matrices
    A and B. """
    return torch.einsum('ij,jk,kl,li->', A, B, C, D)


def isotropic_sm_weights(mu, sigma, exact=True):
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
    smMeanW = hyp1f1(a=1, b=df/2+2, c=-nc/2, exact=exact) * (1/(df+2))
    # Weighting factors for the identity matrix in second moment estimation
    smNoiseW = hyp1f1(a=1, b=df/2+1, c=-nc/2, exact=exact) * (1/df)
    return torch.as_tensor(nc), smMeanW, smNoiseW


def iso_sm_weights(mu, sigma, nc=None, exact=True):
    """ Compute the weights of the mean outer product and of the identity
    matrix in the isotropic projected Gaussian SM formula."""
    # If precomputed weights are not given, compute them here
    df = len(mu)
    if nc is None:
        nc = non_centrality(mu=mu, sigma=sigma)
    hypFunNoise = hyp1f1(a=1, b=df/2+1, c=-nc/2, exact=exact)
    noiseW = hypFunNoise * (1/df)
    hypFunMean = hyp1f1(a=1, b=df/2+2, c=-nc/2, exact=exact)
    meanW = hypFunMean * (1/(df+2))
    return noiseW, meanW


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


#def make_Aij(dim, i, j):
#  A = torch.zeros(dim, dim)
#  A[i,j] = 0.5
#  A[j,i] = A[j,i] + 0.5
#  return A


def prnorm_du2(u, v):
    """ Compute Second derivative of X_i/||X|| wrt X_i,
    evaluated at point u, v. Does the derivative for each
    X_i in X.
    ----------------
    Arguments:
    ----------------
      - u: Vector with value of X_i at which to evaluate the
            derivative, for each different i (nX)
      - v: Vector with values of (\sum_{j!=i} X_j**2) - X_i^2 at which
            to evaluate, for each different i (nX)
    ----------------
    Outputs:
    ----------------
      - dfdu2: Second derivative of X_i/||X|| wrt X_i
          for each different i (nX)
    """
    dfdu2 = u*(3*u**2/(u**2+v) - 3) / (u**2+v)**(3/2)
    return dfdu2


def prnorm_dv2(u, v):
    """ Compute Second derivative of X_i/||X|| wrt
    (\sum_{j!=i} X_j**2) - X_i^2, evaluated at point u, v.
    Does the derivative for each X_i in X.
    ----------------
    Arguments:
    ----------------
      - u: Vector with values of X_i at which to evaluate (nX)
      - v: Vector with values of (\sum_j X_j**2) - u^2 at which
            to evaluate (nX)
    ----------------
    Outputs:
    ----------------
      - dfdv2: Second derivative of X_i/||X|| wrt
            (\sum_{j!=i} X_j**2) - X_i^2 for each different i (nX)
    """
    dfdv2 = 3*u/(4*(u**2+v)**(5/2))
    return dfdv2


def prnorm_dudv(u, v):
    """ Compute Second derivative of X_i/||X|| wrt
    (\sum_{j!=i} X_j**2) - X_i^2, evaluated at point u, v.
    Does the derivative for each X_i in X.
    ----------------
    Arguments:
    ----------------
      - u: Vector with values of X_i at which to evaluate (nX)
      - v: Vector with values of (\sum_j X_j**2) - u^2 at which
            to evaluate (nX)
    ----------------
    Outputs:
    ----------------
      - dfdv2: Second derivative of X_i/||X|| wrt
            (\sum_{j!=i} X_j**2) - X_i^2 for each different i (nX)
    """
    dfdudv = (u**2 - v/2) / (u**2 + v)**(5/2)
    return dfdudv

