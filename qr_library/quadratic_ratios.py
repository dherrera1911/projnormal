import numpy as np
import torch
from torch.special import gammaln
import scipy.special as sps
import torch.distributions.multivariate_normal as mvn

#### Terminology:
# X: Random variable with multidimensional Gaussian distribution
# mu: Mean of X
# sigma: Standard deviation of X (if isotropic)
# covariance: Covariance of X
# nc: non-centrality parameter
# nDim: Dimensions of freedom of X


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
        hypGeomVal = hyp1f1(nDim/2+s, nDim/2, nc/2)
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
    hypGeomVal = hyp1f1(1/2, nDim/2, -nc/2)
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
    hypFunRes = hyp1f1(1, nDim/2, -nc/2)
    expectedValue = (gammaQRes * hypFunRes) / sigma**2
    return expectedValue


##################################
##################################
#
## PROJECTED NORMAL MOMENTS
#
##################################
##################################

#############
#### GENERAL CASE, TAYLOR APPROXIMATION
#############

def prnorm_mean_taylor(mu, covariance, B=None, c50=0):
    """ Approximated expected value of the generalized projected gaussian
    Y = X/(X'BX + c50)^0.5, where X~N(mu, covariance). The approximation is
    based on the Taylor expansion of the function f(u,v) = u/sqrt(b*u^2 + v + c50),
    where u=X_i and v = (X'BX - B_{ii}X_i^2).
      The approximation assumues that B is diagonal, so the problem is diagonalized
    before the approximation is applied.
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nDim)
      - covariance: covariance of X. (nDim x nDim)
      - B: Matrix with normalization weights (nDim x nDim). If None, B=I.
      - c50: Constant added to the denominator. Scalar
    ----------------
    Outputs:
    ----------------
      - YExpected: Expected mean value for each projected normal.
          Shape (nDim)
    """
    if B is not None:
        if check_diagonal(B):
            weights = torch.diagonal(B)
            P = torch.eye(len(mu), dtype=mu.dtype, device=mu.device)
        else:
            if not check_symmetric(B):
                raise ValueError('B must be symmetric')
            # Diagonalize B
            eigvals, eigvecs = torch.linalg.eigh(B)
            # Sort eigenvectors by eigenvalues
            eigvals, indices = torch.sort(eigvals, descending=True)
            eigvecs = eigvecs[:, indices]
            weights = eigvals
            P = eigvecs
            # Project mu to the new basis
            mu = torch.einsum('ij,j->i', P.t(), mu)
            # Project covariance to the new basis
            covariance = torch.einsum('ij,jk,kl->il', P.t(), covariance, P)
    else:
        weights = None
    variances = torch.diagonal(covariance)
    ### Get moments of variable v (||X||^2 - X_i^2) required for the formula
    meanV = v_mean(mu=mu, covariance=covariance, weights=weights)
    varV = v_var(mu=mu, covariance=covariance, weights=weights)
    covV = v_cov(mu=mu, covariance=covariance, weights=weights)
    ### Get the derivatives for the taylor approximation
    dfdu2 = dfdu2(u=mu, v=meanV, b=weights, c50=c50)
    dfdv2 = dfdv2(u=mu, v=meanV, b=weights, c50=c50)
    dfdudv = dfdudv(u=mu, v=meanV, b=weights, c50=c50)
    ### 0th order term
    term0 = f0(u=mu, v=meanV, b=weights, c50=c50)
    ### Compute Taylor approximation
    YExpected = term0 + 0.5*dfdu2*variances + 0.5*dfdv2*varV + dfdudv*covV
    ### If B is not None, project back to original basis
    if B is not None:
        YExpected = torch.einsum('ij,j->i', P, YExpected)
    return YExpected


def prnorm_sm_taylor(mu, covariance, B=None, c50=0):
    """ Approximated second moment matrix of the generalized projected gaussian
    Y = X/(X'BX + c50)^0.5, where X~N(mu, covariance). The approximation is
    based on the Taylor expansion of the function f(N,D) = N/D,
    where N=X_i*X_j and D = X'BX + c50.
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nDim)
      - covariance: Covariance of the normal distributions (nDim x nDim)
      - B: Matrix in the denominator. (nDim x nDim)
      - c50: Constant added to the denominator. Scalar
    ----------------
    Outputs:
    ----------------
      - YSM: Second moment of each projected gaussian.
          Shape (nDim x nDim)
    """
    # Compute the mean of numerator for each matrix A^{ij}
    muN = covariance + torch.einsum('d,b->db', mu, mu)
    # Compute denominator terms
    muD = quadratic_form_mean(mu=mu, covariance=covariance, M=B)
    muD = muD + c50
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


# The Taylor approximation uses the auxiliary variable
# v = X'BX - B_{ii}*X_i^2 for each X_i.
# We use some functions to efficiently compute the statistics of v.
def v_mean(mu, covariance, weights=None):
    """ For random variable X~N(mu, Sigma) and diagonal matrix B, compute
    the expected value of V = [V_1, ..., V_n] where
    V_i = (X'BX - B_{ii}X_i^2).
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nDim)
      - covariance: covariance of X. (nDim x nDim)
      - weights: Diagonal elements of matrix B (nDim). If None, B is assumed
          to be the identity matrix.
    ----------------
    Outputs:
    ----------------
      - meanV: Expected value of V (nDim)
    """
    variances = covariance.diagonal()
    # If weights are not None, scale variances and mu
    if weights is not None:
        variances = weights * variances
    else:
        weights = torch.ones(len(mu))
    # Compute the expected value of X'BX
    meanX2 = torch.sum(variances) + torch.einsum('i,i->', mu, mu*weights)
    # Compute E(B_i*X_i^2)
    meanXi2 = mu**2 * weights + variances
    # Subtract to get E(X'BX - B_i*X_i^2)
    meanV = meanX2 - meanXi2
    return meanV


def v_var(mu, covariance, weights=None):
    """ For random variable X~N(mu, Sigma) and diagonal matrix B, compute
    the variance of each element of V = [V_1, ..., V_n], where
    V_i = (X'BX - B_{ii}X_i^2).
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nDim)
      - covariance: covariance of X. (nDim x nDim)
      - weights: Diagonal elements of matrix B (nDim). If None, B is assumed
          to be the identity matrix.
    ----------------
    Outputs:
    ----------------
      - varV: Variance of each element of V (nDim)
    """
    if weights is not None:
        Bcovariance = torch.einsum('i,ij->ij', weights, covariance)
    else:
        weights = torch.ones(len(mu))
    # Compute the variance of X'BX
    varX2 = 2 * product_trace(Bcovariance, Bcovariance) + \
        4 * torch.einsum('i,ij,j->', mu, Bcovariance, mu*weights)
    # Note: In line above, we implement mu'*B'*Cov*B*mu. Because
    # we already multiplied Cov by B on the left, we just multiply
    # mu by B on the right (i.e. because B is diagonal, we just
    # multiply by 'weights'). Similar logic is applied elsewhere.
    # Next, Compute the term to subtract for each X_i
    # The first term also has B baked into the covariance
    term1 = 2 * torch.einsum('ij,ji->i', Bcovariance, Bcovariance) - \
        Bcovariance.diagonal()**2  # Repeated terms in the trace
    term2 = 2 * torch.einsum('i,ij,j->i', mu, Bcovariance, mu*weights) - \
        mu**2 * torch.diag(Bcovariance) * weights # Repeated terms in (mu'Cov mu)
    # Subtract to get variance
    varV = varX2 - 2*term1 - 4*term2
    return varV


def v_cov(mu, covariance, weights=None):
    """ For random variable X~N(mu, Sigma) and diagonal matrix B, compute
    the covariance between each element of V = [V_1, ..., V_n], where
    V_i = (X'BX - B_{ii}X_i^2), and the corresponding X_i.
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nDim)
      - covariance: covariance of X. (nDim x nDim)
      - weights: Diagonal elements of matrix B (nDim). If None, B is assumed
          to be the identity matrix.
    ----------------
    Outputs:
    ----------------
      - covV: Covariance between each element of V and the
          corresponding X_i (nDim)
    """
    if weights is None:
        weights = torch.ones(len(mu))
    covV = 2 * (torch.einsum('i,ij->j', mu*weights, covariance) - \
        mu * weights * torch.diagonal(covariance))
    return covV


# Derivatives of the function f(u,v) = u/sqrt(u^2 + v + c50)
# that is used in the taylor approximation to the mean
def f0(u, v, b=None, c50=0):
    """ First term of the Taylor approximation of f(u,v) = u/sqrt(b*u^2 + v),
    evaluated at point u,v. b is a constant """
    if b is None:
        b = 1
    f0 = u / torch.sqrt(b*u**2 + v + c50)
    return f0


def dfdu2(u, v, b=None, c50=0):
    """ Second derivative of f(u,v) = u/sqrt(c*u^2 + v) wrt u,
    evaluated at point u,v. b is a constant """
    if b is None:
        b = 1
    dfdu2 = - 3 * b * u * (v + c50) / (b * u**2 + v + c50)**(5/2)
    return dfdu2


def dfdv2(u, v, b=None, c50=0):
    """ Second derivative of f(u,v) = u/sqrt(b*u^2 + v) wrt v,
    evaluated at point u,v. b is a constant """
    if b is None:
        b = 1
    dfdv2 = 0.75 * u / (b * u**2 + v + c50)**(5/2)
    return dfdv2


def dfdudv(u, v, b=None, c50=0):
    """ Mixed second derivative of f(u,v) = u/sqrt(b*u^2 + v),
    evaluated at point u,v. b is a constant"""
    if b is None:
        b = 1
    dfdudv = (b * u**2 - 0.5 * (v + c50)) / (b * u**2 + v + c50)**(5/2)
    return dfdudv


#############
#### ISOTROPIC CASE, EXACT SOLUTION
#############

def prnorm_mean_iso(mu, sigma):
    """ Compute the expected value of projected gaussian Y = X/||X||,
    where X~N(mu, I*sigma^2).
    ----------------
    Arguments:
    ----------------
      - mu: Mean of X. (nDim)
      - sigma: Standard deviation of X elements (Scalar)
    ----------------
    Outputs:
    ----------------
      - YExpected: Expected value of projected normal. Shape (nDim).
    """
    nDim = torch.as_tensor(len(mu))
    nc = non_centrality(mu, sigma)
    gammaln1 = gammaln((nDim+1)/2)
    gammaln2 = gammaln(nDim/2+1)
    gammaRatio = 1/(np.sqrt(2)*sigma) * torch.exp(gammaln1 - gammaln2)
    hypFunRes = hyp1f1(1/2, nDim/2+1, -nc/2)
    YExpected = gammaRatio * hypFunRes * mu
    return YExpected


# Apply the isotropic covariance formula to get the covariance
# for each stimulus
def prnorm_sm_iso(mu, sigma):
    """ Compute the second moment of each projected gaussian
    Yi = Xi/||Xi||, where Xi~N(mu[i,:], sigma*I). Note that
    Xi has isotropic noise.
    ----------------
    Arguments:
    ----------------
      - mu: Mean of X (nDim)
      - sigma: Standard deviation of X elements (Scalar)
    ----------------
    Outputs:
    ----------------
      - YSM: Second moment matrix projected normal. Shape (nDim x nDim)
    """
    nDim = torch.as_tensor(len(mu))
    # If precomputed weights are not given, compute them here
    noiseW, meanW = iso_sm_weights(mu=mu, sigma=sigma)
    # Compute the second moment of each stimulus
    muNorm = mu/sigma
    # Get the outer product of the normalized stimulus, and multiply by weight
    YSM = torch.einsum('d,b->db', muNorm, muNorm) * meanW
    # Add noise term to the diagonal
    diag_idx = torch.arange(nDim)
    YSM[diag_idx, diag_idx] += noiseW
    return YSM


def prnorm_sm_iso_batch(mu, sigma):
    """
    Get the SM of a set of projected normals. This saves a lot
    of computation time by not adding the indentity matrix to each SM.
    ----------------
    Arguments:
    ----------------
      - mu: Means of normal distributions X. (nX x nDim)
      - sigma: Standard deviation of the normal distributions (isotropic)
      - noiseW: If the weighting term for the identity matrix is already
          computed, it can be passed here to avoid computing again.
      - meanW: If the weighting term for the mean outer product is already
          computed, it can be passed here to avoid computing again.
    ----------------
    Outputs:
    ----------------
      - YSM: Second moment of each projected gaussian. Shape (nDim x nDim)
    """
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    nX = mu.shape[0]
    nDim = mu.shape[1]
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
            torch.eye(nDim, device=mu.device) * noiseWMean
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


# Get the values of the hypergeometric function given a and b, for each
# value of non-centrality parameter, which is random-variable  dependent
def hyp1f1(a, b, c):
    """ For each element in c, compute the
    confluent hypergeometric function hyp1f1(a, b, c).
    ----------------
    Arguments:
    ----------------
      - a: First parameter of hyp1f1 (Scalar)
      - b: Second parameter of hyp1f1 (Scalar)
      - c: Vector, usually function of non-centrality parameters (nDim)
    ----------------
    Outputs:
    ----------------
      - hypVal: Value of hyp1f1 for each nc. (nDim)
    """
    hypVal = sps.hyp1f1(a, b, c)
    return hypVal


def iso_sm_weights(mu, sigma, nc=None):
    """ Compute the weights of the mean outer product and of the identity
    matrix in the isotropic projected Gaussian SM formula.
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
    # If precomputed weights are not given, compute them here
    nDim = len(mu)
    if nc is None:
        nc = non_centrality(mu=mu, sigma=sigma)
    hypFunNoise = hyp1f1(a=1, b=nDim/2+1, c=-nc/2)
    noiseW = hypFunNoise * (1/nDim)
    hypFunMean = hyp1f1(a=1, b=nDim/2+2, c=-nc/2)
    meanW = hypFunMean * (1/(nDim+2))
    return noiseW, meanW


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


def check_symmetric(B):
    """ Check if a matrix is symmetric
    ----------------
    Arguments:
    ----------------
      - B: Matrix to check. (nDim x nDim)
    ----------------
    Outputs:
    ----------------
      - isSymmetric: True if B is symmetric, False otherwise
    """
    isSymmetric = torch.allclose(B, B.t(), atol=5e-6)
    return isSymmetric


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


def sample_prnorm(mu, covariance, nSamples, B=None, c50=0):
    """ Sample from the random variable Y = X/(X'BX + c50)^0.5, where X~N(mu, covariance).
    -----------------
    Arguments:
    -----------------
      - mu: Mean. (nDim)
      - covariance: Covariance matrix. (nDim x nDim)
      - nSamples: Number of samples.
      - B: Matrix in the denominator. (nDim x nDim)
      - c50: Constant added to the denominator. Scalar
    -----------------
    Output:
    -----------------
      - prnorm: Samples from the generalized projected normal. (nSamples x nDim)
    """
    mu = mu.squeeze()
    covariance = covariance.squeeze()
    if B is None:
        B = torch.eye(len(mu))
    # Initialize Gaussian distribution to sample from
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    # Take nSamples
    X = dist.sample([nSamples])
    # Compute normalizing quadratic form
    if check_diagonal(B):
        D = torch.diagonal(B)
        q = torch.sqrt(torch.einsum('ni,i,in->n', X, D, X.t()) + c50)
    else:
        q = torch.sqrt(torch.einsum('ni,ij,jn->n', X, B, X.t()) + c50)
    # Normalize samples
    Y = torch.einsum('ni,n->ni', X, 1/q)
    return Y


def empirical_moments_prnorm(mu, covariance, nSamples, B=None, c50=0):
    """ Compute the mean, covariance and second moment of the projected normal
    Y = X/(X'BX + c50)^0.5, where X~N(mu, covariance).
    -----------------
    Arguments:
    -----------------
      - mu: Mean. (nDim)
      - covariance: Covariance matrix. (nDim x nDim)
      - nSamples: Number of samples.
      - B: Matrix in the denominator. (nDim x nDim)
      - c50: Constant added to the denominator. Scalar
    -----------------
    Output:
    -----------------
      - mean: Mean of the projected normal. (nDim)
      - covariance: Covariance of the projected normal. (nDim x nDim)
      - secondM: Second moment of the projected normal. (nDim x nDim)
    """
    samples = sample_prnorm(mu, covariance, nSamples=nSamples, B=B, c50=c50)
    mean = torch.mean(samples, dim=0)
    covariance = torch.cov(samples.t())
    secondM = torch.einsum('in,nj->ij', samples.t(), samples) / nSamples
    return {'mean':mean, 'covariance':covariance, 'secondM':secondM}


