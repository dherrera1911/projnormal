##################################
##################################
#
## PROJECTED NORMAL MOMENTS AND SAMPLING
#
##################################
##################################

#### Terminology:
# X: Random variable with multidimensional Gaussian distribution
# mu: Mean of X
# sigma: Standard deviation of X (if isotropic)
# covariance: Covariance of X
# nDim: Dimensions of X
# y: Random variable with projected Gaussian distribution


import numpy as np
import scipy.special as sps
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.distributions import Normal
import torch.distributions.multivariate_normal as mvn
import scipy.linalg
import geotorch
import projected_normal as pn


##################################
##################################
#
## PROJECTED NORMAL PDF
#
##################################
##################################

def prnorm_pdf(mu, covariance, y):
    """ Compute the probability density function of a projected
    Gaussian distribution with parameters mu and covariance.
    ----------------
    Arguments:
    ----------------
      - mu: Mean of the non-projected Gaussian. Shape (n).
      - covariance: Covariance matrix of the non-projected Gaussian. Shape (n x n).
      - y: Points where to evaluate the PDF. Shape (n x m).
    ----------------
    Outputs:
    ----------------
      - pdf: PDF evaluated at y. Shape (m).
    """
    n = torch.tensor(mu.size(0))
    # Compute the precision matrix
    precision = torch.linalg.inv(covariance)
    # If y is a single point, add a dimension
    if len(y.size()) == 1:
        y = y.unsqueeze(0)
    # Compute the terms
    q1 = torch.einsum('i,ij,j->', mu, precision, mu)
    q2 = torch.einsum('i,ij,jk->k', mu, precision, y.t())
    q3 = torch.einsum('ki,ij,jk->k', y, precision, y.t())
    alpha = q2 / torch.sqrt(q3)
    M = M_value(alpha, nDim=n)
    pdf = 1/(2*np.pi)**((n-1)/2) * 1/torch.sqrt(torch.det(covariance)) * \
      q3**(-n/2) * torch.exp(1/2 * (alpha**2 - q1)) * M
    return pdf


def prnorm_logpdf(mu, covariance, y):
    """ Compute the log probability density function of a projected
    normal distribution with parameters mu and covariance.
    ----------------
    Arguments:
    ----------------
      - mu: Mean of the non-projected Gaussian. Shape (n).
      - covariance: Covariance matrix of the non-projected Gaussian. Shape (n x n).
      - y: Points where to evaluate the PDF. Shape (nPoints x n).
    ----------------
    Outputs:
    ----------------
      - lpdf: log-PDF evaluated at y. Shape (nPoints).
    """
    n = torch.tensor(mu.size(0))
    # Compute the precision matrix
    precision = torch.linalg.inv(covariance)
    # If y is a single point, add a dimension
    if len(y.size()) == 1:
        y = y.unsqueeze(0)
    # Compute the terms
    q1 = torch.einsum('i,ij,j->', mu, precision, mu)
    q2 = torch.einsum('i,ij,jk->k', mu, precision, y.t())
    q3 = torch.einsum('ki,ij,jk->k', y, precision, y.t())
    alpha = q2 / torch.sqrt(q3)
    M = M_value(alpha, nDim=n)
    # 
    term1 = -(n-1)/2 * torch.log(torch.tensor(2 * torch.pi))
    term2 = -0.5 * torch.logdet(covariance)
    term3 = -(n/2) * torch.log(q3)
    term4 = 0.5 * (alpha**2 - q1)
    term5 = torch.log(M)
    lpdf = term1 + term2 + term3 + term4 + term5
    return lpdf


def M_value(alpha, nDim):
    """ Compute value of function M in the projected normal pdf, with input alpha.
    The form of M depends on the dimension of the distribution, and it is computed
    with a recursive formula here.
    ----------------
    Arguments:
    ----------------
      - alpha: Input to function M (n).
      - nDim: Dimension of the non-projected Gaussian.
    ----------------
    Outputs:
    ----------------
      - M_vals: Value of M(alpha) (n).
    """
    # Create a standard normal distribution
    normal_dist = Normal(0, 1)
    # Calculate the cumulative distribution function (CDF) of alpha
    normcdf = normal_dist.cdf(alpha)
    # Calculate the probability density function (PDF) of alpha
    normpdf = torch.exp(normal_dist.log_prob(alpha))
    if nDim == 1:
        return normcdf
    elif nDim == 2:
        return alpha * normcdf + normpdf
    elif nDim==3:
        return (1 + alpha**2) * normcdf + alpha * normpdf
    else:
        M_vals = [normcdf, alpha * normcdf + normpdf]
        for i in range(3, nDim+1):
            M_next = alpha * M_vals[-1] + (i-1) * M_vals[-2]
            M_vals.append(M_next)
        return M_vals[-1]


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
        if pn.check_diagonal(B):
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
    dfdu2Val = dfdu2(u=mu, v=meanV, b=weights, c50=c50)
    dfdv2Val = dfdv2(u=mu, v=meanV, b=weights, c50=c50)
    dfdudvVal = dfdudv(u=mu, v=meanV, b=weights, c50=c50)
    ### 0th order term
    term0 = f0(u=mu, v=meanV, b=weights, c50=c50)
    ### Compute Taylor approximation
    YExpected = term0 + 0.5*dfdu2Val*variances + 0.5*dfdv2Val*varV + dfdudvVal*covV
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
    muD = pn.quadratic_form_mean(mu=mu, covariance=covariance, M=B)
    muD = muD + c50
    varD = pn.quadratic_form_var(mu=mu, covariance=covariance, M=B)
    # Compute covariance between numerator and denominator for each
    # matrix A^{ij}
    if B is not None:
        covB = torch.einsum('ij,jk->ik', covariance, B)
    else:
        covB = covariance
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
        Bcovariance = covariance
    # Compute the variance of X'BX
    varX2 = 2 * pn.product_trace(Bcovariance, Bcovariance) + \
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
    nc = pn.non_centrality(mu, sigma)
    gammaln1 = gammaln((nDim+1)/2)
    gammaln2 = gammaln(nDim/2+1)
    gammaRatio = 1/(np.sqrt(2)*sigma) * torch.exp(gammaln1 - gammaln2)
    hypFunRes = sps.hyp1f1(1/2, nDim/2+1, -nc/2)
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
    hypFunNoise = sps.hyp1f1(a=1, b=nDim/2+1, c=-nc/2)
    noiseW = hypFunNoise * (1/nDim)
    hypFunMean = sps.hyp1f1(a=1, b=nDim/2+2, c=-nc/2)
    meanW = hypFunMean * (1/(nDim+2))
    return noiseW, meanW


#### MATRIX CHECKS ####
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
## PROJECTED NORMAL SAMPLING
#
##################################
##################################


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
    if pn.check_diagonal(B):
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


##################################
##################################
#
## PROJECTED NORMAL FITTING
#
##################################
##################################


#### Class for the projected normal distribution with learnable parameters
class ProjNorm(nn.Module):
    """
    The ProjNorm class implements a projected normal distirbution
    with several functionalities. The parameters of the distribution
    can be fitted to a given mean and covariance using the method fit.
    ----------------
    Inputs:
    ----------------
      - nDim: Dimension of the distribution.
      - muInit: Initial mean. Default is zero.
      - covInit: Initial covariance. Default is the identity.
    ----------------
    Attributes:
    ----------------
      - mu: Mean of the distribution. Shape (n).
            Constrained to the sphere.
      - cov: Covariance of the distribution. Shape (n x n).
            Constrained to the positive definite cone.
    ----------------
    Methods:
    ----------------
      - moment_match: Optimize distribution parameters to match observed moments
      - ml_fit: Fit the distribution to observed data maximizing the log likelihood
      - get_moments: Compute the Taylor approximation to the normalized (Y) mean
                    and covariance for the attribute mean and covariance.
      - logpdf: Compute the log probability of given points under the distribution.
      - sample: Sample from the distribution.
      - empiral_moments: Compute the mean and covariance the normalized
                    (Y) mean and covariance by sampling from the distribution.
    """
    def __init__(self, nDim, muInit=None, covInit=None, requires_grad=True,
                 dtype=torch.float32):
        super().__init__()
        if muInit is None:
            muInit = torch.randn(nDim, dtype=dtype)
            muInit = muInit / torch.norm(muInit)
        else:
            muInit = torch.as_tensor(muInit, dtype=dtype)
            muInit = muInit / torch.norm(muInit)
        if covInit is None:
            covInit = torch.eye(nDim, dtype=dtype)
        else:
            covInit = torch.as_tensor(covInit, dtype=dtype)
        if requires_grad:
            self.mu = nn.Parameter(muInit.clone())
            geotorch.sphere(self, "mu")
            self.mu = muInit
            self.cov = nn.Parameter(covInit.clone())
            geotorch.positive_definite(self, "cov")
            self.cov = covInit
        else:
            self.mu = muInit
            self.cov = covInit


    def get_moments(self):
        """ Compute the normalized (Y) mean and covariance for the
        attribute mean and covariance.
        ----------------
        Outputs:
        ----------------
          - muOut: Normalized mean. Shape (n).
          - covOut: Normalized covariance. Shape (n x n).
        """
        muOut = prnorm_mean_taylor(mu=self.mu, covariance=self.cov)
        smOut = prnorm_sm_taylor(mu=self.mu, covariance=self.cov)
        covOut = pn.secondM_2_cov(secondM=smOut, mean=muOut)
        return muOut, covOut


    def logpdf(self, y):
        """ Compute the log probability of a given point under the distribution.
        ----------------
        Inputs:
        ----------------
          - y: Points to evaluate. Shape (nPoints x n).
        ----------------
        Outputs:
        ----------------
          - lpdf: Log probability of the point. (nPoints)
        """
        lpdf = prnorm_logpdf(mu=self.mu, covariance=self.cov, y=y)
        return lpdf


    def moment_match(self, muObs, covObs, nIter=100, lr=0.1, lrGamma=0.7, decayIter=10,
            lossType="mse", nCycles=1, cycleMult=0.25, optimizerType='NAdam'):
        """ Optimize the parameters of the distribution to match the observed
        moments.
        ----------------
        Inputs:
        ----------------
          - mu: Mean to fit (observed). Shape (n).
          - cov: Covariance to fit (observed). Shape (n x n).
          - nIter: Number of iterations. Default is 100.
          - lr: Learning rate. Default is 0.01.
          - lrGamma: Learning rate decay. Default is 0.8.
          - decayIter: Decay the learning rate every decayIter iterations. Default is 10.
          - lossType: Loss function to use. Options are "norm", "mse", and "wasserstein".
          - nCycles: Number of learning-rate cycles. Default is 1.
          - cycleMult: Multiplicative factor for learning rate in each cycle. Default is 0.25.
          - optimizerType: Optimizer to use. Options are "SGD", "Adam", and "NAdam".
        ----------------
        Outputs:
        ----------------
          - lossVec: Vector of losses at each iteration.
        """
        # Define the loss function
        if lossType == "norm":
            lossFun = loss_norm
        elif lossType == "mse":
            lossFun = loss_mse
        elif lossType == "wasserstein":
            lossFun = loss_wasserstein
        else:
            raise ValueError('Loss function not recognized.')
        # Initialize the loss list
        lossList = []
        for c in range(nCycles):
            lrCycle = lr * cycleMult**c # Decrease the initial learning rate
            # Initialize the optimizer
            if optimizerType == 'SGD':
                optimizer = torch.optim.SGD(self.parameters(), lr=lrCycle)
            elif optimizerType == 'Adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=lrCycle)
            elif optimizerType == 'NAdam':
                optimizer = torch.optim.NAdam(self.parameters(), lr=lrCycle)
            # Initialize the scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decayIter,
                                                        gamma=lrGamma)
            # Iterate over the number of iterations
            for i in range(nIter):
                # Zero the gradients
                optimizer.zero_grad()
                muOut, covOut = self.get_moments()
                loss = lossFun(muOut, covOut, muObs, covObs)
                # Compute the gradients
                loss.backward()
                # Optimize the parameters
                optimizer.step()
                # Append the loss to the list
                lossList.append(loss.item())
                # Step the scheduler
                scheduler.step()
        lossVec = torch.tensor(lossList)
        return lossVec


    def ml_fit(self, y, nIter=100, lr=0.1, lrGamma=0.7, decayIter=10,
            nCycles=1, cycleMult=0.25, optimizerType='NAdam'):
        """ Optimize the distribution parameters to maximize the log-likelihood
        of the observed data.
        ----------------
        Inputs:
        ----------------
          - y: Observed points from the distribution. Shape (nPoints x n).
          - nIter: Number of iterations. Default is 100.
          - lr: Learning rate. Default is 0.01.
          - lrGamma: Learning rate decay. Default is 0.8.
          - decayIter: Decay the learning rate every decayIter iterations. Default is 10.
          - lossType: Loss function to use. Options are "norm", "mse", and "wasserstein".
          - nCycles: Number of learning-rate cycles. Default is 1.
          - cycleMult: Multiplicative factor for learning rate in each cycle. Default is 0.25.
          - optimizerType: Optimizer to use. Options are "SGD", "Adam", and "NAdam".
        ----------------
        Outputs:
        ----------------
          - nlog_likelihood: Vector of negative log_likelihood at each iteration.
        """
        # Initialize the loss list
        nllList = []
        for c in range(nCycles):
            lrCycle = lr * cycleMult**c # Decrease the initial learning rate
            # Initialize the optimizer
            if optimizerType == 'SGD':
                optimizer = torch.optim.SGD(self.parameters(), lr=lrCycle)
            elif optimizerType == 'Adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=lrCycle)
            elif optimizerType == 'NAdam':
                optimizer = torch.optim.NAdam(self.parameters(), lr=lrCycle)
            # Initialize the scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decayIter,
                                                        gamma=lrGamma)
            # Iterate over the number of iterations
            for i in range(nIter):
                # Zero the gradients
                optimizer.zero_grad()
                nlog_likelihood = - torch.sum(self.logpdf(y))
                # Compute the gradients
                nlog_likelihood.backward()
                # Optimize the parameters
                optimizer.step()
                # Append the loss to the list
                nllList.append(nlog_likelihood.item())
                # Step the scheduler
                scheduler.step()
        nlog_likelihood = torch.tensor(nllList)
        return nlog_likelihood


    def sample(self, nSamples):
        """ Sample from the distribution.
        ----------------
        Inputs:
        ----------------
          - nSamples: Number of samples to draw.
        ----------------
        Outputs:
        ----------------
          - samples: Samples from the distribution. Shape (nSamples x n).
        """
        with torch.no_grad():
            samples = sample_prnorm(mu=self.mu, covariance=self.cov,
                                    B=None, c50=0.0, nSamples=nSamples)
        return samples


    def empirical_moments(self, nSamples):
        """ Compute the mean and covariance the normalized (Y) mean and covariance
        by sampling from the distribution.
        ----------------
        Inputs:
        ----------------
          - nSamples: Number of samples to draw.
        ----------------
        Outputs:
        ----------------
          - muOut: Empirical mean. Shape (n).
          - covOut: Empirical covariance. Shape (n x n).
        """
        with torch.no_grad():
            statsDict = empirical_moments_prnorm(mu=self.mu, covariance=self.cov,
                                                 B=None, c50=0.0, nSamples=nSamples)
        muOut = statsDict["mean"]
        covOut = statsDict["covariance"]
        return muOut, covOut


def loss_norm(mu1, cov1, mu2, cov2):
    """ Compute the norm of the difference between the observed and predicted moments.
    ----------------
    Inputs:
    ----------------
      - mu: Predicted mean. Shape (n).
      - cov: Predicted covariance. Shape (n x n).
      - muObs: Observed mean. Shape (n).
      - covObs: Observed covariance. Shape (n x n).
    ----------------
    Outputs:
    ----------------
      - loss: Loss between the observed and predicted moments.
    """
    loss = (mu1 - mu2).norm() + (cov1 - cov2).norm()
    return loss


def loss_mse(mu1, cov1, mu2, cov2):
    """ Compute the mean squared error between the observed and predicted moments.
    ----------------
    Inputs:
    ----------------
      - mu: Predicted mean. Shape (n).
      - cov: Predicted covariance. Shape (n x n).
      - muObs: Observed mean. Shape (n).
      - covObs: Observed covariance. Shape (n x n).
    ----------------
    Outputs:
    ----------------
      - loss: Loss between the observed and predicted moments.
    """
    loss = (mu1 - mu2).pow(2).sum() + (cov1 - cov2).pow(2).sum()
    return loss


def loss_wasserstein(mu1, cov1, mu2, cov2):
    """ Compute the Wasserstein distance between the observed and predicted moments.
    ----------------
    Inputs:
    ----------------
      - mu: Predicted mean. Shape (n).
      - cov: Predicted covariance. Shape (n x n).
      - muObs: Observed mean. Shape (n).
      - covObs: Observed covariance. Shape (n x n).
    ----------------
    Outputs:
    ----------------
      - loss: Loss between the observed and predicted moments.
    """
    loss = (mu1 - mu2).pow(2).sum() + bw_dist_sq(cov1, cov2)
    return loss


# Make a function that computes the matrix square root of a positive
# definite matrix, with a backward pass.
class MatrixSquareRoot(torch.autograd.Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)
            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input

# Create the function to compute the matrix square root.
sqrtm = MatrixSquareRoot.apply

def bw_dist_sq(mat1, mat2):
    """ Wasserstein distance between two positive definite matrices
    ----------------
    Inputs:
    ----------------
      - mat1: Positive definite matrix. Shape (n x n).
      - mat2: Positive definite matrix. Shape (n x n).
    ----------------
    Outputs:
    ----------------
      - squared_dist: Squared Wasserstein distance between the matrices.
    """
    product = torch.matmul(mat1, mat2)
    sqrt_product = sqrtm(product)
    trace_a = torch.trace(mat1)
    trace_b = torch.trace(mat2)
    trace_prod = torch.trace(sqrt_product)
    squared_dist = trace_a + trace_b - 2.0 * trace_prod
    return squared_dist


