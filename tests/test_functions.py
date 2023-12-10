import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.distributions.multivariate_normal as mvn
import scipy


# Make mean vectors
def make_mu_sin(nX, nDim):
    mu = torch.zeros(nX, nDim)
    angles = torch.rand(nX) * torch.pi*2
    amplitudes = torch.rand(nX) + 0.1
    xVals = torch.linspace(0, torch.pi*2, nDim)
    for s in range(nX):
        freq = torch.rand(1) * 10
        mu[s,:] = torch.sin(xVals * freq + angles[s]) * amplitudes[s]
    return mu


# Function to create custom covariance matrices
def make_covariance(nDim, sigmaScale=1, covType='random'):
    """ Generate a covariance matrix with specified properties.
    -----------------
    Arguments:
    -----------------
      - nDim: Dimensionality of the matrix.
      - sigmaScale: For most covTypes, it is a scalar that multiplies the
        resulting covariance matrix. For covType='scaled', it is a vector
        that is used as the diagonal (+ baseline).
      - covType: Indicates the type of process that the covariance
        describes. Is a string that can take values:
          -'random': Random diagonal and off-diagonal elements.
          -'diagonal': Random diagonal elements. Off-diagonal elements are 0.
          -'isotropic': All diagonal elements are equal. Off-diagonal elements are 0.
    -----------------
    Output:
    -----------------
      - covariance: A covariance matrix of size nDim x nDim.
    """
    if covType=='random':
        isPD = False
        while not isPD:
            diag = torch.diag(torch.rand(nDim) * sigmaScale**2)
            randMat = torch.randn(int(nDim), int(nDim))  # Make random matrix
            svd = torch.linalg.svd(randMat)
            orth = svd[0]  # Orthogonal matrix
            covariance = torch.einsum('ij,jk,kl->il', orth, diag, orth.T)
            eigVals = torch.real(torch.linalg.eigvals(covariance))
            isPD = all(eigVals > 0)
    elif covType=='diagonal':
        covariance = torch.diag((torch.rand(nDim)+0.5)*2-1) * sigmaScale**2
    elif covType=='isotropic':
        covariance = torch.eye(nDim) * sigmaScale**2
    return covariance


def empirical_moments_isotropic_gaussian_norm(mu, covariance, nSamples):
    nX, nDim = mu.shape
    norm = torch.zeros(nX)      # ||X||
    norm2 = torch.zeros(nX)     # ||X||^2
    invNorm = torch.zeros(nX)   # 1/||X||
    invNorm2 = torch.zeros(nX)  # 1/||X||^2
    for s in range(nX):
        dist = mvn.MultivariateNormal(loc=mu[s,:], covariance_matrix=covariance)
        X = dist.sample([nSamples])
        X2 = torch.einsum('ni,ni->n', X, X)
        norm[s] = torch.mean(torch.sqrt(X2))
        norm2[s] = torch.mean(X2)
        invNorm[s] = torch.mean(1/torch.sqrt(X2))
        invNorm2[s] = torch.mean(1/X2)
    return {'norm': norm, 'norm2': norm2, 'invNorm': invNorm, 'invNorm2': invNorm2}


def sample_projected_gaussian(mu, covariance, nSamples, B=None):
    assert mu.dim()==1 or mu.shape[0] == 1, "Only a single mean vector is supported."
    mu = mu.squeeze()
    covariance = covariance.squeeze()
    nDim = mu.shape[0]
    if B is None:
        B = torch.eye(nDim)
    # Initialize Gaussian distribution to sample from
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    # Sample from it nSamples
    X = dist.sample([nSamples])
    # Compute normalizing quadratic form
    q = torch.sqrt(torch.einsum('ni,ij,nj->n', X, B, X))
    # Normalize samples
    Y = torch.einsum('ni,n->ni', X, 1/q)
    return Y


def empirical_moments_projected_gaussian(mu, covariance, nSamples, B=None):
    nX, nDim = mu.shape
    means = torch.zeros(nX, nDim)
    covariances = torch.zeros(nX, nDim, nDim)
    secondM = torch.zeros(nX, nDim, nDim)
    for s in range(nX):
        samples = sample_projected_gaussian(mu[s,:], covariance, nSamples=nSamples, B=B)
        means[s,:] = torch.mean(samples, dim=0)
        covariances[s,:,:] = torch.cov(samples.T)
        secondM[s,:,:] = torch.einsum('ni,nj->ij', samples, samples) / nSamples
    return means, covariances, secondM


def empirical_moments_quadratic_form(mu, covariance, M, nSamples):
    """ Compute the mean and variance of the quadratic form
    qf = X^T M X
    where X is a multivariate Gaussian with mean mu and covariance matrix
    covariance.
    -----------------
    Arguments:
    -----------------
      - mu: Mean vector of the Gaussian.
      - covariance: Covariance matrix of the Gaussian.
      - M: Matrix of the quadratic form.
      - nSamples: Number of samples to use to compute the moments.
    -----------------
    Output:
    -----------------
      - means: Mean of the quadratic form.
      - var: Variance of the quadratic form.
    """
    nX, nDim = mu.shape
    means = torch.zeros(nX)
    var = torch.zeros(nX)
    secondM = torch.zeros(nX)
    for s in range(nX):
        dist = mvn.MultivariateNormal(loc=mu[s,:], covariance_matrix=covariance)
        X = dist.sample([nSamples])
        qf = torch.einsum('ni,ij,nj->n', X, M, X)
        means[s] = torch.mean(qf)
        var[s] = torch.var(qf)
        secondM[s] = torch.mean(qf**2)
    return {'mean': means, 'var':var, 'secondM': secondM}


def empirical_covariance_quadratic_form(mu, covariance, M1, M2, nSamples):
    """ Compute the mean and variance of the quadratic form
    qf = X^T M X
    where X is a multivariate Gaussian with mean mu and covariance matrix
    covariance.
    -----------------
    Arguments:
    -----------------
      - mu: Mean vector of the Gaussian.
      - covariance: Covariance matrix of the Gaussian.
      - M: Matrix of the quadratic form.
      - nSamples: Number of samples to use to compute the moments.
    -----------------
    Output:
    -----------------
      - means: Mean of the quadratic form.
      - var: Variance of the quadratic form.
    """
    nX, nDim = mu.shape
    cov = torch.zeros(nX)
    for s in range(nX):
        dist = mvn.MultivariateNormal(loc=mu[s,:], covariance_matrix=covariance)
        X = dist.sample([nSamples])
        qf1 = torch.einsum('ni,ij,nj->n', X, M1, X)
        qf2 = torch.einsum('ni,ij,nj->n', X, M2, X)
        cov[s] = torch.cov(torch.cat((qf1.unsqueeze(0), qf2.unsqueeze(0))))[0,1]
    return cov



