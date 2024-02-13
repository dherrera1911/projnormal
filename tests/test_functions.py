import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.distributions.multivariate_normal as mvn
import scipy


# Make mean vectors
def make_mu_sin(nDim):
    mu = torch.zeros(nDim)
    angles = torch.rand(1) * torch.pi*2
    amplitude = torch.rand(1) + 0.1
    xVals = torch.linspace(0, torch.pi*2, nDim)
    freq = torch.rand(1) * 10
    mu = torch.sin(xVals * freq + angles) * amplitude
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
    nDim = len(mu)
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    X = dist.sample([nSamples])
    X2 = torch.einsum('ni,ni->n', X, X)
    norm = torch.mean(torch.sqrt(X2))
    norm2 = torch.mean(X2)
    invNorm = torch.mean(1/torch.sqrt(X2))
    invNorm2 = torch.mean(1/X2)
    return {'norm': norm, 'norm2': norm2, 'invNorm': invNorm, 'invNorm2': invNorm2}


def sample_prnorm(mu, covariance, nSamples, B=None):
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


def empirical_moments_prnorm(mu, covariance, nSamples, B=None):
    nDim = len(mu)
    samples = sample_prnorm(mu, covariance, nSamples=nSamples, B=B)
    mean = torch.mean(samples, dim=0)
    covariance = torch.cov(samples.T)
    secondM = torch.einsum('ni,nj->ij', samples, samples) / nSamples
    return mean, covariance, secondM


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
    nDim = len(mu)
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    X = dist.sample([nSamples])
    qf = torch.einsum('ni,ij,nj->n', X, M, X)
    mean = torch.mean(qf)
    var = torch.var(qf)
    secondM = torch.mean(qf**2)
    return {'mean': mean, 'var':var, 'secondM': secondM}


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
      - cov: Covariance between the two quadratic forms.
    """
    nDim = len(mu)
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    X = dist.sample([nSamples])
    qf1 = torch.einsum('ni,ij,nj->n', X, M1, X)
    qf2 = torch.einsum('ni,ij,nj->n', X, M2, X)
    cov = torch.cov(torch.cat((qf1.unsqueeze(0), qf2.unsqueeze(0))))[0,1]
    return cov


def make_Aij(dim, i, j):
  A = torch.zeros(dim, dim)
  A[i,j] = 0.5
  A[j,i] = A[j,i] + 0.5
  return A


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


def make_random_covariance(variances, eig):
    """ Create a random covariance matrix with the given variances, and
    whose correlation matrix has the given eigenvalues.
    ----------------
    Arguments:
    ----------------
      - variances: Vector of variances. Shape (n).
      - eig: Vector of eigenvalues of correlation matrix. Shape (n).
    ----------------
    Outputs:
    ----------------
      - covariance: Random covariance matrix. Shape (n x n).
    """
    # Make sume of eigenvalues equal to n
    eig = np.array(eig)
    eig = eig / eig.sum() * len(eig)
    randCorr = scipy.stats.random_correlation(eig).rvs()
    randCorr = torch.as_tensor(randCorr, dtype=torch.float32)
    D = torch.diag(torch.sqrt(variances))
    covariance = torch.einsum('ij,jk,kl->il', D, randCorr, D)
    return covariance


