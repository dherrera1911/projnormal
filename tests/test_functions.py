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


