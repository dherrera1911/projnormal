import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.distributions.multivariate_normal as mvn
import scipy
import projected_normal.auxiliary as pna


# Make mean vectors
def make_mu_sin(n_dim):
    mu = torch.zeros(n_dim)
    angles = torch.rand(1) * torch.pi*2
    amplitude = torch.rand(1) + 0.1
    xVals = torch.linspace(0, torch.pi*2, n_dim)
    freq = torch.rand(1) * 2
    mu = torch.sin(xVals * freq + angles) * amplitude
    return mu

def make_mu(n_dim, mu_type='sin'):
    if mu_type == 'sin':
        mu = make_mu_sin(n_dim=n_dim)
    elif mu_type == 'ones':
        mu = torch.ones(n_dim)
    elif mu_type == 'sparse':
        mu = torch.zeros(n_dim)
        # Set values to 1
        mu[::3] = 1
    mu = mu / torch.norm(mu)
    return mu

# Function to create custom covariance matrices
def make_covariance(n_dim, cov_scale=1, cov_type='random'):
    """ Generate a covariance matrix with specified properties.
    -----------------
    Arguments:
    -----------------
      - n_dim: Dimensionality of the matrix.
      - cov_scale: For most cov_types, it is a scalar that multiplies the
        resulting covariance matrix. For cov_type='scaled', it is a vector
        that is used as the diagonal (+ baseline).
      - cov_type: Indicates the type of process that the covariance
        describes. Is a string that can take values:
          -'random': Random diagonal and off-diagonal elements.
          -'diagonal': Random diagonal elements. Off-diagonal elements are 0.
          -'isotropic': All diagonal elements are equal. Off-diagonal elements are 0.
    -----------------
    Output:
    -----------------
      - covariance: A covariance matrix of size n_dim x n_dim.
    """
    if cov_type=='random':
        covariance = make_spdm(n_dim)
    elif cov_type=='diagonal':
        covariance = torch.diag((torch.rand(n_dim)+0.5)*2-0.95) * cov_scale**2
    elif cov_type=='isotropic':
        covariance = torch.eye(n_dim) * cov_scale**2
    return covariance


def make_Aij(n_dim, i, j):
  A = torch.zeros(n_dim, n_dim)
  A[i,j] = 0.5
  A[j,i] = A[j,i] + 0.5
  return A


def make_spdm(n_dim):
    """ Make a random symmetric positive definite matrix
    ----------------
    Arguments:
    ----------------
      - n_dim: Dimension of matrix
    ----------------
    Outputs:
    ----------------
      - M: Random symmetric positive definite matrix
    """
    M = torch.randn(n_dim, n_dim * 2)
    M = torch.einsum('ij,jk->ik', M, M.transpose(0,1)) / n_dim**2
    assert pna.is_positive_definite(M)
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


