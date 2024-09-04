import numpy as np
import torch
import torch.nn as nn
from torch.special import gammaln
import torch.distributions.multivariate_normal as mvn
import scipy
import scipy.special as sps


#################################
## CONVERT BETWEEN MOMENTS
#################################


def second_moment_2_cov(second_moment, mean):
    """
    Compute the covariance given the second moment and the
    mean vector.
    ----------------
    Arguments:
    ----------------
      - second_moment: Second moment matrix.
           (n_classes x nFilt x nFilt), or nFilt x nFilt)
      - mean: mean matrix (n_classes x nFilt, or nFilt)
    ----------------
    Outputs:
    ----------------
      - covariance: Covariance matrices. (n_classes x nFilt x nFilt)
    """
    # Get the multiplying factor to make covariance unbiased
    covariance = second_moment - torch.einsum("d,b->db", mean, mean)
    return covariance


def cov_2_corr(covariance):
    """
    Convert covariance matrices to correlation matrices
    ----------------
    Arguments:
    ----------------
      - covariance: Covariance matrices. (n_classes x nFilt x nFilt)
    ----------------
    Outputs:
    ----------------
      - correlation: Correlation matrices. (n_classes x nFilt x nFilt)
    """
    std = torch.sqrt(torch.diagonal(covariance))
    correlation = covariance / torch.einsum("a,b->ab", std, std)
    return correlation


#################################
## PROJECT TO B DIAGONAL BASIS
#################################

def diagonalize_B(mu, covariance, B):
    """
    Diagonalize the matrix B and project mu and covariance to the new basis.
    ----------------
    Arguments:
    ----------------
      - mu : Means of normal distributions X. (n_dim)
      - covariance : covariance of X. (n_dim x n_dim)
      - B : Matrix with normalization weights (n_dim x n_dim). If None, B=I.
      - c50 : Constant added to the denominator. Scalar
    ----------------
    Outputs:
    ----------------
      - mu : Projected mean. Shape (n_dim)
      - covariance : Projected covariance. Shape (n_dim x n_dim)
    """
    # Diagonalize B
    eigvals, eigvecs = torch.linalg.eigh(B)

    # Sort eigenvectors by eigenvalues
    eigvals, indices = torch.sort(eigvals, descending=True)
    eigvecs = eigvecs[:, indices]
    P = eigvecs

    # Project mu to the new basis
    mu_proj = torch.einsum("ij,j->i", P.t(), mu)

    # Project covariance to the new basis
    covariance_proj = torch.einsum("ij,jk,kl->il", P.t(), covariance, P)

    return mu_proj, covariance_proj, eigvals, P


def project_back(mu, covariance, P):
    """
    Project the mean back to the original basis.
    ----------------
    Arguments:
    ----------------
      - mu : Means of normal distributions X. (n_dim)
      - P : Matrix with normalization weights (n_dim x n_dim). If None, P=I.
    ----------------
    Outputs:
    ----------------
      - mu : Projected mean. Shape (n_dim)
    """
    # Project mu to the new basis
    mu_proj = torch.einsum("ij,j->i", P, mu)

    return mu_proj


#################################
## MATRIX CHECKS
#################################

def is_diagonal(matrix):
    """
    Check if a matrix is diagonal
    ----------------
    Arguments:
    ----------------
      - matrix: Matrix to check. (n_dim x n_dim)
    ----------------
    Outputs:
    ----------------
      - True if B is diagonal, False otherwise
    """
    return torch.allclose(matrix, torch.diag(torch.diagonal(matrix)))


def is_symmetric(matrix):
    """Check if a matrix is symmetric
    ----------------
    Arguments:
    ----------------
      - matrix : Matrix to check. (n_dim x n_dim)
    ----------------
    Outputs:
    ----------------
      - True if B is symmetric, False otherwise
    """
    return torch.allclose(matrix, matrix.t(), atol=5e-6)


def is_positive_definite(matrix):
    """Check if a matrix is symmetric
    ----------------
    Arguments:
    ----------------
      - matrix : Matrix to check. (n_dim x n_dim)
    ----------------
    Outputs:
    ----------------
      - True if B is symmetric, False otherwise
    """
    return torch.all(torch.linalg.eigh(matrix)[0] > 0)


#################################
## PARAMETRIZATIONS FOR CONSTRAINED PARAMETERS
#################################

# Define the sphere constraint
class Sphere(nn.Module):
    def forward(self, X):
        """ Function to parametrize sphere vector S """
        # X is any vector
        S = X / torch.norm(X) # Unit norm vector
        return S

    def right_inverse(self, S):
        """ Function to assign to parametrization""" 
        return S * S.shape[0]

# Define positive scalar constraint
class PositiveScalar(nn.Module):
    def forward(self, X):
        # X is any scalar
        P = torch.exp(X) # Positive number
        return P

    def right_inverse(self, P):
        X = torch.log(P) # Log of positive number
        return X

# Define positive scalar constraint
class SPDLogCholesky(nn.Module):
    def __init__(self, scale=1.0, dtype=torch.float32):
        super().__init__()
        self.scale = torch.as_tensor(scale, dtype=dtype)

    def forward(self, X):
        # Take strictly lower triangular matrix
        L_strict = X.tril(diagonal=-1)

        # Exponentiate diagonal elements
        D = torch.diag(torch.exp(X.diag()))

        # Make the Cholesky decomposition matrix
        L = L_strict + D

        # Generate SPD matrix
        SPD = torch.matmul(L, L.t())

        return SPD

    def right_inverse(self, SPD):
        L = torch.linalg.cholesky(SPD)
        L_strict = L.tril(diagonal=-1)
        D = torch.diag(torch.log(L.diag()))
        X = L_strict + D
        return X

# Define positive scalar constraint
def symmetric(X):
    # Use upper triangular part to construct symmetric matrix
    return X.triu() + X.triu(1).transpose(0, 1)

class SPDMatrixLog(nn.Module):
    def __init__(self, scale=1.0, dtype=torch.float32):
        super().__init__()
        self.scale = torch.as_tensor(scale, dtype=dtype)

    def forward(self, X):
        # Make symmetric matrix and exponentiate
        SPD = torch.linalg.matrix_exp(symmetric(X)) / self.scale
        return SPD

    def right_inverse(self, SPD):
        # Take logarithm of matrix
        dtype = SPD.dtype
        symmetric = scipy.linalg.logm(SPD.numpy() * self.scale.numpy())
        X = torch.triu(torch.tensor(symmetric))
        X = torch.as_tensor(X, dtype=dtype)
        return X


#################################
## SIMPLE COMPUTATIONS
#################################

def non_centrality(mu, sigma):
    """
    Non-centrality parameter, i.e. squared norm of standardized vector
    (mu/sigma). mu is the mean
    of the isotropic gaussian and sigma is the standard deviation.
    ----------------
    Arguments:
    ----------------
      - mu: Means of gaussians. (nX x n_dim)
      - sigma: standard deviation of noise. (n_dim)
    ----------------
    Outputs:
    ----------------
      - nc: Non-centrality parameter of each random variable, which is
            ||mu_normalized||^2, where mu_normalized = mu/sigma. Shape (nX)
    """
    mu_normalized = mu / sigma
    nc = mu_normalized.norm(dim=-1) ** 2
    return nc

def product_trace(A, B):
    """
    Efficiently compute tr(A*B).
    """
    return torch.einsum("ij,ji->", A, B)

def product_trace4(A, B, C, D):
    """
    Efficiently compute tr(A*B*C*D).
    """
    return torch.einsum("ij,jk,kl,li->", A, B, C, D)


