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
           (n_dim x n_dim)
      - mean: mean vector (n_dim)
    ----------------
    Outputs:
    ----------------
      - covariance: Covariance matrix. (n_dim x n_dim)
    """
    # Get the multiplying factor to make covariance unbiased
    covariance = second_moment - torch.einsum("d,b->db", mean, mean)
    return covariance


def cov_2_second_moment(covariance, mean):
    """
    Compute the second moment given the covariance and the mean vector.
    ----------------
    Arguments:
    ----------------
      - covariance: Covariance matrices. (n_dim x n_dim)
      - mean: mean vector (n_dim)
    ----------------
    Outputs:
    ----------------
      - second_moment: Second moment matrix.
           (n_dim x n_dim)
    """
    second_moment = covariance + torch.einsum("d,b->db", mean, mean)
    return second_moment


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
