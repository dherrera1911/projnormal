#################################
#
# PARAMETRIZATIONS FOR CONSTRAINED PARAMETERS
#
#################################

import numpy as np
import torch
import torch.nn as nn
import torch.linalg as LA
import scipy

################
# SPHERE PARAMETRIZATION
################

# Define the sphere constraint
class Sphere(nn.Module):
    def forward(self, X):
        """ Function to parametrize sphere vector S """
        # X is any vector
        S = X / LA.vector_norm(X) # Unit norm vector
        return S

    def right_inverse(self, S):
        """ Function to assign to parametrization""" 
        return S * S.shape[0]


################
# POSITIVE NUMBER PARAMETRIZATION
################

def softmax(X):
    epsilon = torch.tensor(1e-7, dtype=X.dtype)
    one = torch.tensor(1.0, dtype=X.dtype)
    P = torch.log(one + torch.exp(X)) + epsilon
    return P

def inv_softmax(P):
    epsilon = torch.tensor(1e-7, dtype=P.dtype)
    one = torch.tensor(1.0, dtype=P.dtype)
    X = torch.log(torch.exp(P - epsilon) - one) # Positive number
    return X

# Define positive number 
class SoftMax(nn.Module):
    def forward(self, X):
        # X is any scalar
        return softmax(X)

    def right_inverse(self, P):
        return inv_softmax(P)

################
# ORTHOGONAL MATRIX PARAMETRIZATION
################

def mat_2_skew(X):
    """ Function to convert matrix to skew symmetric matrix """
    return X.triu(1) - X.triu(1).t()

def mat_2_orthogonal(X):
    """ Function to convert matrix to orthogonal matrix """
    # Convert matrix to skew symmetric matrix
    S = mat_2_skew(X)
    # Convert skew symmetric matrix to orthogonal matrix
    Q = LA.matrix_exp(S)
    return Q

def orthogonal_2_triu(Q):
    """ Function to invert matrix to orthogonal transformation"""
    S = torch.as_tensor(scipy.linalg.logm(Q), dtype=Q.dtype)
    X = torch.triu(S, 1)
    return X

class Orthogonal(nn.Module):
    def forward(self, X):
        """ Function to parametrize orthogonal matrix Q """
        return mat_2_orthogonal(X)

    def right_inverse(self, Q):
        """ Function to assign to parametrization"""
        return orthogonal_2_triu(Q)


################
# SYMMETRIC POSITIVE DEFINITE MATRIX PARAMETRIZATION
################

# LOG CHOLESKY PARAMETRIZATION

class SPDLogCholesky(nn.Module):
    def __init__(self):
        super().__init__()

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

# SOFTMAX CHOLESKY PARAMETRIZATION

class SPDSoftmaxCholesky(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # Take strictly lower triangular matrix
        L_strict = X.tril(diagonal=-1)

        # Exponentiate diagonal elements
        D = torch.diag(softmax(X.diag()))

        # Make the Cholesky decomposition matrix
        L = L_strict + D

        # Generate SPD matrix
        SPD = torch.matmul(L, L.t())

        return SPD

    def right_inverse(self, SPD):
        L = torch.linalg.cholesky(SPD)
        L_strict = L.tril(diagonal=-1)
        D = torch.diag(inv_softmax(L.diag()))
        X = L_strict + D
        return X

# MATRIX LOGARITHM PARAMETRIZATION

def symmetric(X):
    # Use upper triangular part to construct symmetric matrix
    return X.triu() + X.triu(1).transpose(0, 1)

class SPDMatrixLog(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # Make symmetric matrix and exponentiate
        SPD = torch.linalg.matrix_exp(symmetric(X))
        return SPD

    def right_inverse(self, SPD):
        # Take logarithm of matrix
        dtype = SPD.dtype
        symmetric = scipy.linalg.logm(SPD.numpy())
        X = torch.triu(torch.tensor(symmetric))
        X = torch.as_tensor(X, dtype=dtype)
        return X


# SPECTRAL DECOMPOSITION PARAMETRIZATION

class SPDSpectral(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        # Turn matrix into orthogonal matrix
        Q = mat_2_orthogonal(X)
        # Turn vector into positive vector
        eigvals = softmax(Y)
        # Generate SPD matrix
        SPD = torch.matmul(torch.matmul(Q, torch.diag(eigvals)), Q.t())
        return SPD

    def right_inverse(self, SPD):
        # Take spectral decomposition of matrix
        eigvals, Q = torch.linalg.eigh(SPD)
        # Convert orthogonal matrix to triu
        X = orthogonal_2_triu(Q)
        # Convert positive vector to vector
        Y = inv_softmax(eigvals)
        return X, Y

