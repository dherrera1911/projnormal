"""
Constraints to keep the distribution parameters in a valid region.
"""
import torch
import torch.nn as nn
import torch.linalg as LA
import scipy

__all__ = ["Sphere", "SoftMax", "SPD"]


def __dir__():
    return __all__


################
# SPHERE PARAMETRIZATION
################

class Sphere(nn.Module):
    """Constrains the input tensor to lie on the sphere."""

    def forward(self, X):
        """
        Normalize the input tensor so that it lies on the sphere.

        The norm pooled across channels is computed and used to normalize the tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor in Euclidean space with shape (n_filters, n_dim).

        Returns
        -------
        torch.Tensor
            Normalized tensor lying on the sphere with shape
            (n_filters, n_dim).
        """
        X_normalized = X / X.norm(dim=-1, keepdim=True)
        return X_normalized

    def right_inverse(self, S):
        """
        Identity function to assign to parametrization.

        Parameters
        ----------
        S : torch.Tensor
            Input tensor. Should be different from zero.

        Returns
        -------
        torch.Tensor
            Returns the input tensor `S`.
        """
        return S


################
# POSITIVE NUMBER PARAMETRIZATION
################

def _softmax(X):
    """ Function to convert elements of X to positive numbers."""
    epsilon = torch.tensor(1e-7, dtype=X.dtype)
    one = torch.tensor(1.0, dtype=X.dtype)
    P = torch.log(one + torch.exp(X)) + epsilon
    return P

def _inv_softmax(P):
    """ Inverse of function to convert numbers to positive."""
    epsilon = torch.tensor(1e-7, dtype=P.dtype)
    one = torch.tensor(1.0, dtype=P.dtype)
    X = torch.log(torch.exp(P - epsilon) - one) # Positive number
    return X

class SoftMax(nn.Module):
    """Constrains the input tensor to lie on the sphere."""

    def forward(self, X):
        """
        Transform the input tensor to a positive number.

        Parameters
        ----------
        X : torch.Tensor. Scalar
            Input number in the real line

        Returns
        -------
        torch.Tensor
            Positive number.
        """
        return _softmax(X)

    def right_inverse(self, P):
        """
        Inverse of the function to convert positive number to scalar.

        Parameters
        ----------
        P : torch.Tensor. Positive number
            Input positive number.

        Returns
        -------
        torch.Tensor
            Scalar.
        """
        return _inv_softmax(P)


################
# SYMMETRIC POSITIVE DEFINITE MATRIX PARAMETRIZATION
################

class SPD(nn.Module):
    """Constrains the matrix to be symmetric positive definite."""

    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        """
        Transform vectors X and Y, parametrizing the orthogonal matrix
        of eigenvectors and the positive eigenvalues respectively
        to a symmetric positive definite matrix.

        Parameters
        ----------
        X : torch.Tensor. Shape ((n_dim-1) * (n_dim -2) / 2)
            Vector with upper triangular elements of n_dim x n_dim matrix.

        Y : torch.Tensor. Shape (n_dim)

        Returns
        -------
        torch.Tensor
            Symmetric positive definite matrix.
        """
        # Turn vector X into orthogonal matrix
        Q = _mat_2_orthogonal(_vec_2_triu(X))
        # Turn vector Y into positive vector
        eigvals = _softmax(Y)
        # Generate SPD matrix
        SPD = torch.einsum('ij,j,jk->ik', Q, eigvals, Q.t())
        return SPD

    def right_inverse(self, SPD):
        # Take spectral decomposition of matrix
        eigvals, Q = torch.linalg.eigh(SPD)
        # Convert orthogonal matrix to vector
        X = _triu_2_vec(
          _orthogonal_2_skew(Q)
        )
        # Convert positive vector to vector
        Y = _inv_softmax(eigvals)
        return X, Y


def _mat_2_orthogonal(X):
    """ Function to convert matrix to orthogonal matrix O, such that O^T O = I.

    Parameters
    ---------- 
    X : torch.Tensor. Shape (n_dim, n_dim)
        Input matrix.

    Returns
    -------
    torch.Tensor
        Orthogonal matrix.
    """
    skew = _mat_2_skew(X)
    # Convert skew symmetric matrix to orthogonal matrix
    orthogonal = torch.linalg.matrix_exp(skew)
    return orthogonal


def _mat_2_skew(X):
    """ Function to convert matrix to skew symmetric matrix.

    Parameters
    ----------
    X : torch.Tensor. Shape (n_dim, n_dim)
        Input matrix.

    Returns
    -------
    torch.Tensor
        Skew symmetric matrix.
    """
    return X.triu(1) - X.triu(1).t()


def _orthogonal_2_skew(orthogonal):
    """ Function to invert matrix to orthogonal transformation

    Parameters
    ----------
    orthogonal : torch.Tensor. Shape (n_dim, n_dim)
        Orthogonal matrix.

    Returns
    -------
    torch.Tensor
        Upper triangular matrix.
    """
    skew = torch.as_tensor(
      scipy.linalg.logm(orthogonal),
      dtype=orthogonal.dtype,
      device=orthogonal.device
    )
    return skew

def _triu_2_vec(X):
    """ Function to convert upper triangular matrix to vector.

    Parameters
    ----------
    X : torch.Tensor. Shape (n_dim, n_dim)
        Upper triangular matrix.

    Returns
    -------
    torch.Tensor
        Vector.
    """
    triu_inds = torch.triu_indices(X.shape[0], X.shape[1], offset=1)
    return X[triu_inds[0], triu_inds[1]]


def _vec_2_triu(vec):
    """ Function to convert vector to upper triangular matrix.

    Parameters
    ----------
    vec : torch.Tensor. Shape (n_dim * (n_dim - 1) / 2)
        Vector.

    Returns
    -------
    torch.Tensor
        Upper triangular matrix.
    """
    n_dim = int((2 * vec.shape[0]) ** 0.5 + 1)
    triu_inds = torch.triu_indices(n_dim, n_dim, offset=1)
    X = torch.zeros(n_dim, n_dim, dtype=vec.dtype, device=vec.device)
    X[triu_inds[0], triu_inds[1]] = vec
    return X
