"""
Constraints to keep the distribution parameters in a valid region.
"""
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal

__all__ = [
  "Sphere",
  "Positive",
  "PositiveOffset",
  "Isotropic",
  "Diagonal",
]


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
        X : torch.Tensor, shape (..., n_dim)
            Input tensor in Euclidean space.

        Returns
        -------
        torch.Tensor, shape (..., n_dim)
            Normalized tensor lying on the sphere with shape.
        """
        X_normalized = X / X.norm(dim=-1, keepdim=True)
        return X_normalized

    def right_inverse(self, S):
        """
        Identity function to assign to parametrization.

        Parameters
        ----------
        S : torch.Tensor, shape (..., n_dim)
            Input tensor. Should be different from zero.

        Returns
        -------
        torch.Tensor, shape (..., n_dim)
            Returns the input tensor `S`.
        """
        return S


################
# POSITIVE NUMBER PARAMETRIZATION
################

def _softmax(X):
    """ Function to convert elements of X to positive numbers.
    The function applied is P = log(1 + exp(X)) + epsilon.

    Parameters
    ----------
    X: torch.Tensor, shape (...)
        Input tensor in the real line.

    Returns
    -------
    torch.Tensor, shape (...)
        Tensor with positive numbers.
    """
    epsilon = torch.tensor(1e-7, dtype=X.dtype)
    one = torch.tensor(1.0, dtype=X.dtype)
    P = torch.log(one + torch.exp(X)) + epsilon
    return P


def _inv_softmax(P):
    """ Inverse of softmax, converts positive numbers to reals.

    Parameters
    ----------
    P: torch.Tensor, shape (...)
        Input tensor with positive numbers.

    Returns
    -------
    torch.Tesor, shape (...)
        Tensor with real numbers.
    """
    epsilon = torch.tensor(1e-7, dtype=P.dtype)
    one = torch.tensor(1.0, dtype=P.dtype)
    X = torch.log(torch.exp(P - epsilon) - one) # Positive number
    return X


class Positive(nn.Module):
    """Constrains the input vector to be positive."""

    def forward(self, X):
        """
        Transform the input tensor to a positive number.

        Parameters
        ----------
        X : torch.Tensor, shape (...)
            Input vector in the real line

        Returns
        -------
        torch.Tensor, shape (...)
            Positive vector.
        """
        return _softmax(X)

    def right_inverse(self, P):
        """
        Inverse of the function to convert positive number to scalar.

        Parameters
        ----------
        P : torch.Tensor, shape (...)
            Input positive vector

        Returns
        -------
        torch.Tensor, shape (...)
            Scalar
        """
        return _inv_softmax(P)


class PositiveOffset(nn.Module):
    """Constrains the input number to be positive larger than an offset."""

    def __init__(self, offset=1.0):
        """
        Parameters
        ----------
        offset : float
            Offset to be added to the positive number.
        """
        super().__init__()
        self.register_buffer("offset", torch.as_tensor(offset))

    def forward(self, X):
        """
        Transform the input tensor to a positive number.

        Parameters
        ----------
        X : torch.Tensor, shape (...)
            Input number in the real line

        Returns
        -------
        torch.Tensor, shape (...)
            Positive number
        """
        return _softmax(X) + self.offset


    def right_inverse(self, P):
        """
        Inverse of the function to convert positive number to scalar.

        Parameters
        ----------
        P : torch.Tensor, shape (...)
            Input positive number

        Returns
        -------
        torch.Tensor, shape (...)
            Real number
        """
        return _inv_softmax(P - self.offset)


################
# SPD types parametrization
################

class Isotropic(nn.Module):
    """Constrains the matrix M to be of the form val*torch.eye(n_dim)."""

    def __init__(self, n_dim=None):
        """
        Parameters
        ----------
        n_dim : int
            Dimension of the matrix. If None, the parameter must
            be initialized using some matrix M.
        """
        super().__init__()
        self.n_dim = n_dim


    def forward(self, val):
        """
        Transform the input number into an isotropic matrix

        Parameters
        ----------
        val : torch.Tensor, shape (1,).
            Input number in the real line.

        Returns
        -------
        torch.Tensor, shape (n_dim, n_dim)
            Isotropic matrix with positive diagonal
        """
        val_pos = _softmax(val)
        return torch.diag(val_pos.expand(self.n_dim))


    def right_inverse(self, M):
        """
        Assign as val tr(M)/n_dim

        Parameters
        ----------
        M : torch.Tensor, shape (n_dim, n_dim).
            Input isotropic matrix.

        Returns
        -------
        torch.Tensor, shape (1,).
            Scalar value.
        """
        self.n_dim = M.shape[0]
        val_pos = torch.trace(M) / self.n_dim
        return _inv_softmax(val_pos)


class Diagonal(nn.Module):
    """Constrains the matrix M to be diagonal."""

    def forward(self, diagonal):
        """
        Transform the input vector into matrix.

        Parameters
        ----------
        diagonal : torch.Tensor (n_dim,).
            Input vector in the real line.

        Returns
        -------
        torch.Tensor (n_dim, n_dim).
            Diagonal matrix with positive diagonal of shape
        """
        diagonal_pos = _softmax(diagonal)
        return torch.diag(diagonal_pos)

    def right_inverse(self, M):
        """
        Assign as diagonal vector the diagonal entries of M.

        Parameters
        ----------
        M : torch.Tensor
            Input matrix. Must have positive diagonal entries.

        Returns
        -------
        torch.Tensor (n_dim,).
            Vector with diagonal entries of M.
        """
        diagonal_pos = torch.diagonal(M)
        return _inv_softmax(diagonal_pos)

