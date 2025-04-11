"""
Constraints to keep the distribution parameters in a valid region.
"""
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal

__all__ = ["Sphere", "Positive", "PositiveOffset"]


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


class Positive(nn.Module):
    """Constrains the input number to be positive."""

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
        X : torch.Tensor. Scalar
            Input number in the real line

        Returns
        -------
        torch.Tensor
            Positive number.
        """
        return _softmax(X) + self.offset


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
        return _inv_softmax(P - self.offset)
