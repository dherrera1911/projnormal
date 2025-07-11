"""Constraints to keep the distribution parameters in a valid region."""
import torch
import torch.nn as nn

__all__ = [
  "Sphere",
  "Positive",
  "PositiveOffset",
  "Isotropic",
  "Diagonal",
  "ConstrainedSPD",
]


def __dir__():
    return __all__


################
# SPHERE PARAMETRIZATION
################

class Sphere(nn.Module):
    """Unit norm constraint."""

    def forward(self, X):
        """
        Normalize the input tensor so that it lies on the sphere.


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
    """
    Convert elements of X to positive numbers.
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
    """
    Inverse of softmax, converts positive numbers to reals.

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
    """Positive value constraint."""

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
    """Positive and larger than offset value constraint."""

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
    r"""Constrain matrix to be :math:`M = \lambda \cdot I_n`, where
    :math:`\lambda>0` and :math:`I_n` is the identity matrix.
    """

    def __init__(self, n_dim=None):
        """
        Parameters
        ----------
        n_dim : ``int``
            Dimension of the matrix. If None, the parameter must
            be initialized using some matrix M.
        """
        super().__init__()
        self.n_dim = n_dim


    def forward(self, val):
        """
        Transform the input number into an isotropic matrix.

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
        Assign as val tr(M)/n_dim.

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
    """Diagonal matrix with positive diagonal entries."""

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


class ConstrainedSPD(nn.Module):
    r"""Constrain matrix :math:`M` to be
    :math:`M = d \cdot I_n + W`, where :math:`d>0` is a fixed scalar,
    :math:`I_n` is the identity matrix, and
    :math:`W` is a symmetric positive semi-definite matrix of rank at most `k`.
    """

    def __init__(self, d=1.0, k=1):
        """
        Parameters
        ----------
        d : float
            Fixed positive scalar to be added to the diagonal of the matrix.

        k : int
            Rank of the symmetric positive semi-definite matrix W.
            Must be less than or equal to n_dim.
        """
        super().__init__()
        self.register_buffer("d", torch.as_tensor(d))
        self.k = k


    def forward(self, vecs):
        """
        Generate a constrained SPD matrix as eye + vecs @ vecs.T.

        Parameters
        ----------
        vecs : torch.Tensor (n_dim, k).
            Input vectors in euclidean space.

        Returns
        -------
        torch.Tensor (n_dim, n_dim).
            Constrained SPD matrix.
        """
        low_rank = torch.einsum("ik,jk->ij", vecs, vecs)
        Id = torch.eye(vecs.shape[0], device=vecs.device, dtype=vecs.dtype)
        return self.d * Id + low_rank


    def right_inverse(self, M):
        """
        Set the vectors that are used to make the low rank matrix W
        as the (scaled) k eigenvectors with largest eigenvalues of M.

        Parameters
        ----------
        M : torch.Tensor
            Input matrix. Must have positive diagonal entries.

        Returns
        -------
        torch.Tensor (n_dim, k).
            Vectors that can be used to reconstruct the SPD matrix.
        """
        # Eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(M)

        # Get the indices of the k largest eigenvalues
        k_largest_indices = torch.argsort(eigenvalues, descending=True)[:self.k]
        # Select the corresponding eigenvectors
        vecs = eigenvectors[:, k_largest_indices]

        # Scale by the square root of the eigenvalues minus the fixed scalar d
        vecs = torch.einsum(
          'ik,k->ik', vecs, torch.sqrt(eigenvalues[k_largest_indices] - self.d)
        )

        return vecs
