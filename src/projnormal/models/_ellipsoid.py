"""Class implementing ellipsoid matrix B with constraints."""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import projnormal.param_sampling as par_samp

from projnormal.models._constraints import Positive
from projnormal.ellipse_linalg import spd_sqrt
from torch.nn.utils.parametrizations import orthogonal


__all__ = [
  "Ellipsoid",
  "EllipsoidFixed",
]


def __dir__():
    return __all__


class Ellipsoid(nn.Module):
    """
    This class implements a symmetric positive definite matrix B
    of size n_dim x n_dim, where (n_dim - k) eigenvalues are
    equal to `rad_sq`. and the other eigenvalues along directions
    `eigvecs[0]` and `eigvecs[1]` are equal to `eigvals[0]` and `eigvals[1]`.

    The matrix B is given by:
    B = I * rad_sq
      + eigvecs[0] * eigvecs[0].T * (eigvals[0] - rad_sq)
      ...
      + eigvecs[k] * eigvecs[k].T * (eigvals[k] - rad_sq)

    where I is the identity matrix and {eigvecs[0] ... eigvecs[k]}
    are orthogonal.

    Attributes
    -----------
      B_rad_sq : torch.Tensor, shape (n_dim)
          The common eigenvalue of the n_dim - 2 eigenvalues.

      B_eigvecs : torch.Tensor, shape (n_dirs, n_dim)
          The two eigenvectors of the matrix B.

      B_eigvals : torch.Tensor, shape (n_dirs)
          The two eigenvalues of the matrix B.
    """

    def __init__(self, n_dim, n_dirs=None, B_eigvals=None, B_eigvecs=None, B_rad_sq=1.0):
        super().__init__()

        # Parse inputs
        if B_eigvals is None and B_eigvecs is None:
            if n_dirs is None:
                n_dirs = 1
            B_eigvals = torch.tensor([B_rad_sq] * n_dirs)
            B_eigvecs = par_samp.make_ortho_vectors(n_dim=n_dim, n_vec=n_dirs)

        elif B_eigvals is None:
            if B_eigvecs.shape[-1] != n_dim:
                raise ValueError(
                  "B_eigvecs must have the same number of columns as n_dim"
                )
            n_dirs = B_eigvecs.shape[0]
            B_eigvals = torch.ones(n_dirs) * B_rad_sq

        elif B_eigvecs is None:
            n_dirs = B_eigvals.shape[0]
            B_eigvecs = par_samp.make_ortho_vectors(n_dim=n_dim, n_vec=n_dirs)

        else:
            n_dirs = B_eigvals.shape[0]
            if B_eigvals.shape[0] != B_eigvecs.shape[0]:
                raise ValueError(
                  "B_eigvals and B_eigvecs must have the same number of rows"
                )

        self.n_dim = n_dim
        self.n_dirs = n_dirs

        self.rad_sq = nn.Parameter(B_rad_sq.clone())
        parametrize.register_parametrization(self, "rad_sq", Positive())

        self.eigvals = nn.Parameter(B_eigvals)
        parametrize.register_parametrization(self, "eigvals", Positive())
        self.eigvals = B_eigvals.clone()

        self.eigvecs = nn.Parameter(B_eigvecs)
        orthogonal(self, "eigvecs")
        self.eigvecs = B_eigvecs.clone()


    def get_B(self):
        """
        Return the ellipsoid matrix B.
        """
        term1 = torch.eye(self.n_dim) * self.rad_sq
        term2 = torch.einsum(
          'ki,k,kj->ij', self.eigvecs, self.eigvals - self.rad_sq, self.eigvecs
        )
        return term1 + term2


    def get_B_logdet(self):
        """
        Return the log determinant of the ellipsoid matrix B.
        """
        B_logdet = torch.log(self.rad_sq) * (self.n_dim - self.n_dirs) \
            + torch.log(self.eigvals).sum()
        return B_logdet


    def get_B_sqrt(self):
        """
        Return the square root of the ellipsoid matrix B.
        """
        rad = torch.sqrt(self.rad_sq)
        eigval_sqrt = torch.sqrt(self.eigvals)
        B_sqrt = torch.eye(self.n_dim) * rad + torch.einsum(
          'ki,k,kj->ij', self.eigvecs, eigval_sqrt - rad, self.eigvecs
        )
        return B_sqrt


    def get_B_sqrt_inv(self):
        """
        Return the square root of the ellipsoid matrix B.
        """
        rad = 1/torch.sqrt(self.rad_sq)
        eigval_sqrt_inv = 1/torch.sqrt(self.eigvals)
        B_sqrt_inv = torch.eye(self.n_dim) * rad + torch.einsum(
          'ki,k,kj->ij', self.eigvecs, eigval_sqrt_inv - rad, self.eigvecs
        )
        return B_sqrt_inv


class EllipsoidFixed(nn.Module):
    """
    This class implements a symmetric positive definite matrix B.

    Attributes
    -----------
      B : torch.Tensor, shape (n_dim)
          The common eigenvalue of the n_dim - 2 eigenvalues.

      eigvecs : torch.Tensor, shape (n_dim, n_dim)
          The eigenvectors of the matrix B.

      eigvals : torch.Tensor, shape (n_dim)
          The eigenvalues of the matrix B.
    """

    def __init__(self, B):
        super().__init__()

        self.n_dim = B.shape[0]

        self.register_buffer("_B", B)
        B_sqrt, B_sqrt_inv = spd_sqrt(B)
        self.register_buffer("B_sqrt", B_sqrt)
        self.register_buffer("B_sqrt_inv", B_sqrt_inv)
        self.register_buffer("B_logdet", torch.logdet(B))
        eigvals, eigvecs = torch.linalg.eigh(B)
        self.register_buffer("eigvals", eigvals)
        self.register_buffer("eigvecs", eigvecs)


    @property
    def B(self):
        return self._B


    @B.setter
    def B(self, value):
        self._B = value
        B_sqrt, B_sqrt_inv = spd_sqrt(value)
        self.B_sqrt = B_sqrt
        self.B_sqrt_inv = B_sqrt_inv
        self.B_logdet = torch.logdet(value)
        eigvals, eigvecs = torch.linalg.eigh(value)
        self.eigvals = eigvals
        self.eigvecs = eigvecs


    def get_B(self):
        """
        Return the ellipsoid matrix B.
        """
        return self.B


    def get_B_logdet(self):
        """
        Return the log determinant of the ellipsoid matrix B.
        """
        return self.B_logdet


    def get_B_sqrt(self):
        """
        Return the square root of the ellipsoid matrix B.
        """
        return self.B_sqrt


    def get_B_sqrt_inv(self):
        """
        Return the square root of the ellipsoid matrix B.
        """
        return self.B_sqrt_inv

