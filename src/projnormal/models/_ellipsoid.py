"""Class implementing ellipsoid matrix B with constraints."""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from projnormal.models._constraints import Positive

from torch.nn.utils.parametrizations import orthogonal


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
      rad_sq : torch.Tensor, shape (n_dim)
          The common eigenvalue of the n_dim - 2 eigenvalues.

      eigvecs : torch.Tensor, shape (n_dirs, n_dim)
          The two eigenvectors of the matrix B.

      eigvals : torch.Tensor, shape (n_dirs)
          The two eigenvalues of the matrix B.
    """

    def __init__(self, n_dim, n_dirs=2, rad_sq=1.0):
        super().__init__()
        self.n_dim = n_dim
        self.n_dirs = n_dirs

        self.rad_sq = nn.Parameter(rad_sq.clone())
        parametrize.register_parametrization(self, "rad_sq", Positive())

        eigvals = torch.ones(n_dirs) * rad_sq
        self.eigvals = nn.Parameter(eigvals)
        parametrize.register_parametrization(self, "eigvals", Positive())

        eigvecs = torch.randn(n_dirs, n_dim)
        self.eigvecs = nn.Parameter(eigvecs)
        orthogonal(self, "eigvecs")


    def get_B(self):
        """
        Return the ellipsoid matrix B.
        """
        term1 = torch.eye(self.n_dim) * self.rad_sq
        term2 = torch.einsum('ki,k,kj->ij', self.eigvecs, self.eigvals - self.rad_sq, self.eigvecs)
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

