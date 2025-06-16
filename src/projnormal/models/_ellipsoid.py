"""Class implementing ellipsoid matrix B with constraints."""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import projnormal.param_sampling as par_samp
import geotorch

from projnormal.linalg import spd_sqrt

from .constraints import Sphere, Positive, PositiveOffset

__all__ = [
  "Ellipsoid",
  "EllipsoidFixed",
]


def __dir__():
    return __all__


class Ellipsoid(nn.Module):
    """
    This class implements a symmetric positive definite matrix B
    that can be optimized.

    This class actually parametrizes the square root of the matrix B,
    B_sqrt. Matrix B_sqrt is parametrized by a diagonal matrix D = diag(rad)
    and n_dirs unit vectors v_1, ..., v_n_dirs that need not be orthogonal,
    together with their coefficients c_1, ..., c_n_dirs.
    B_sqrt is given by B_sqrt = D + sum_i c_i v_i v_i^T.

    Attributes
    -----------
      sqrt_diag : torch.Tensor, shape (n_dim)
          The value of the diagonal elements

      sqrt_vecs : torch.Tensor, shape (n_dirs, n_dim)
          The vectors parametrizing B.

      sqrt_coefs : torch.Tensor, shape (n_dirs)
          The coefficients of the vectors parametrizing B.
    """

    def __init__(self, n_dim, n_dirs=None, sqrt_coefs=None,
                 sqrt_vecs=None, sqrt_diag=1.0):
        """
        Initialize the ellipsoid matrix B.

        Parameters
        ----------
          n_dim : int
              The dimension of the matrix B.

          n_dirs : int, optional
              The number of vectors and coefficients that will be fit.
              If None, defaults to 1.

          sqrt_coefs : torch.Tensor, shape (n_dirs), optional
              The coefficients of the vectors for B_sqrt.
              If None, defaults to `B_rad_sq * 2`.

          sqrt_vecs : torch.Tensor, shape (n_dirs, n_dim), optional
              The vectors added to B_sqrt. If None, defaults to
              random orthogonal vectors.

          sqrt_diag : float, optional
              The value of the elements in the Diagonal matrix
              parametrizing B_sqrt. If None, defaults to 1.0.
        """
        super().__init__()

        # Parse inputs
        if sqrt_coefs is None and sqrt_vecs is None:
            if n_dirs is None:
                n_dirs = 1
            sqrt_coefs = torch.tensor([sqrt_diag * 4.0] * n_dirs)
            sqrt_vecs = par_samp.make_ortho_vectors(n_dim=n_dim, n_vec=n_dirs)

        elif sqrt_coefs is None:
            if sqrt_vecs.shape[-1] != n_dim:
                raise ValueError(
                  "sqrt_vecs must have the same number of columns as n_dim"
                )
            n_dirs = sqrt_vecs.shape[0]
            sqrt_coefs = torch.tensor([sqrt_diag * 4.0] * n_dirs)

        elif sqrt_vecs is None:
            n_dirs = sqrt_coefs.shape[0]
            sqrt_vecs = par_samp.make_ortho_vectors(n_dim=n_dim, n_vec=n_dirs)

        else:
            n_dirs = sqrt_coefs.shape[0]
            if sqrt_coefs.shape[0] != sqrt_vecs.shape[0]:
                raise ValueError(
                  "sqrt_coefs and sqrt_vecs must have the same number of rows"
                )

        # Define model attributes
        self.n_dim = n_dim
        self.n_dirs = n_dirs

        #self.sqrt_diag = nn.Parameter(sqrt_diag.clone())
        #parametrize.register_parametrization(self, "sqrt_diag", Positive())
        self.register_buffer("sqrt_diag", sqrt_diag.clone())

        self.sqrt_coefs = nn.Parameter(sqrt_coefs)
        parametrize.register_parametrization(self, "sqrt_coefs", PositiveOffset())
        self.sqrt_coefs = sqrt_coefs.clone()

        self.sqrt_vecs = nn.Parameter(sqrt_vecs)
        parametrize.register_parametrization(self, "sqrt_vecs", Sphere())
        self.sqrt_vecs = sqrt_vecs.clone()


    def get_B(self):
        """
        Return the ellipsoid matrix B.
        """
        B_sqrt = self.get_B_sqrt()
        return B_sqrt @ B_sqrt


    def get_B_logdet(self):
        """
        Return the log determinant of the ellipsoid matrix B.
        """
        B_logdet = torch.logdet(self.get_B_sqrt()) * 2
        return B_logdet


    def get_B_sqrt(self):
        """
        Return the square root of the ellipsoid matrix B.
        """
        diag = torch.eye(
          self.n_dim, dtype=self.sqrt_coefs.dtype, device=self.sqrt_coefs.device
        ) * self.sqrt_diag
        B_sqrt = diag + torch.einsum(
          "ij,i,im->jm", self.sqrt_vecs, self.sqrt_coefs, self.sqrt_vecs
        )
        return B_sqrt


    def get_B_sqrt_inv(self):
        """
        Return the square root of the ellipsoid matrix B.
        """
        B_sqrt_inv = torch.linalg.inv(self.get_B_sqrt())
        return B_sqrt_inv


    @property
    def B(self):
        return self.get_B()


    def __dir__(self):
        return ["sqrt_diag", "sqrt_vecs", "sqrt_coefs"]


class EllipsoidFixed(nn.Module):
    """
    This class implements a symmetric positive definite matrix B.
    It is not optimized, but rather fixed.

    Attributes
    -----------
      B : torch.Tensor, shape (n_dim)
          The common eigenvalue of the n_dim - 2 eigenvalues.
    """

    def __init__(self, B):
        """
        Initialize the ellipsoid matrix B.

        Parameters
        ----------
          B : torch.Tensor, shape (n_dim, n_dim)
              The ellipsoid matrix B.
        """
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


    def __dir__(self):
        return ["B", "eigvals", "eigvecs", "B_sqrt", "B_sqrt_inv", "B_logdet"]


