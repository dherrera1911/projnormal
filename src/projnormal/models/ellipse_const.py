"""Cass for the general projected normal distribution."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import projnormal.distribution as dist

from ._constraints import Positive
from .ellipse import ProjNormalEllipse


__all__ = [
  "ProjNormalEllipseConst",
]


def __dir__():
    return __all__


class ProjNormalEllipseConst(ProjNormalEllipse):
    """
    This class implements the general projected normal distribution but with
    projection to the interior of an ellipse.
    The variable Y following the distribution
    is defined as Y = X / sqrt(X'BX + const), where X~N(mean_x, covariance_x),
    B is a symmetric positive definite matrix, and const is a positive scalar.
    The class can be used to fit distribution parameters to data.

    Attributes
    -----------
      mean_x : torch.Tensor, shape (n_dim)
          Mean of X. It is constrained to the unit sphere.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
          Covariance of X. It is constrained to be symmetric positive definite.

      const : torch.Tensor, shape (1)
          The const denominator constant. It is constrained to be positive.

      B : torch.Tensor, shape (n_dim, n_dim)
          The ellipse matrix. It is constrained to be symmetric positive definite.

    Methods
    ----------
      moments():
          Compute the moments using a Taylor approximation.

      log_pdf() :
          Compute the value of the log pdf at given points.

      pdf() :
          Compute the value of the pdf at given points.

      moments_empirical() :
          Compute the moments by sampling from the distribution

      sample() :
          Sample points from the distribution.

      moment_match() :
          Fit the distribution parameters to the observed moments.

      max_likelihood() :
          Fit the distribution parameters to the observed data
          using maximum likelihood.
    """

    def __init__(
        self,
        n_dim=None,
        mean_x=None,
        covariance_x=None,
        const=None,
        n_dirs=None,
        B_sqrt_coefs=None,
        B_sqrt_vecs=None,
        B_sqrt_diag=1.0,
    ):
        """Initialize an instance of the ProjNormal class.

        Parameters
        ------------
          n_dim : int, optional
              Dimension of the underlying Gaussian distribution. If mean
              and covariance are provided, this is not required.

          mean_x : torch.Tensor, shape (n_dim), optional
              Mean of X. It is converted to unit norm. Default is random.

          covariance_x : torch.Tensor, shape (n_dim, n_dim), optional
              Initial covariance. Default is the identity.

          const : torch.Tensor, shape (1), optional
              The const denominator constant. Default is 1.

          n_dirs : int, optional
              Number of directions to use in the optimization. Default is 1.
              If `B_eigvals` is provided, it is ignored.

          B_eigvals : torch.Tensor, shape (n_dirs), optional
              Initial eigenvalues of the associated eigenvectors vectors in
              B_eigvecs.

          B_eigvecs : torch.Tensor, shape (n_dirs, n_dim), optional
              Initial eigenvectors of the ellipse matrix. Default is random
              orthogonal vectors.

          B_rad_sq : torch.Tensor, shape (), optional
              Initial eigenvalue of all directions not in B_eigvecs. Default
              is 1.
        """
        super().__init__(
          n_dim=n_dim,
          mean_x=mean_x,
          covariance_x=covariance_x,
          n_dirs=n_dirs,
          B_sqrt_coefs=B_sqrt_coefs,
          B_sqrt_vecs=B_sqrt_vecs,
          B_sqrt_diag=B_sqrt_diag,
        )

        # Parse const
        if const is None:
            const = torch.tensor(1.0)
        elif not torch.is_tensor(const) or const.dim() != 0 or const <= 0:
            if const.dim() == 1 and const.numel() == 1:
                const = const.squeeze()
            else:
                raise ValueError("const must be a positive scalar tensor.")

        self.const = nn.Parameter(const.clone())
        parametrize.register_parametrization(self, "const", Positive())


    def log_pdf(self, y):
        """
        Compute the log pdf at points y under the projected normal distribution.

        Parameters
        ----------------
          y : torch.Tensor, shape (n_points, n_dim)
              Points to evaluate the log pdf.

        Returns
        ----------------
          torch.Tensor, shape (n_points)
              Log PDF of the point. (n_points)
        """
        # Extract B matrices needed
        B = self.ellipse.get_B()
        B_sqrt = self.ellipse.get_B_sqrt()
        B_sqrt_ldet = self.ellipse.get_B_sqrt_inv()

        lpdf = dist.ellipse_const.pdf.log_pdf(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            y=y,
            const=self.const,
            B=B,
            B_sqrt=B_sqrt,
            B_sqrt_ldet=B_sqrt_ldet
        )
        return lpdf


    def __dir__(self):
        return super().__dir__() + ["const"]
