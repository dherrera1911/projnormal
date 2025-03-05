"""Class for the general projected normal distribution with c50 denominator constant."""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import projnormal.distribution as prnorm
import geotorch
from ._constraints import Sphere, Positive
from ._optim import lbfgs_loop, nadam_loop
from .general_projected_normal import ProjectedNormal


__all__ = [
  "ProjectedNormalConst",
]


def __dir__():
    return __all__


class ProjectedNormalConst(ProjectedNormal):
    """
    This class implements the general projected normal distirbution with
    a c50 denominator constant. The variable Y following the distribution
    is defined as Y = X / sqrt(||X||^2 + c50), where X~N(mean_x, covariance_x).
    The class can be used to fit distribution parameters to data.

    Attributes
    -----------
      mean_x : torch.Tensor, shape (n_dim)
          Mean of X. It is constrained to the unit sphere.

      covariance_x : torch.Tensor, shape (n_dim x n_dim)
          Covariance of X. It is constrained to be symmetric positive definite.

      c50 : torch.Tensor, shape (1)
          The c50 denominator constant. It is constrained to be positive.

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
        c50=None,
    ):
        """Initialize an instance of the C50ProjectedNormal class.

        Parameters
        ------------
          n_dim : int, optional
              Dimension of the underlying Gaussian distribution. If mean
              and covariance are provided, this is not required.

          mean : torch.Tensor, shape (n_dim), optional
              Mean of X. It is converted to unit norm. Default is random.

          covariance : torch.Tensor, shape (n_dim x n_dim), optional
              Initial covariance. Default is the identity.

          c50 : torch.Tensor, shape (1), optional
              The c50 denominator constant. Default is 1.
        """
        super().__init__(n_dim=n_dim, mean_x=mean_x, covariance_x=covariance_x)
        # Parse c50
        if c50 is None:
            c50 = torch.tensor(1.0)
        elif not torch.is_tensor(c50) or c50.dim() != 0 or c50 <= 0:
            if c50.dim() == 1 and c50.numel() == 1:
                c50 = c50.squeeze()
            else:
                raise ValueError("c50 must be a positive scalar tensor.")
        self.c50 = nn.Parameter(c50.clone())
        parametrize.register_parametrization(self, "c50", Positive())


    def log_pdf(self, y):
        """
        Compute the log pdf at points y under the projected normal distribution.

        Parameters
        ----------------
          y : torch.Tensor, shape (n_points x n_dim)
              Points to evaluate the log pdf.

        Returns
        ----------------
          torch.Tensor, shape (n_points)
              Log PDF of the point. (n_points)
        """
        lpdf = prnorm.c50.pdf.log_pdf(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            c50=self.c50,
            y=y,
        )
        return lpdf
