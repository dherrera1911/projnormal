"""Class for the general projected normal distribution with const denominator constant."""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import projnormal.distribution as dist

from ._constraints import Positive
from .projected_normal import ProjNormal


__all__ = [
  "ProjNormalConst",
]


def __dir__():
    return __all__


class ProjNormalConst(ProjNormal):
    """
    This class implements the general projected normal distirbution with
    a const denominator constant. The variable Y following the distribution
    is defined as Y = X / sqrt(||X||^2 + const), where X~N(mean_x, covariance_x).
    The class can be used to fit distribution parameters to data.

    Attributes
    -----------
      mean_x : torch.Tensor, shape (n_dim)
          Mean of X. It is constrained to the unit sphere.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
          Covariance of X. It is constrained to be symmetric positive definite.

      const : torch.Tensor, shape (1)
          The const denominator constant. It is constrained to be positive.

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
    ):
        """Initialize an instance of the ProjNormalConst class.

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
        """
        super().__init__(n_dim=n_dim, mean_x=mean_x, covariance_x=covariance_x)

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
        lpdf = dist.const.pdf.log_pdf(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            const=self.const,
            y=y,
        )
        return lpdf
