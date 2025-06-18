"""Class for the general projected normal distribution."""
from abc import ABC, abstractmethod
import geotorch
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import projnormal.distribution as dist

from ..linalg import spd_sqrt
from .constraints import Positive
from .projected_normal import ProjNormal


__all__ = [
  "ProjNormalEllipse",
]


def __dir__():
    return __all__


class ProjNormalEllipse(ProjNormal):
    """
    Implementation of the general projected normal distribution with
    projection onto an ellipse given by Y'BY = 1.
    The variable Y following the distribution
    is defined as Y = X / sqrt(X'BX), where X~N(mean_x, covariance_x)
    and B is a symmetric positive definite matrix.
    The class can be used to fit distribution parameters to data.

    Attributes
    -----------
      mean_x : torch.Tensor, shape (n_dim)
          Mean of X. It is constrained to the unit sphere.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
          Covariance of X. It is constrained to be symmetric positive definite.

      B : torch.Tensor, shape (n_dim, n_dim)
          The ellipse matrix. It is constrained to be symmetric positive definite.

    Methods
    ----------
      moments():
          Compute the moments using a Taylor approximation.

      log_pdf() :
          Compute the value of the log pdf at given points. (Not implemented)

      pdf() :
          Compute the value of the pdf at given points. (Not implemented)

      moments_empirical() :
          Compute the moments by sampling from the distribution

      sample() :
          Sample points from the distribution.

      moment_match() :
          Fit the distribution parameters to the observed moments.

      max_likelihood() :
          Fit the distribution parameters to the observed data
          using maximum likelihood. (Not implemented)

      moment_init() :
          Initialize the distribution parameters using the observed moments
          as the initial guess.

      add_covariance_parametrization() :
          Substitute the current covariance_x constraint with a new parametrization.
    """

    def __init__(
        self,
        n_dim=None,
        mean_x=None,
        covariance_x=None,
        B=None,
    ):
        """Initialize an instance of the ProjNormalEllipseParent class.

        Parameters
        ------------
          n_dim : int, optional
              Dimension of the underlying Gaussian distribution. If mean
              and covariance are provided, this is not required.

          mean_x : torch.Tensor, shape (n_dim), optional
              Mean of X. It is converted to unit norm. Default is random.

          covariance_x : torch.Tensor, shape (n_dim, n_dim), optional
              Initial covariance. Default is the identity.

          B : torch.Tensor, shape (n_dim, n_dim), optional
              SPD matrix defining the ellipse. If not provided, it is initialized
              as an identity matrix.
        """
        super().__init__(n_dim=n_dim, mean_x=mean_x, covariance_x=covariance_x)
        if B is None:
            B = torch.eye(self.n_dim)
        self.B = nn.Parameter(B.clone())
        geotorch.positive_definite(self, "B")


    def moments(self):
        """
        Compute the Taylor approximation to the moments
        of the variable Y = X/sqrt(X'BX), where X~N(mean_x, covariance_x)
        and B is a symmetric positive definite matrix.

        Returns
        ---------
        dict
            Dictionary containing the mean, covariance and second moment
            of the projected normal.
        """
        # Change basis to make B the identity
        B_chol = torch.linalg.cholesky(self.B)

        # Use dist.ellipse_const to not redefine method for the EllipseConst class
        gamma = dist.ellipse_const.moments.mean(
            mean_x=mean_z,
            covariance_x=covariance_z,
            const=self.const,
            B_chol=B_chol,
        )
        second_moment = dist.ellipse_const.moments.second_moment(
            mean_x=mean_z,
            covariance_x=covariance_z,
            const=self.const,
            B_chol=B_chol,
        )
        cov = second_moment - torch.einsum("i,j->ij", gamma, gamma)

        return {"mean": gamma, "covariance": cov, "second_moment": second_moment}


    def moments_empirical(self, n_samples=500000):
        """
        Compute the moments of the variable Y = X/sqrt(X'BX),
        where X~N(mean_x, covariance_x), by sampling from the distribution.

        Returns
        ----------------
        dict
            Dictionary containing the mean, covariance and second moment
            of the projected normal.
        """
        with torch.no_grad():
            stats_dict = dist.ellipse_const.sampling.empirical_moments(
                mean_x=self.mean_x,
                covariance_x=self.covariance_x,
                n_samples=n_samples,
                const=self.const,
                B=self.B,
            )
        return stats_dict


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
        raise NotImplementedError(
            "A formula for the pdf of the projected normal on an ellipse is not available."
        )


    def pdf(self, y):
        """
        Compute the pdf at points y under the projected normal distribution.

        Parameters
        ----------------
          y : torch.Tensor, shape (n_points, n_dim)
              Points to evaluate the pdf.

        Returns
        ----------------
          torch.Tensor, shape (n_points)
              PDF of the point.
        """
        raise NotImplementedError(
            "A formula for the pdf of the projected normal on an ellipse is not available."
        )


    def sample(self, n_samples):
        """Sample from the distribution.

        Parameters
        ----------------
          n_samples : int
              Number of samples to draw.

        Returns
        ----------------
          torch.Tensor, shape (n_samples, n_dim)
              Samples from the distribution.
        """
        with torch.no_grad():
            samples = dist.ellipse_const.sampling.sample(
                mean_x=self.mean_x,
                covariance_x=self.covariance_x,
                n_samples=n_samples,
                const=self.const,
                B=B,
            )
        return samples

    def __dir__(self):
        return super().__dir__() + ["B"]
