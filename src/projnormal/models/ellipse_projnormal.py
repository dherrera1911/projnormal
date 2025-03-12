"""Class for the general projected normal distribution."""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import projnormal.distribution as prnorm

from ._constraints import Sphere
from .general_projnormal import ProjNormal
from ._ellipsoid import Ellipsoid


__all__ = [
  "ProjNormalEllipse",
]


def __dir__():
    return __all__


class ProjNormalEllipse(ProjNormal):
    """
    This class implements the general projected normal distribution but with
    projection on an ellipse instead of the sphere.
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
        n_dirs=2,
        B_eigvecs=None,
        B_eigvals=None,
        B_rad_sq=1.0,
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

          n_dirs : int, optional
              Number of directions to use in the optimization. Default is 2.
              If `B_eigvals` is provided, it is ignored.

          B_eigvecs : torch.Tensor, shape (n_dirs, n_dim), optional
              Initial eigenvectors of the ellipse matrix. Default is random
              orthogonal vectors.

          B_eigvals : torch.Tensor, shape (n_dirs), optional
              Initial eigenvalues of the associated eigenvectors vectors in
              B_eigvecs.

          B_rad_sq : torch.Tensor, shape (), optional
              Initial eigenvalue of all directions not in B_eigvecs. Default
              is 1.
        """
        super().__init__(n_dim=n_dim, mean_x=mean_x, covariance_x=covariance_x)

        # Parse content
        if B_eigvals is not None:
            n_dirs = B_eigvals.shape[0]
            if B_eigvecs is None and B_eigvecs.shape[0] != B_eigvals.shape[0]:
                raise ValueError("The number of eigenvectors must match the number of eigenvalues.")

        # Initialize the ellipse class
        self.ellipse = Ellipsoid(
            n_dim=self.n_dim,
            n_dirs=n_dirs,
            rad_sq=torch.as_tensor(B_rad_sq),
        )

        if B_eigvecs is not None:
            if B_eigvecs.shape[-1] != self.n_dim:
                raise ValueError("The eigenvectors must have the same dimension as the data.")
            self.ellipse.eigvecs = B_eigvecs.clone()
        if B_eigvals is not None:
            self.ellipse.eigvals = B_eigvals.clone()


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
        # Extract B matrices needed
        B_sqrt = self.ellipse.get_B_sqrt()
        B_sqrt_inv = self.ellipse.get_B_sqrt_inv()

        gamma = prnorm.ellipse.moments.mean(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            B_sqrt=B_sqrt,
            B_sqrt_inv=B_sqrt_inv,
        )

        second_moment = prnorm.ellipse.moments.second_moment(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            B_sqrt=B_sqrt,
            B_sqrt_inv=B_sqrt_inv,
        )

        cov = second_moment - torch.einsum("i,j->ij", gamma, gamma)

        return {"mean": gamma, "covariance": cov, "second_moment": second_moment}


    def moments_empirical(self, n_samples=200000):
        """
        Compute the moments of the variable Y = X/||X||, where X~N(mean_x, covariance_x),
        by sampling from the distribution.

        Returns
        ----------------
        dict
            Dictionary containing the mean, covariance and second moment
            of the projected normal.
        """
        with torch.no_grad():
            B_sqrt = self.ellipse.get_B_sqrt()
            B_sqrt_inv = self.ellipse.get_B_sqrt_inv()
            stats_dict = prnorm.ellipse.sampling.empirical_moments(
                mean_x=self.mean_x,
                covariance_x=self.covariance_x,
                n_samples=n_samples,
                B_sqrt=B_sqrt,
                B_sqrt_inv=B_sqrt_inv,
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
        # Extract B matrices needed
        B = self.ellipse.get_B()
        B_sqrt = self.ellipse.get_B_sqrt()
        B_sqrt_ldet = self.ellipse.get_B_sqrt_inv()

        lpdf = prnorm.ellipse.pdf.log_pdf(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            y=y,
            B=B,
            B_sqrt=B_sqrt,
            B_sqrt_ldet=B_sqrt_ldet
        )
        return lpdf


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
        pdf = torch.exp(self.log_pdf(y))
        return pdf


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
            B_sqrt = self.ellipse.get_B_sqrt()
            B_sqrt_inv = self.ellipse.get_B_sqrt_inv()
            samples = prnorm.const.sampling.sample(
                mean_x=self.mean_x,
                covariance_x=self.covariance_x,
                n_samples=n_samples,
                const=self.const,
                B_sqrt=B_sqrt,
                B_sqrt_inv=B_sqrt_inv,
            )
        return samples


    @property
    def B(self):
        return self.ellipse.get_B()

    @B.setter
    def B(self):
        raise AttributeError(
            "The ellipse matrix B can't be set directly."
            "Set ellipse.eigvecs, ellipse.eigvals and ellipse.rad_sq instead."
        )

    def __dir__(self):
        return ["mean_x", "covariance_x", "moments", "log_pdf", "pdf",
                "moments_empirical", "sample", "moment_match", "moment_init"]

