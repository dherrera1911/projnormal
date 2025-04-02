"""Class for the general projected normal distribution."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import projnormal.distribution as prnorm

from ._constraints import Positive
from .general_projnormal import ProjNormal
from ._ellipsoid import Ellipsoid, EllipsoidFixed, EllipsoidFull


__all__ = [
  "ProjNormalEllipse",
  "ProjNormalEllipseIso",
  "ProjNormalEllipseFixed",
  "ProjNormalEllipseIsoFull",
]


def __dir__():
    return __all__


class ProjNormalEllipseParent(ABC, ProjNormal):
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
        """
        super().__init__(n_dim=n_dim, mean_x=mean_x, covariance_x=covariance_x)


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

        gamma = prnorm.ellipse_const.moments.mean(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            const=self.const,
            B_sqrt=B_sqrt,
            B_sqrt_inv=B_sqrt_inv,
        )

        second_moment = prnorm.ellipse_const.moments.second_moment(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            const=self.const,
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
            stats_dict = prnorm.ellipse_const.sampling.empirical_moments(
                mean_x=self.mean_x,
                covariance_x=self.covariance_x,
                const=self.const,
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
            samples = prnorm.ellipse_const.sampling.sample(
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
                "moments_empirical", "sample", "moment_match", "moment_init",
                "ellipse", "B"]


class ProjNormalEllipse(ProjNormalEllipseParent):
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
        n_dirs=None,
        B_eigvals=None,
        B_eigvecs=None,
        B_rad_sq=1.0,
    ):
        """Initialize an instance of the ProjNormalEllipse class.

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
        super().__init__(n_dim=n_dim, mean_x=mean_x, covariance_x=covariance_x)

        # Initialize the ellipse class
        self._init_ellipse(n_dirs, B_eigvals, B_eigvecs, B_rad_sq)


    def _init_ellipse(self, n_dirs, B_eigvals, B_eigvecs, B_rad_sq):
        """
        Initialize the ellipse class with the provided parameters.
        """
        self.ellipse = Ellipsoid(
            n_dim=self.n_dim,
            n_dirs=n_dirs,
            B_eigvals=B_eigvals,
            B_eigvecs=B_eigvecs,
            B_rad_sq=torch.as_tensor(B_rad_sq),
        )


class ProjNormalEllipseIso(ProjNormalEllipse):
    """
    This class implements the general projected normal distribution but with
    projection on an ellipse instead of the sphere, and with an isotropic
    covariance_x matrix.

    The variable Y following the distribution
    is defined as Y = X / sqrt(X'BX), where X~N(mean_x, I * sigma**2)
    and B is a symmetric positive definite matrix.
    The class can be used to fit distribution parameters to data.

    Attributes
    -----------
      mean_x : torch.Tensor, shape (n_dim)
          Mean of X. It is constrained to the unit sphere.

      sigma : torch.Tensor, shape (n_dim, n_dim)
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
        sigma2=None,
        n_dirs=2,
        B_eigvecs=None,
        B_eigvals=None,
        B_rad_sq=1.0,
    ):
        """Initialize an instance of the ProjNormalEllipseIso class.

        Parameters
        ------------
          n_dim : int, optional
              Dimension of the underlying Gaussian distribution. If mean
              and covariance are provided, this is not required.

          mean_x : torch.Tensor, shape (n_dim), optional
              Mean of X. It is converted to unit norm. Default is random.

          sigma2 : torch.Tensor, shape (n_dim, n_dim), optional
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
        super().__init__(
          n_dim=n_dim,
          mean_x=mean_x,
          covariance_x=None,
          n_dirs=n_dirs,
          B_eigvecs=B_eigvecs,
          B_eigvals=B_eigvals,
          B_rad_sq=B_rad_sq,
        )

        # Remove the inherited nn.Parameter
        parametrize.remove_parametrizations(self, "covariance_x")
        delattr(self, "covariance_x")

        if sigma2 is None:
            sigma2 = torch.tensor(1.0)
        self.sigma = nn.Parameter(torch.sqrt(torch.as_tensor(sigma2)))
        parametrize.register_parametrization(self, "sigma", Positive())


    def moment_init(self, data_moments):
        """
        Initialize the distribution parameters using the observed moments
        as the initial guess (making sure the mean is normalized).

        Parameters
        ----------------
          data : dict
            Dictionary containing the observed moments. Must contain the keys
              - 'mean': torch.Tensor, shape (n_dim)
              - 'covariance': torch.Tensor, shape (n_dim, n_dim)
              - 'second_moment': torch.Tensor, shape (n_dim, n_dim)
        """
        data_mean_normalized = data_moments["mean"] / torch.norm(data_moments["mean"])
        self.mean_x = data_mean_normalized
        self.sigma = torch.sqrt(data_moments["covariance"].trace() / self.n_dim)


    @property
    def covariance_x(self):
        covariance_x = torch.eye(
          self.n_dim, dtype=self.mean_x.dtype, device=self.mean_x.device
        ) * self.sigma**2
        return covariance_x


class ProjNormalEllipseFixed(ProjNormalEllipseParent):
    """
    This class implements the general projected normal distribution but with
    projection on an ellipse instead of the sphere.
    The variable Y following the distribution
    is defined as Y = X / sqrt(X'BX), where X~N(mean_x, covariance_x)
    and B is a symmetric positive definite matrix.
    The class can be used to fit distribution parameters to data.

    In this class, B is not optimized and is fixed to a given value.

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
        B=None,
    ):
        """Initialize an instance of the ProjNormalEllipse class.

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
              The ellipse matrix. Default is the identity.
        """
        super().__init__(n_dim=n_dim, mean_x=mean_x, covariance_x=covariance_x)

        # Initialize the ellipse class
        self._init_ellipse(B)


    def _init_ellipse(self, B):
        """
        Initialize the ellipse class with the provided parameters.
        """
        if B is None:
            B = torch.eye(self.n_dim)

        self.ellipse = EllipsoidFixed(
            B=B
        )



class ProjNormalEllipseFull(ProjNormalEllipseParent):
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
        B=None,
    ):
        """Initialize an instance of the ProjNormalEllipse class.

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
              The ellipse matrix. Default is the identity.
        """
        super().__init__(n_dim=n_dim, mean_x=mean_x, covariance_x=covariance_x)

        # Initialize the ellipse class
        if B is None:
            B = torch.eye(self.n_dim)
        self._init_ellipse(B)


    def _init_ellipse(self, B):
        """
        Initialize the ellipse class with the provided parameters.
        """
        self.ellipse = EllipsoidFull(
            n_dim=self.n_dim,
            B=B,
        )


class ProjNormalEllipseIsoFull(ProjNormalEllipseFull):
    """
    This class implements the general projected normal distribution but with
    projection on an ellipse instead of the sphere, and with an isotropic
    covariance_x matrix.

    The variable Y following the distribution
    is defined as Y = X / sqrt(X'BX), where X~N(mean_x, I * sigma**2)
    and B is a symmetric positive definite matrix.
    The class can be used to fit distribution parameters to data.

    Attributes
    -----------
      mean_x : torch.Tensor, shape (n_dim)
          Mean of X. It is constrained to the unit sphere.

      sigma : torch.Tensor, shape (n_dim, n_dim)
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
        sigma2=None,
        B=None,
    ):
        """Initialize an instance of the ProjNormalEllipseIso class.

        Parameters
        ------------
          n_dim : int, optional
              Dimension of the underlying Gaussian distribution. If mean
              and covariance are provided, this is not required.

          mean_x : torch.Tensor, shape (n_dim), optional
              Mean of X. It is converted to unit norm. Default is random.

          sigma2 : torch.Tensor, shape (n_dim, n_dim), optional
              Initial covariance. Default is the identity.

          B : torch.Tensor, shape (n_dim, n_dim), optional
              The ellipse matrix. Default is the identity.
        """
        super().__init__(
          n_dim=n_dim,
          mean_x=mean_x,
          covariance_x=None,
          B=B,
        )

        # Remove the inherited nn.Parameter
        parametrize.remove_parametrizations(self, "covariance_x")
        delattr(self, "covariance_x")

        if sigma2 is None:
            sigma2 = torch.tensor(1.0)
        self.sigma = nn.Parameter(torch.sqrt(torch.as_tensor(sigma2)))
        parametrize.register_parametrization(self, "sigma", Positive())


    def moment_init(self, data_moments):
        """
        Initialize the distribution parameters using the observed moments
        as the initial guess (making sure the mean is normalized).

        Parameters
        ----------------
          data : dict
            Dictionary containing the observed moments. Must contain the keys
              - 'mean': torch.Tensor, shape (n_dim)
              - 'covariance': torch.Tensor, shape (n_dim, n_dim)
              - 'second_moment': torch.Tensor, shape (n_dim, n_dim)
        """
        data_mean_normalized = data_moments["mean"] / torch.norm(data_moments["mean"])
        self.mean_x = data_mean_normalized
        self.sigma = torch.sqrt(data_moments["covariance"].trace() / self.n_dim)


    @property
    def covariance_x(self):
        covariance_x = torch.eye(
          self.n_dim, dtype=self.mean_x.dtype, device=self.mean_x.device
        ) * self.sigma**2
        return covariance_x

