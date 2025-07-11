"""Class for the general projected normal distribution."""
import geotorch
import torch
import torch.nn as nn

import projnormal.formulas.projected_normal_Bc as pnbc_formulas

from .projected_normal import ProjNormal

__all__ = [
  "ProjNormalEllipse",
]


def __dir__():
    return __all__


class ProjNormalEllipse(ProjNormal):
    r"""
    Projected normal distribution variant, describing the variable
    :math:`y=x/\sqrt{x^T B x}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    follows a multivariate normal distribution and :math:`B` is a symmetric positive
    definite matrix.

    Parameters
    ----------
      n_dim : ``int``
          Dimension of :math:`x` (the embedding space). Optional: If ``mean_x``
          and ``covariance_x`` are provided, it is not required.

      mean_x : ``torch.Tensor``, optional
          Mean of :math:`x`. Shape ``(n_dim)``. Default is random.

      covariance_x : ``torch.Tensor``, optional
          Covariance of :math:`x`. Shape ``(n_dim, n_dim)``. Default is the identity.

      B : ``torch.Tensor``, optional
          SPD matrix defining the ellipse. Shape ``(n_dim, n_dim)``. Default is the identity matrix.


    Attributes
    ----------
      mean_x : ``torch.Tensor``
          Mean of :math:`x`. Learnable parameter constrained to have unit norm. Shape ``(n_dim)``.

      covariance_x : ``torch.Tensor``
          Covariance of :math:`x`. Learnable parameter constrained to be SPD. Shape ``(n_dim, n_dim)``.

      B : ``torch.Tensor``
          Quadratic form matrix of the denominator. Shape ``(n_dim, n_dim)``. Learnable parameter constrained to be SPD.
    """

    def __init__(
        self,
        n_dim=None,
        mean_x=None,
        covariance_x=None,
        B=None,
    ):
        super().__init__(n_dim=n_dim, mean_x=mean_x, covariance_x=covariance_x)
        if B is None:
            B = torch.eye(self.n_dim)
        self.B = nn.Parameter(B.clone())
        geotorch.positive_definite(self, "B")
        self.B = B.clone()


    def moments(self):
        """
        Compute moments of the distribution via Taylor approximation.

        Returns
        -------
          dict
              Dictionary with keys ``mean``, ``covariance`` and ``second_moment``,
              containing the corresponding moments of the distribution.
        """
        # Change basis to make B the identity
        B_chol = torch.linalg.cholesky(self.B)

        # Use dist.ellipse_const to not redefine method for the EllipseConst class
        gamma = pnbc_formulas.mean(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            const=self.const,
            B_chol=B_chol,
        )
        second_moment = pnbc_formulas.second_moment(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            const=self.const,
            B_chol=B_chol,
        )
        cov = second_moment - torch.einsum("i,j->ij", gamma, gamma)

        return {"mean": gamma, "covariance": cov, "second_moment": second_moment}


    def moments_empirical(self, n_samples=500000):
        """
        Compute moments of the distribution via sampling.

        Parameters
        ----------
          n_samples : ``int``
              Number of samples to draw for empirical moments. Default is ``200000``.

        Returns
        -------
          dict
              Dictionary with keys ``mean``, ``covariance`` and ``second_moment``,
              containing the corresponding moments of the distribution.
        """
        with torch.no_grad():
            stats_dict = pnbc_formulas.empirical_moments(
                mean_x=self.mean_x,
                covariance_x=self.covariance_x,
                n_samples=n_samples,
                const=self.const,
                B=self.B,
            )
        return stats_dict


    def log_pdf(self, y):
        """
        Compute the log pdf of points `y`.

        Parameters
        ----------
          y : ``torch.Tensor``
              Points to evaluate the log pdf. Shape ``(n_points, n_dim)``.

        Returns
        -------
          ``torch.Tensor``
              Log-PDF of `y`. Shape ``(n_points)``.
        """
        raise NotImplementedError(
            "A formula for the pdf of the projected normal on an ellipse is not available."
        )


    def pdf(self, y):
        """
        Compute the pdf of points `y`.

        Parameters
        ----------
          y : ``torch.Tensor``
              Points to evaluate the pdf. Shape ``(n_points, n_dim)``.

        Returns
        -------
          ``torch.Tensor``
              PDF of the points `y`. Shape ``(n_points)``.
        """
        raise NotImplementedError(
            "A formula for the pdf of the projected normal on an ellipse is not available."
        )


    def sample(self, n_samples):
        """
        Sample from the distribution.

        Parameters
        ----------
          n_samples : ``int``
              Number of samples to draw.

        Returns
        -------
          ``torch.Tensor``
              Samples from the distribution. Shape ``(n_samples, n_dim)``.
        """
        with torch.no_grad():
            samples = pnbc_formulas.sample(
                mean_x=self.mean_x,
                covariance_x=self.covariance_x,
                n_samples=n_samples,
                const=self.const,
                B=self.B,
            )
        return samples

    def __dir__(self):
        """List of methods available in the ProjNormal class."""
        return super().__dir__() + ["B"]
