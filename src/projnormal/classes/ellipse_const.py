"""Cass for the general projected normal distribution."""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

import projnormal.formulas.projected_normal_Bc as pnbc_formulas

from .constraints import Positive
from .ellipse import ProjNormalEllipse

__all__ = [
  "ProjNormalEllipseConst",
]


def __dir__():
    return __all__


class ProjNormalEllipseConst(ProjNormalEllipse):
    r"""
    Projected normal distribution variant, describing the variable
    :math:`y=x/\sqrt{x^T B x + c}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    follows a multivariate normal distribution, :math:`B` is a symmetric positive
    definite matrix, and :math:`c` is a positive scalar constant.

    Parameters
    ----------
      n_dim : ``int``
          Dimension of :math:`x` (the embedding space). Optional: If ``mean_x``
          and ``covariance_x`` are provided, it is not required.

      mean_x : ``torch.Tensor``, optional
          Mean of :math:`x`. Shape ``(n_dim)``. Default is random.

      covariance_x : ``torch.Tensor``, optional
          Covariance of :math:`x`. Shape ``(n_dim, n_dim)``. Default is the identity.

      const : ``torch.Tensor``, optional
          The denominator additive constant. Shape ``(1,)``. Default is 1.

      B : ``torch.Tensor``, optional
          SPD matrix defining the ellipse. Shape ``(n_dim, n_dim)``. Default is the identity matrix.


    Attributes
    ----------
      mean_x : ``torch.Tensor``
          Mean of :math:`x`. Learnable parameter constrained to have unit norm. Shape ``(n_dim)``.

      covariance_x : ``torch.Tensor``
          Covariance of :math:`x`. Learnable parameter constrained to be SPD. Shape ``(n_dim, n_dim)``.

      const : ``torch.Tensor``
          Denominator additive constant. Shape ``(1,)``. Learnable parameter constained to be positive.

      B : ``torch.Tensor``
          Quadratic form matrix of the denominator. Shape ``(n_dim, n_dim)``. Learnable parameter constrained to be SPD.
    """

    def __init__(
        self,
        n_dim=None,
        mean_x=None,
        covariance_x=None,
        const=None,
        B=None,
    ):
        super().__init__(
          n_dim=n_dim,
          mean_x=mean_x,
          covariance_x=covariance_x,
          B=B,
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
        lpdf = pnbc_formulas.log_pdf(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            y=y,
            const=self.const,
            B=self.B,
        )
        return lpdf


    def __dir__(self):
        """List of methods available in the ProjNormal class."""
        return super().__dir__() + ["const"]
