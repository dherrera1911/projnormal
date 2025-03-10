"""Approximation to the moments of the general projected normal distribution projected onto ellipse given by matrix B."""
import torch
from projnormal.ellipse_linalg import spd_sqrt

from ..const import moments as _pnc_moments

__all__ = ["mean", "second_moment"]


def __dir__():
    return __all__


def mean(mean_x, covariance_x, const, B=None, B_sqrt=None, B_sqrt_inv=None):
    """
    Compute the Taylor approximation to the expected value of the variable
    Y = X/(X'BX + C)^0.5, where X~N(mean_x, covariance_x) and B is a symmetric positive
    defininte matrix. Y is an elliptical version of the general projected normal
    distribution.

    Either the matrix B or its square root and its inverse can be provided.
    If B is provided, the square root and inverse are computed internally.

    Parameters:
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

      const : torch.Tensor, shape ()
        Constant added to the denominator.

      B : torch.Tensor, shape (n_dim, n_dim), optional
        Symmetric positive definite matrix defining the ellipse.

      B_sqrt : torch.Tensor, shape (n_dim, n_dim), optional
        Square root of B.

      B_sqrt_inv : torch.Tensor, shape (n_dim, n_dim), optional
        Inverse of the square root of B.

    Returns:
    ----------------
      torch.Tensor, shape (n_dim,)
          Expected value for the projected normal on ellipse.
    """
    if B_sqrt is None or B_sqrt_inv is None:
        if B is None:
            raise ValueError("Either B or B_sqrt and B_sqrt_inv must be provided.")
        B_sqrt, B_sqrt_inv = spd_sqrt(B)

    # Change basis to make B the identity
    mean_z = B_sqrt @ mean_x
    covariance_z = B_sqrt @ covariance_x @ B_sqrt

    # Compute the mean in the new basis
    gamma_z = _pnc_moments.mean(
      mean_x=mean_z,
      covariance_x=covariance_z,
      const=const
    )

    # Change back to the original basis
    gamma = B_sqrt_inv @ gamma_z
    return gamma


def second_moment(mean_x, covariance_x, const=None, B=None, B_sqrt=None, B_sqrt_inv=None):
    """
    Compute the Taylor approximation to the second moment matrix of the
    variable Y = X/(X'X)^0.5, where X~N(mean_x, covariance_x). Y has a
    general projected normal distribution.

    The approximation is based on the Taylor expansion of the
    function f(n,d) = n/d, where n = X_i*X_j and d = X'X.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

      const : torch.Tensor, shape ()
        Constant added to the denominator.

      B : torch.Tensor, shape (n_dim, n_dim), optional
        Symmetric positive definite matrix defining the ellipse.

      B_sqrt : torch.Tensor, shape (n_dim, n_dim), optional
        Square root of B.

      B_sqrt_inv : torch.Tensor, shape (n_dim, n_dim), optional
        Inverse of the square root of B.

    Returns
    ----------------
      torch.Tensor, shape (n_dim, n_dim)
          Second moment matrix of Y
    """
    if B_sqrt is None or B_sqrt_inv is None:
        if B is None:
            raise ValueError("Either B or B_sqrt and B_sqrt_inv must be provided.")
        B_sqrt, B_sqrt_inv = spd_sqrt(B)

    mean_z = B_sqrt @ mean_x
    covariance_z = B_sqrt @ covariance_x @ B_sqrt

    # Compute the second moment in the new basis
    sm_z = _pnc_moments.second_moment(
      mean_x=mean_z,
      covariance_x=covariance_z,
      const=const
    )

    # Change back to the original basis
    sm = B_sqrt_inv @ sm_z @ B_sqrt_inv

    return sm
