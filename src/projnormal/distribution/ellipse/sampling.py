"""Sampling functions for the general projected normal distribution."""
import torch
from projnormal.ellipse_linalg import spd_sqrt

from ..general import sampling as _png_sampling

__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, n_samples, B=None, B_sqrt=None, B_sqrt_inv=None):
    """
    Sample from the variable Y = X/(X'BX)^0.5, where X~N(mean_x, covariance_x)
    and B is a symmetric positive definite matrix.
    The variable Y has a general projected normal distribution with
    projection to the ellipsoid defined by B.

    Parameters:
    -----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
          Covariance matrix of X elements.

      n_samples : int
          Number of samples.

      B : torch.Tensor, shape (n_dim, n_dim), optional
        Symmetric positive definite matrix defining the ellipse.

      B_sqrt : torch.Tensor, shape (n_dim, n_dim), optional
        Square root of B.

      B_sqrt_inv : torch.Tensor, shape (n_dim, n_dim), optional
        Inverse of the square root of B.

    Returns:
    -----------------
      torch.Tensor, shape (n_samples, n_dim)
          Samples from the projected normal.
    """
    if B_sqrt is None or B_sqrt_inv is None:
        if B is None:
            raise ValueError("Either B or B_sqrt and B_sqrt_inv must be provided.")
        B_sqrt, B_sqrt_inv = spd_sqrt(B)

    # Change basis to make B the identity
    mean_z = B_sqrt @ mean_x
    covariance_z = B_sqrt @ covariance_x @ B_sqrt

    # Sample from the standard projected normal
    samples_prnorm_z = _png_sampling.sample(
      mean_x=mean_z, covariance_x=covariance_z, n_samples=n_samples
    )

    # Change basis back to the original space
    samples_prnorm = samples_prnorm_z @ B_sqrt_inv
    return samples_prnorm


def empirical_moments(mean_x, covariance_x, n_samples, B=None, B_sqrt=None, B_sqrt_inv=None):
    """
    Compute the mean, covariance and second moment of the variable
    Y = X/(X'X)^0.5, where X~N(mean_x, covariance_x), by sampling from the
    distribution. The variable Y has a general projected normal distribution.

    Parameters:
    -----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
          Covariance matrix of X elements.

      n_samples : int
          Number of samples.

      B : torch.Tensor, shape (n_dim, n_dim), optional
        Symmetric positive definite matrix defining the ellipse.

      B_sqrt : torch.Tensor, shape (n_dim, n_dim), optional
        Square root of B.

      B_sqrt_inv : torch.Tensor, shape (n_dim, n_dim), optional
        Inverse of the square root of B.

    Returns:
    -----------------
      dict
          Dictionary with the following keys and values
            'mean' : torch.Tensor, shape (n_dim,)
                Mean of the projected normal.
            'covariance' : torch.Tensor, shape (n_dim, n_dim)
                Covariance of the projected normal.
            'second_moment' : torch.Tensor, shape (n_dim, n_dim)
                Second moment of the projected normal.
    """
    if B_sqrt is None or B_sqrt_inv is None:
        if B is None:
            raise ValueError("Either B or B_sqrt and B_sqrt_inv must be provided.")
        B_sqrt, B_sqrt_inv = spd_sqrt(B)

    # Change basis to make B the identity
    mean_z = B_sqrt @ mean_x
    covariance_z = B_sqrt @ covariance_x @ B_sqrt

    moment_dict_z = _png_sampling.empirical_moments(
      mean_x=mean_z, covariance_x=covariance_z, n_samples=n_samples
    )

    # Change basis back to the original space
    moment_dict = {}
    moment_dict["mean"] = B_sqrt_inv @ moment_dict_z["mean"]
    moment_dict["covariance"] = B_sqrt_inv @ moment_dict_z["covariance"] @ B_sqrt_inv
    moment_dict["second_moment"] = (
      B_sqrt_inv @ moment_dict_z["second_moment"] @ B_sqrt_inv
    )
    return moment_dict
