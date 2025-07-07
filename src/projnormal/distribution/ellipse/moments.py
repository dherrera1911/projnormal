"""Approximation to the moments of the general projected normal distribution projected onto ellipse given by matrix B."""

from ..ellipse_const import moments as _pnec_moments

__all__ = ["mean", "second_moment"]


def __dir__():
    return __all__


def mean(mean_x, covariance_x, B=None, B_chol=None):
    """
    Compute the Taylor approximation to the expected value of the variable
    Y = X/(X'BX)^0.5, where X~N(mean_x, covariance_x) and B is a symmetric positive
    defininte matrix. Y is an elliptical version of the general projected normal
    distribution.

    Either the matrix B or its square root and its inverse can be provided.
    If B is provided, the square root and inverse are computed internally.

    Parameters
    ----------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

      B : torch.Tensor, shape (n_dim, n_dim), optional
        Symmetric positive definite matrix defining the ellipse.

      B_chol : torch.Tensor, shape (n_dim, n_dim), optional
        Cholesky decomposition of B. Can be provided to avoid recomputing it.

    Returns
    -------
      torch.Tensor, shape (n_dim,)
          Expected value for the projected normal on ellipse.
    """
    return _pnec_moments.mean(mean_x=mean_x, covariance_x=covariance_x,
                              const=0, B=B, B_chol=B_chol)


def second_moment(mean_x, covariance_x, B=None, B_chol=None):
    """
    Compute the Taylor approximation to the second moment matrix of the
    variable Y = X/(X'BX)^0.5, where X~N(mean_x, covariance_x). Y has a
    general projected normal distribution.

    Parameters
    ----------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

      B : torch.Tensor, shape (n_dim, n_dim), optional
        Symmetric positive definite matrix defining the ellipse.

      B_chol : torch.Tensor, shape (n_dim, n_dim), optional
        Cholesky decomposition of B. Can be provided to avoid recomputing it.

    Returns
    -------
      torch.Tensor, shape (n_dim, n_dim)
          Second moment matrix of Y.
    """
    return _pnec_moments.second_moment(mean_x=mean_x, covariance_x=covariance_x,
                                      const=0, B=B, B_chol=B_chol)
