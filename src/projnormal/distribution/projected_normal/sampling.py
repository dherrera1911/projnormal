"""Sampling functions for the general projected normal distribution."""
from ..const import sampling as _const_sampling

__all__ = ["sample", "empirical_moments"]


def __dir__():
    return __all__


def sample(mean_x, covariance_x, n_samples):
    """
    Sample from the variable Y = X/(X'X)^0.5, where X~N(mean_x, covariance_x).
    The variable Y has a general projected normal distribution.

    Parameters
    ----------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
          Covariance matrix of X elements.

      n_samples : int
          Number of samples.

    Returns
    -------
      torch.Tensor, shape (n_samples, n_dim)
          Samples from the projected normal.
    """
    return _const_sampling.sample(
      mean_x=mean_x,
      covariance_x=covariance_x,
      n_samples=n_samples,
      const=0
    )


def empirical_moments(mean_x, covariance_x, n_samples):
    """
    Compute the mean, covariance and second moment of the variable
    Y = X/(X'X)^0.5, where X~N(mean_x, covariance_x), by sampling from the
    distribution. The variable Y has a general projected normal distribution.

    Parameters
    ----------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
          Covariance matrix of X elements.

      n_samples : int
          Number of samples.

    Returns
    -------
      dict
          Dictionary with the following keys and values
            'mean' : torch.Tensor, shape (n_dim,)
                Mean of the projected normal.
            'covariance' : torch.Tensor, shape (n_dim, n_dim)
                Covariance of the projected normal.
            'second_moment' : torch.Tensor, shape (n_dim, n_dim)
                Second moment of the projected normal.
    """
    return _const_sampling.empirical_moments(
      mean_x=mean_x,
      covariance_x=covariance_x,
      n_samples=n_samples,
      const=0
    )
