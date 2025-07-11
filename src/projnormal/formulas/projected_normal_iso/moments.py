"""Exact moments of the projected normal distribution with isotropic covariance."""

import scipy.special as sps
import torch

__all__ = ["mean", "second_moment", "batch_second_moment"]


def __dir__():
    return __all__


def mean(mean_x, var_x):
    r"""
    Compute the mean of :math:`y = x/\sqrt{x^T x}`,
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)` and
    :math:`\Sigma_x = \sigma^2 I`. This is done using the
    exact analytic formulas.


    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      var_x : ``torch.tensor``
          variance of `x`. shape is ``()``.

    Returns
    -------
      ``torch.Tensor``
          Expected value for the projected normal. Shape is ``(n_dim,)``.
    """
    sigma = torch.sqrt(var_x)
    n_dim = torch.as_tensor(mean_x.shape[-1])
    non_centrality = torch.norm(mean_x / sigma, dim=-1) ** 2

    # Compute terms separately
    gln1 = torch.special.gammaln((n_dim + 1) / 2)
    gln2 = torch.special.gammaln(n_dim / 2 + 1)
    g_ratio = 1 / (torch.sqrt(torch.as_tensor(2.0)) * sigma) * torch.exp(gln1 - gln2)
    hyp_val = sps.hyp1f1(1 / 2, n_dim / 2 + 1, -non_centrality / 2)

    # Multiply terms to get the expected value
    gamma = torch.einsum("...d,...->...d", mean_x, g_ratio * hyp_val)

    return gamma


# Apply the isotropic covariance formula to get the covariance
# for each stimulus
def second_moment(mean_x, var_x):
    r"""
    Compute the second moment matrix of :math:`y = x/\sqrt{x^T x}`,
    where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`\Sigma_x = \sigma^2 I`. This is done using
    the exact analytic formulas.

    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      var_x : ``torch.tensor``
          variance of `x`. shape is ``()``.

    Returns
    -------
      ``torch.Tensor``
          Second moment matrix of :math:`y`. Shape is ``(n_dim, n_dim)``.
    """
    sigma = torch.sqrt(var_x)
    n_dim = torch.as_tensor(mean_x.shape[-1])
    # Compute weights for mean and identity terms
    noise_w, mean_w = _iso_sm_weights(mean_x=mean_x, sigma=sigma)

    # Compute the second moment of each stimulus
    mean_x_normalized = mean_x / sigma
    # Get the outer product of the normalized stimulus, and multiply by weight
    second_moment = torch.einsum(
        "...d,...b,...->...db", mean_x_normalized, mean_x_normalized, mean_w
    )

    # Add noise term to the diagonal
    diag_idx = torch.arange(n_dim)
    is_batch = mean_x.dim() == 2
    if is_batch:
        n_batch = mean_x.shape[0]
        for i in range(n_batch):
            second_moment[i, diag_idx, diag_idx] += noise_w[i]
    else:
        second_moment[diag_idx, diag_idx] += noise_w

    return second_moment


def batch_second_moment(mean_x, var_x):
    """
    Compute the average second moment of a set of projected
    normals Y_i = X_i/||X_i||, where X_i~N(mean_x_i, sigma^2*I).

    This function computes this average efficiently.

    Parameters
    ----------
      mean_x : torch.Tensor, shape (n_points, n_dim)
          Mean of X.

      var_x : torch.Tensor, shape ()
          Variance of X elements.

    Returns
    -------
      torch.Tensor, shape (n_dim, n_dim)
          Average second moment of projected gaussians.
    """
    sigma = torch.sqrt(var_x)
    n_points = mean_x.shape[0]
    n_dim = mean_x.shape[1]

    # Compute mean SM
    noise_w = torch.zeros(n_points, device=mean_x.device)
    mean_w = torch.zeros(n_points, device=mean_x.device)
    for i in range(n_points):
        noise_w[i], mean_w[i] = _iso_sm_weights(mean_x=mean_x[i, :], sigma=sigma)

    # Get the total weight of the identity across stim SM
    noise_w_mean = noise_w.mean()

    # Scale each stimulus by the sqrt of the outer prods weights
    mean_w_normalized = torch.sqrt(mean_w / (n_points)) / sigma
    mean_x_scaled = torch.einsum("nd,n->nd", mean_x, mean_w_normalized)

    # Compute average
    second_moment = (
        torch.einsum("nd,nb->db", mean_x_scaled, mean_x_scaled)
        + torch.eye(n_dim, device=mean_x.device) * noise_w_mean
    ) / n_points
    return second_moment


def _iso_sm_weights(mean_x, sigma):
    """
    Compute the weights of the mean outer product and of the identity
    matrix in the formula for the second moment matrix of the
    isotropic projected normal.

    Parameters
    ----------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      sigma : torch.Tensor, shape ()
          Standard deviation of the noise.

    Returns
    -------
      torch.Tensor, shape (n_points) 
          Weigths for the outer products of the means for
          each random variable.

      torch.Tensor, shape (n_points)
          Weights for the identity matrices.
    """
    n_dim = mean_x.shape[-1]
    non_centrality = torch.norm(mean_x / sigma, dim=-1) ** 2
    # Noise weights
    hyp_noise_val = sps.hyp1f1(1, n_dim / 2 + 1, -non_centrality / 2)
    noise_w = hyp_noise_val * (1 / n_dim)
    # Mean weights
    hyp_mean_val = sps.hyp1f1(1, n_dim / 2 + 2, -non_centrality / 2)
    mean_w = hyp_mean_val * (1 / (n_dim + 2))
    return noise_w, mean_w
