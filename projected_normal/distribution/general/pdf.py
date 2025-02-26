"""Probability density function (PDF) for the general projected normal distribution."""
import torch


__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mean_x, covariance_x, y):
    """
    Compute the probability density function at points y for the variable
    Y = X/(X'X)^0.5 where X ~ N(mean_x, covariance_x). Y has a general projected
    normal distribution.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      var_x : torch.Tensor, shape ()
          Variance of X elements.

      y : torch.Tensor, shape (n_points, n_dim)
          Points where to evaluate the PDF.

    Returns
    ----------------
      torch.Tensor, shape (n_points)
          PDF evaluated at each y.
    """
    lpdf = log_pdf(mean_x, covariance_x, y)
    pdf = torch.exp(lpdf)
    return pdf


def log_pdf(mean_x, covariance_x, y):
    """
    Compute the log probability density function at points y for the variable
    Y = X/(X'X)^0.5 where X ~ N(mean_x, covariance_x). Y has a general projected
    normal distribution.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      var_x : torch.Tensor, shape ()
          Variance of X elements.

      y : torch.Tensor, shape (n_points, n_dim)
          Points where to evaluate the PDF.

    Returns
    ----------------
      torch.Tensor, shape (n_points)
          Log-PDF evaluated at each y.
    """
    n_dim = torch.tensor(mean_x.size(0))
    # Compute the precision matrix
    precision = torch.linalg.inv(covariance_x)

    # Compute the terms
    q1 = torch.einsum("i,ij,j->", mean_x, precision, mean_x)
    q2 = torch.einsum("i,ij,j...->...", mean_x, precision, y.t())
    q3 = torch.einsum("...i,ij,j...->...", y, precision, y.t())
    alpha = q2 / torch.sqrt(q3)
    M = _M_value(alpha, n_dim=n_dim)

    # Compute the log PDF
    term1 = -(n_dim / 2.0) * torch.log(torch.tensor(2.0) * torch.pi)
    term2 = -0.5 * torch.logdet(covariance_x)
    term3 = -(n_dim / 2.0) * torch.log(q3)
    term4 = 0.5 * (alpha**2 - q1)
    term5 = torch.log(M)
    lpdf = term1 + term2 + term3 + term4 + term5
    return lpdf


def _M_value(alpha, n_dim):
    """
    Compute value of recursive function M in the projected normal pdf, with input alpha.

    Parameters
    ----------------
      alpha : torch.Tensor, shape (n_points)
          Input to function M.

      n_dim : int
          Dimension of the non-projected Gaussian.

    Returns
    ----------------
      torch.Tensor, shape(n_points)
          Value of M(alpha).
    """
    # Create a standard normal distribution
    normal_dist = torch.distributions.Normal(0, 1)
    # Calculate the cumulative distribution function (CDF) of alpha
    norm_cdf = normal_dist.cdf(alpha)

    # Calculate unnormalized pdf
    exp_alpha = torch.exp(-0.5 * alpha**2)

    # Calculate the value of M recursively
    # List with values to modify iteratively
    M1 = torch.sqrt(torch.tensor(2.0) * torch.pi) * norm_cdf
    M2 = exp_alpha + alpha * M1
    M_vals = [M1, M2]
    for i in range(3, n_dim + 1):
        M_next = (i - 2) * M_vals[0] + alpha * M_vals[1]
        M_vals[0] = M_vals[1].clone()
        M_vals[1] = M_next.clone()

    return M_vals[1]
