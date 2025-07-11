"""Probability density function (PDF) for the general projected normal distribution."""

__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mean_x, covariance_x, y, B=None):
    r"""
    Compute the pdf at points y for the distribution of the variable
    :math:`y = x/\sqrt{x^T B x}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`B` is a symmetric positive definite matrix.


    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
        Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      y : ``torch.Tensor``
          Points where to evaluate the PDF. Shape is ``(n_points, n_dim)``.

      B : ``torch.Tensor``, optional
          Matrix B used in the denominator of the projection. If not provided,
          the identity matrix is used. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          PDF evaluated at each y. Shape is ``(n_points,)``.
    """
    raise NotImplementedError(
        "The PDF for the projected normal distribution with \
      denominator \sqrt(x'Bx) is not implemented. "
    )


def log_pdf(mean_x, covariance_x, y, B=None):
    r"""
    Compute the log-pdf at points y for the distribution of the variable
    :math:`y = x/\sqrt{x^T B x}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`B` is a symmetric positive definite matrix.


    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
        Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      y : ``torch.Tensor``
          Points where to evaluate the PDF. Shape is ``(n_points, n_dim)``.

      B : ``torch.Tensor``, optional
          Matrix B used in the denominator of the projection. If not provided,
          the identity matrix is used. Shape is ``(n_dim, n_dim)``.

    Returns
    -------
      ``torch.Tensor``
          Log-PDF evaluated at each y. Shape is ``(n_points,)``.
    """
    raise NotImplementedError(
        "The log PDF for the projected normal distribution with \
      denominator \sqrt(x'Bx) is not implemented. "
    )
