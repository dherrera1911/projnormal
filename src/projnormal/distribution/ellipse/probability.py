"""Probability density function (PDF) for the general projected normal distribution."""

__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mean_x, covariance_x, y, B=None):
    """
    Compute the probability density function (PDF) for the
    variable Y = X/sqrt(X'BX) where X ~ N(mean_x, covariance_x).
    It is not currently implemented.
    """
    raise NotImplementedError(
        "The PDF for the projected normal distribution with \
      denominator \sqrt(x'Bx) is not implemented. "
    )


def log_pdf(mean_x, covariance_x, y, B=None):
    """
    Compute the log probability density function (log PDF) for the
    variable Y = X/sqrt(X'BX) where X ~ N(mean_x, covariance_x).
    It is not currently implemented.
    """
    raise NotImplementedError(
        "The log PDF for the projected normal distribution with \
      denominator \sqrt(x'Bx) is not implemented. "
    )
