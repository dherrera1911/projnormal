"""Approximation to the moments of the general projected normal distribution with additive const term in denominator."""
import torch
import projnormal.quadratic_forms.moments as qfm


__all__ = ["mean", "second_moment"]


def __dir__():
    return __all__


def mean(mean_x, covariance_x, const):
    """
    Compute the Taylor approximation to the variable Y = X/(X'X + const)^0.5,
    where X~N(mean_x, covariance_x). Y has a projected normal distribution with an extra
    constant const added to the denominator.

    The approximation is based on the function
    f(u,v) = u/sqrt(u^2 + v + const), where u=X_i and v = (X'X - X_i^2).

    Parameters:
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

      const : torch.Tensor, shape ()
        Constant added to the denominator. Must be >= 0.

    Returns:
    ----------------
      torch.Tensor, shape (n_dim,)
          Expected value for the projected normal.
    """
    # Process inputs
    variances = torch.diagonal(covariance_x)

    ### Get moments of variable v (||X||^2 - X_i^2) required for the formula
    # Note, these don't depend on const
    v_mean = _get_v_mean(mean_x=mean_x, covariance_x=covariance_x)
    v_var = _get_v_var(mean_x=mean_x, covariance_x=covariance_x)
    v_cov = _get_v_cov(mean_x=mean_x, covariance_x=covariance_x)

    ### Get the derivatives for the taylor approximation evaluated
    ### at the mean of u and v
    dfdu2_val = _get_dfdu2(u=mean_x, v=v_mean, const=const)
    dfdv2_val = _get_dfdv2(u=mean_x, v=v_mean, const=const)
    dfdudv_val = _get_dfdudv(u=mean_x, v=v_mean, const=const)

    ### 0th order term
    term0 = _get_f0(u=mean_x, v=v_mean, const=const)

    ### Compute Taylor approximation
    gamma = (
        term0
        + 0.5 * dfdu2_val * variances
        + 0.5 * dfdv2_val * v_var
        + dfdudv_val * v_cov
    )

    return gamma


def second_moment(mean_x, covariance_x, const):
    """
    Compute the Taylor approximation to the second moment matrix of the
    variable Y = X/(X'X + const)^0.5, where X~N(mean_x, covariance_x). Y has a
    projected normal distribution with an extra constant const added to the denominator.

    The approximation is based on the Taylor expansion of the
    function f(n,d) = n/d, where n = X_i*X_j and d = X'X + const.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

      const : torch.Tensor, shape ()
        Constant added to the denominator. Must be >= 0.

    Returns
    ----------------
      torch.Tensor, shape (n_dim, n_dim)
          Second moment matrix of Y
    """
    # Compute the mean of numerator for each matrix A^{ij}
    numerator_mean = covariance_x + torch.einsum("d,b->db", mean_x, mean_x)

    # Compute denominator terms
    denominator_mean = qfm.mean(
        mean_x=mean_x, covariance_x=covariance_x
    )
    denominator_mean = denominator_mean + const
    denominator_var = qfm.variance(
        mean_x=mean_x, covariance_x=covariance_x
    )

    # Compute covariance between numerator and denominator for each
    # matrix A^{ij}
    term1 = torch.einsum("ij,jk->ik", covariance_x, covariance_x)
    term2 = torch.einsum("i,j,jk->ik", mean_x, mean_x, covariance_x)
    numerator_denominator_cov = 2 * (term1 + term2 + term2.transpose(0, 1))

    # Compute second moment of projected normal
    epsilon = 1e-6  # Small value to avoid division by zero
    sm = (
        numerator_mean
        / denominator_mean
        * (
            1
            - numerator_denominator_cov / (numerator_mean * denominator_mean + epsilon)
            + denominator_var / denominator_mean**2
        )
    )

    return sm


def _get_v_mean(mean_x, covariance_x):
    """
    Compute the expected value of the auxiliary variables
    v_i = (X'X - X_i^2) used in the Taylor approximation
    of the projected normal mean, where X~N(mean_x, covariance_x).

    Computes the expected value of each v_i.

    Parameters:
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

    Returns:
    ----------------
      torch.Tensor, shape (n_dim,)
          Expected value of auxiliary variable V.
    """
    # Get variances
    variances = covariance_x.diagonal()
    # Compute the expected value of X'X
    mean_X2 = torch.sum(variances) + torch.einsum("i,i->", mean_x, mean_x)
    # Compute expected value of each elements individual quadratic form
    # i.e. E(X_i^2) vector
    mean_Xi2 = mean_x**2 + variances
    # Subtract to get E(X'X - X_i^2)
    v_mean = mean_X2 - mean_Xi2

    return v_mean


def _get_v_var(mean_x, covariance_x):
    """
    Compute the variance of the auxiliary variables
    v_i = (X'X - X_i^2) used in the Taylor approximation
    of the projected normal mean, where X~N(mean_x, covariance_x).
    Computes the variance of each v_i.

    Parameters:
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

    Returns:
    ----------------
      torch.Tensor, shape (n_dim,)
          Variance of auxiliary variable V.
    """
    # Compute the variance of the quadratic form X'X
    var_X2 = qfm.variance(
        mean_x=mean_x, covariance_x=covariance_x
    )

    # Next, Compute the term to subtract for each X_i, with the
    # X_i-dependent elements
    term1 = (
        2 * torch.einsum("ij,ji->i", covariance_x, covariance_x)
        - covariance_x.diag() ** 2
    )  # Repeated terms in the trace

    term2 = (
        2 * torch.einsum("i,ij,j->i", mean_x, covariance_x, mean_x)
        - mean_x**2 * covariance_x.diag()
    )  # Repeated terms in (mean_x' Cov mean_x)

    # Subtract to get variance
    v_var = var_X2 - 2 * term1 - 4 * term2

    return v_var


def _get_v_cov(mean_x, covariance_x):
    """
    Compute the covariance between the auxiliary variables
    v_i = (X'X - X_i^2) used in the Taylor approximation
    of the projected normal mean, where X~N(mean_x, covariance_x).
    Computes the covariance between each element of V and the
    corresponding X_i.

    Parameters:
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

    Returns:
    ----------------
      torch.Tensor, shape (n_dim,)
          Covariance between each element of V and the corresponding X_i.
    """
    v_cov = 2 * (
        torch.einsum("i,ij->j", mean_x, covariance_x)
        - mean_x * torch.diagonal(covariance_x)
    )
    return v_cov


# Derivatives of the function f(u,v) = u/sqrt(u^2 + v + const)
# that is used in the taylor approximation to the mean
def _get_f0(u, v, const):
    """
    First term of the Taylor approximation of f(u,v) = u/sqrt(u^2 + v),
    evaluated at point u,v.
    """
    f0 = u / torch.sqrt(u**2 + v + const)
    return f0


def _get_dfdu2(u, v, const):
    """
    Second derivative of f(u,v) = u/sqrt(c*u^2 + v) wrt u,
    evaluated at point u,v.
    """
    dfdu2 = -3 * u * (v + const) / (u**2 + v + const) ** (5 / 2)
    return dfdu2


def _get_dfdv2(u, v, const):
    """
    Second derivative of f(u,v) = u/sqrt(u^2 + v) wrt v,
    evaluated at point u,v.
    """
    dfdv2 = 0.75 * u / (u**2 + v + const) ** (5 / 2)
    return dfdv2


def _get_dfdudv(u, v, const):
    """
    Mixed second derivative of f(u,v) = u/sqrt(u^2 + v),
    evaluated at point u,v.
    """
    dfdudv = (u**2 - 0.5 * (v + const)) / (u**2 + v + const) ** (5 / 2)
    return dfdudv
