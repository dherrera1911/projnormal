"""Probability density function (PDF) for the general projected normal distribution."""
import torch


__all__ = ["mean", "second_moment"]


def __dir__():
    return __all__


def mean(mu, covariance, B_diagonal=None, c50=0):
    """
    Compute the Taylor approximation to the variable Y = X/(X'BX + c50)^0.5,
    where X~N(mu, covariance) and B is a diagonal matrix. Y has a projected
    normal distribution with an extra constant c50 added to the denominator.

    The approximation is based on the function
    f(u,v) = u/sqrt(b*u^2 + v + c50), where u=X_i and v = (X'BX - B_{ii}X_i^2).

    Parameters:
    ----------------
      - mu : Means of normal distributions X. (n_dim)
      - covariance : covariance of X. (n_dim x n_dim)
      - B_diagonal : Diagonal elements of B. Default is B=I (n_dim)
      - c50 : Constant added to the denominator. Scalar

    Returns:
    ----------------
      Expected mean value for each projected normal. Shape (n_dim)
    """
    # Process inputs
    if B_diagonal is None:
        B_diagonal = torch.ones(len(mu))
    variances = torch.diagonal(covariance)

    ### Get moments of variable v (||X||^2 - X_i^2) required for the formula
    # Note, these don't depend on c50
    v_mean = get_v_mean(mu=mu, covariance=covariance, B_diagonal=B_diagonal)
    v_var = get_v_var(mu=mu, covariance=covariance, B_diagonal=B_diagonal)
    v_cov = get_v_cov(mu=mu, covariance=covariance, B_diagonal=B_diagonal)

    ### Get the derivatives for the taylor approximation evaluated
    ### at the mean of u and v
    dfdu2_val = dfdu2(u=mu, v=v_mean, b=B_diagonal, c50=c50)
    dfdv2_val = dfdv2(u=mu, v=v_mean, b=B_diagonal, c50=c50)
    dfdudv_val = dfdudv(u=mu, v=v_mean, b=B_diagonal, c50=c50)

    ### 0th order term
    term0 = f0(u=mu, v=v_mean, b=B_diagonal, c50=c50)

    ### Compute Taylor approximation
    gamma = (
        term0
        + 0.5 * dfdu2_val * variances
        + 0.5 * dfdv2_val * v_var
        + dfdudv_val * v_cov
    )

    return gamma


def prnorm_sm_taylor(mu, covariance, B_diagonal=None, c50=0):
    """
    Approximate the second moment matrix of the generalized projected normal
    distribution with a Taylor expansion. The projected normal is
    Y = X/(X'BX + c50)^0.5, where X~N(mu, covariance).

    The approximation is based on the Taylor expansion of the
    function f(n,d) = n/d, where n = X_i*X_j and d = X'BX + c50.
    ----------------
    Arguments:
    ----------------
      - mu : Means of normal distributions X. (n_dim)
      - covariance : Covariance of the normal distributions (n_dim x n_dim)
      - B_diagonal : Diagonal elements of B. (n_dim) Default is B=I
      - c50 : Constant added to the denominator. Scalar
    ----------------
    Outputs:
    ----------------
      - second_moment : Second moment of each projected gaussian.
          Shape (n_dim x n_dim)
    """
    # Process inputs
    if B_diagonal is None:
        B_diagonal = torch.ones(len(mu))

    # Compute the mean of numerator for each matrix A^{ij}
    numerator_mean = covariance + torch.einsum("d,b->db", mu, mu)

    # Compute denominator terms
    denominator_mean = qf.quadratic_form_mean_diagonal(
        mu=mu, covariance=covariance, M_diagonal=B_diagonal
    )

    denominator_mean = denominator_mean + c50
    denominator_var = qf.quadratic_form_var_diagonal(
        mu=mu, covariance=covariance, M_diagonal=B_diagonal
    )

    # Compute covariance between numerator and denominator for each
    # matrix A^{ij}
    term1 = torch.einsum("ij,j,jk->ik", covariance, B_diagonal, covariance)
    term2 = torch.einsum("i,j,j,jk->ik", mu, mu, B_diagonal, covariance)
    numerator_denominator_cov = 2 * (term1 + term2 + term2.transpose(0, 1))

    # Compute second moment of projected normal
    epsilon = 1e-6  # Small value to avoid division by zero
    second_moment = (
        numerator_mean
        / denominator_mean
        * (
            1
            - numerator_denominator_cov / (numerator_mean * denominator_mean + epsilon)
            + denominator_var / denominator_mean**2
        )
    )

    return second_moment


def get_v_mean(mu, covariance, B_diagonal):
    """
    Compute the expected value of the auxiliary variables
    v_i = (X'BX - B_{ii}X_i^2) used in the Taylor approximation
    of the projected normal mean, where X~N(mu, covariance).

    Computes the expected value of each v_i.
    ----------------
    Arguments:
    ----------------
      - mu : Mean of X. (n_dim)
      - covariance : Covariance of X. (n_dim x n_dim)
      - B_diagonal : Diagonal elements of matrix B (n_dim).
    ----------------
    Outputs:
    ----------------
      - v_mean : Expected value of V (n_dim)
    """
    # Get variances
    variances = covariance.diagonal()
    variances = B_diagonal * variances

    # Compute the expected value of X'BX
    mean_X2 = torch.sum(variances) + torch.einsum("i,i->", mu, mu * B_diagonal)

    # Compute expected value of each elements individual quadratic form
    # i.e. E(B_i*X_i^2) vector
    mean_Xi2 = mu**2 * B_diagonal + variances

    # Subtract to get E(X'BX - B_i*X_i^2)
    v_mean = mean_X2 - mean_Xi2

    return v_mean


def get_v_var(mu, covariance, B_diagonal):
    """
    Compute the variance of the auxiliary variables
    v_i = (X'BX - B_{ii}X_i^2) used in the Taylor approximation
    of the projected normal mean, where X~N(mu, covariance).
    Computes the variance of each v_i.
    ----------------
    Arguments:
    ----------------
      - mu : Mean of X. (n_dim)
      - covariance : Covariance of X. (n_dim x n_dim)
      - B_diagonal : Diagonal elements of matrix B (n_dim).
    ----------------
    Outputs:
    ----------------
      - v_var : Variance of each element of V (n_dim)
    """
    # Compute the variance of the quadratic form X'BX
    var_X2 = qf.quadratic_form_var_diagonal(
        mu=mu, covariance=covariance, M_diagonal=B_diagonal
    )

    # Next, Compute the term to subtract for each X_i, with the
    # X_i-dependent elements
    term1 = (
        2 * B_diagonal * torch.einsum("ij,ji->i", covariance * B_diagonal, covariance)
        - B_diagonal**2 * covariance.diag() ** 2
    )  # Repeated terms in the trace

    term2 = (
        2 * torch.einsum("i,ij,j->i", mu * B_diagonal, covariance, mu * B_diagonal)
        - mu**2 * covariance.diag() * B_diagonal**2
    )  # Repeated terms in (mu' B Cov B mu)

    # Subtract to get variance
    v_var = var_X2 - 2 * term1 - 4 * term2

    return v_var


def get_v_cov(mu, covariance, B_diagonal):
    """
    Compute the covariance between the auxiliary variables
    v_i = (X'BX - B_{ii}X_i^2) used in the Taylor approximation
    of the projected normal mean, where X~N(mu, covariance).
    Computes the covariance between each element of V and the
    corresponding X_i.
    ----------------
    Arguments:
    ----------------
      - mu : Means of normal distributions X. (n_dim)
      - covariance : covariance of X. (n_dim x n_dim)
      - B_diagonal : Diagonal elements of matrix B (n_dim).
    ----------------
    Outputs:
    ----------------
      - v_cov : Covariance between each element of V and the
          corresponding X_i (n_dim)
    """
    v_cov = 2 * (
        torch.einsum("i,ij->j", mu * B_diagonal, covariance)
        - mu * B_diagonal * torch.diagonal(covariance)
    )
    return v_cov


# Derivatives of the function f(u,v) = u/sqrt(u^2 + v + c50)
# that is used in the taylor approximation to the mean
def f0(u, v, b, c50):
    """
    First term of the Taylor approximation of f(u,v) = u/sqrt(b*u^2 + v),
    evaluated at point u,v. b is a constant
    """
    f0 = u / torch.sqrt(b * u**2 + v + c50)
    return f0


def dfdu2(u, v, b, c50):
    """
    Second derivative of f(u,v) = u/sqrt(c*u^2 + v) wrt u,
    evaluated at point u,v. b is a constant
    """
    dfdu2 = -3 * b * u * (v + c50) / (b * u**2 + v + c50) ** (5 / 2)
    return dfdu2


def dfdv2(u, v, b, c50):
    """
    Second derivative of f(u,v) = u/sqrt(b*u^2 + v) wrt v,
    evaluated at point u,v. b is a constant
    """
    dfdv2 = 0.75 * u / (b * u**2 + v + c50) ** (5 / 2)
    return dfdv2


def dfdudv(u, v, b, c50):
    """
    Mixed second derivative of f(u,v) = u/sqrt(b*u^2 + v),
    evaluated at point u,v. b is a constant
    """
    dfdudv = (b * u**2 - 0.5 * (v + c50)) / (b * u**2 + v + c50) ** (5 / 2)
    return dfdudv


