"""Approximation to the moments of the projected normal distribution."""
import torch


def prnorm_mean_taylor(mu, covariance, B_diagonal=None, c50=0):
    """
    Approximate the mean of the generalized projected normal distribution with
    a taylor expansion. The projected normal is Y = X/(X'BX + c50)^0.5,
    where X~N(mu, covariance).

    The approximation is based on the function
    f(u,v) = u/sqrt(b*u^2 + v + c50), where u=X_i and v = (X'BX - B_{ii}X_i^2).

    This function works for diagonal B only.
    ----------------
    Arguments:
    ----------------
      - mu : Means of normal distributions X. (n_dim)
      - covariance : covariance of X. (n_dim x n_dim)
      - B_diagonal : Diagonal elements of B. Default is B=I (n_dim)
      - c50 : Constant added to the denominator. Scalar
    ----------------
    Outputs:
    ----------------
      - gamma : Expected mean value for each projected normal. Shape (n_dim)
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

