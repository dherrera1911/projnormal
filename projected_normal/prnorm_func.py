##################################
##################################
#
## PROJECTED NORMAL MOMENTS AND SAMPLING
#
##################################
##################################

#### Terminology:
# X : Random variable with multidimensional Gaussian distribution
# mu : Mean of X
# sigma : Standard deviation of X (if isotropic)
# covariance : Covariance of X
# n_dim : Dimensions of X
# Y : Random variable with projected Gaussian distribution
# gamma : Mean of Y
# psi : Covariance of Y

import numpy as np
import scipy.special as sps
import torch
import torch.nn as nn
import torch.distributions.multivariate_normal as mvn
import projected_normal.qf_func as qf
from projected_normal.auxiliary import non_centrality, is_symmetric, is_diagonal, product_trace, second_moment_2_cov



##################################
##################################
#
## PROJECTED NORMAL PDF
#
##################################
##################################


def prnorm_pdf(mu, covariance, y):
    """
    Compute the probability density function of a projected
    Gaussian distribution with parameters mu and covariance at points y.
    ----------------
    Arguments:
    ----------------
      - mu : Mean of the non-projected Gaussian. Shape (n_dim).
      - covariance : Covariance matrix of the non-projected Gaussian. Shape (n_dim x n_dim).
      - y : Points where to evaluate the PDF. Shape (n_points x n_dim).
    ----------------
    Outputs:
    ----------------
      - pdf : PDF evaluated at y. Shape (n_points).
    """
    n_dim = torch.tensor(mu.size(0))
    # Compute the precision matrix
    precision = torch.linalg.inv(covariance)

    # Compute the different terms
    q1 = torch.einsum("i,ij,j->", mu, precision, mu)
    q2 = torch.einsum("i,ij,j...->...", mu, precision, y.t())
    q3 = torch.einsum("...i,ij,j...->...", y, precision, y.t())
    alpha = q2 / torch.sqrt(q3)
    M = M_value(alpha, n_dim=n_dim)

    # Compute the PDF
    pdf = (
        ( (2.0 * torch.pi) ** (- n_dim / 2) )
        * 1 / torch.sqrt(torch.det(covariance))
        * (1 / ( q3 ** (n_dim / 2.0) ) )
        * torch.exp(0.5 * (alpha**2 - q1))
        * M
    )

    return pdf


def prnorm_log_pdf(mu, covariance, y):
    """
    Compute the log probability density function of a projected
    normal distribution with parameters mu and covariance.
    ----------------
    Arguments:
    ----------------
      - mu : Mean of the non-projected Gaussian. Shape (n_dim).
      - covariance : Covariance matrix of the non-projected Gaussian. Shape (n_dim x n_dim).
      - y : Points where to evaluate the PDF. Shape (n_points x n_dim).
    ----------------
    Outputs:
    ----------------
      - lpdf : log-PDF evaluated at y. Shape (n_points).
    """
    n_dim = torch.tensor(mu.size(0))
    # Compute the precision matrix
    precision = torch.linalg.inv(covariance)

    # Compute the terms
    q1 = torch.einsum("i,ij,j->", mu, precision, mu)
    q2 = torch.einsum("i,ij,j...->...", mu, precision, y.t())
    q3 = torch.einsum("...i,ij,j...->...", y, precision, y.t())
    alpha = q2 / torch.sqrt(q3)
    M = M_value(alpha, n_dim=n_dim)

    # Compute the log PDF
    term1 = -(n_dim / 2.0) * torch.log(torch.tensor(2.0) * torch.pi)
    term2 = -0.5 * torch.logdet(covariance)
    term3 = -(n_dim / 2.0) * torch.log(q3)
    term4 = 0.5 * (alpha**2 - q1)
    term5 = torch.log(M)
    lpdf = term1 + term2 + term3 + term4 + term5

    return lpdf


def M_value(alpha, n_dim):
    """
    Compute value of recursive function M in the projected normal pdf,
    with input alpha.
    ----------------
    Arguments:
    ----------------
      - alpha : Input to function M (n).
      - n_dim : Dimension of the non-projected Gaussian.
    ----------------
    Outputs:
    ----------------
      - M_vals : Value of M(alpha) (n).
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
    for i in range(3, n_dim+1):
        M_next =  (i - 2) * M_vals[0] + alpha * M_vals[1]
        M_vals[0] = M_vals[1].clone()
        M_vals[1] = M_next.clone()

    return M_vals[1]


def prnorm_c50_pdf(mu, covariance, c50, y):
    """
    Compute the probability density function of projected
    normal Y = X/(X'X + c50)^0.5 at points y.
    implement by projecting y into x, computing the pdf
    and multiplying by the jacobian of the projection.
    ----------------
    Arguments:
    ----------------
      - mu : Mean of the non-projected Gaussian. Shape (n_dim).
      - covariance : Covariance matrix of the non-projected Gaussian. Shape (n_dim x n_dim).
      - c50 : Constant added to the denominator. Scalar
      - y : Points where to evaluate the PDF. Shape (n_points x n_dim).
    ----------------
    Outputs:
    ----------------
      - pdf : PDF evaluated at y. Shape (n_points).
    """
    log_pdf = prnorm_c50_log_pdf(mu, covariance, c50, y)
    # Invert the projection
    pdf = torch.exp(log_pdf)
    return pdf

def prnorm_c50_log_pdf(mu, covariance, c50, y):
    """
    Compute the probability density function of projected
    normal Y = X/(X'X + c50)^0.5 at points y.
    implement by projecting y into x, computing the pdf
    and multiplying by the jacobian of the projection.
    ----------------
    Arguments:
    ----------------
      - mu : Mean of the non-projected Gaussian. Shape (n_dim).
      - covariance : Covariance matrix of the non-projected Gaussian. Shape (n_dim x n_dim).
      - c50 : Constant added to the denominator. Scalar
      - y : Points where to evaluate the PDF. Shape (n_points x n_dim).
    ----------------
    Outputs:
    ----------------
      - log_pdf : PDF evaluated at y. Shape (n_points).
    """
    # Verify that c50 is positive
    if c50 <= 0:
        raise ValueError("c50 must be a positive scalar value.")
    # Invert the projection
    X = invert_projection(y, c50)
    # Compute the PDF under the normal distribution
    normal_dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    log_pdf = normal_dist.log_prob(X)
    # Compute the jacobian of the inverse projection
    J_log_det = invert_projection_log_det(y, c50)
    # Compute the PDF
    log_pdf = log_pdf + J_log_det
    return log_pdf


##################################
##################################
#
## PROJECTED NORMAL MOMENTS
#
##################################
##################################

#############
#### GENERAL CASE, TAYLOR APPROXIMATION
#############


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
      mu=mu,
      covariance=covariance,
      M_diagonal=B_diagonal
    )

    denominator_mean = denominator_mean + c50
    denominator_var = qf.quadratic_form_var_diagonal(
      mu=mu,
      covariance=covariance,
      M_diagonal=B_diagonal
    )

    # Compute covariance between numerator and denominator for each
    # matrix A^{ij}
    term1 = torch.einsum("ij,j,jk->ik", covariance, B_diagonal, covariance)
    term2 = torch.einsum("i,j,j,jk->ik", mu, mu, B_diagonal, covariance)
    numerator_denominator_cov = 2 * (term1 + term2 + term2.transpose(0, 1))

    # Compute second moment of projected normal
    epsilon = 1e-6 # Small value to avoid division by zero
    second_moment = (
        numerator_mean / denominator_mean *
      (1 -
       numerator_denominator_cov / (numerator_mean * denominator_mean + epsilon) +
       denominator_var / denominator_mean**2)
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
    var_X2 = qf.quadratic_form_var_diagonal(mu=mu, covariance=covariance,
                                            M_diagonal=B_diagonal)

    # Next, Compute the term to subtract for each X_i, with the
    # X_i-dependent elements
    term1 = (
        2 * B_diagonal * torch.einsum("ij,ji->i", covariance * B_diagonal, covariance) -
        B_diagonal**2 * covariance.diag()**2
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


#############
#### ISOTROPIC CASE, EXACT SOLUTION
#############


def prnorm_mean_iso(mu, sigma):
    """
    Compute the expected value of projected gaussian Y = X/||X||,
    where X~N(mu, I*sigma^2).
    ----------------
    Arguments:
    ----------------
      - mu : Mean of X. (n_dim)
      - sigma : Standard deviation of X elements (Scalar)
    ----------------
    Outputs:
    ----------------
      - gamma : Expected value of projected normal. Shape (n_dim).
    """
    n_dim = torch.as_tensor(mu.shape[-1])
    nc = non_centrality(mu, sigma)

    # Compute terms separately
    gln1 = torch.special.gammaln((n_dim + 1) / 2)
    gln2 = torch.special.gammaln(n_dim / 2 + 1)
    g_ratio = 1 / (np.sqrt(2) * sigma) * torch.exp(gln1 - gln2)
    hyp_val = sps.hyp1f1(1 / 2, n_dim / 2 + 1, -nc / 2)

    # Multiply terms to get the expected value
    gamma = torch.einsum("...d,...->...d", mu, g_ratio * hyp_val)

    return gamma


# Apply the isotropic covariance formula to get the covariance
# for each stimulus
def prnorm_sm_iso(mu, sigma):
    """
    Compute the second moment of the projected normal with
    isotropic covariance, Y = X/||X||, where X~N(mu, sigma^2*I).
    ----------------
    Arguments:
    ----------------
      - mu : Mean of X (n_dim)
      - sigma : Standard deviation of X elements (Scalar)
    ----------------
    Outputs:
    ----------------
      - second_moment : Second moment matrix projected normal. Shape (n_dim x n_dim)
    """
    n_dim = torch.as_tensor(mu.shape[-1])
    # Compute weights for mean and identity terms
    noise_w, mean_w = iso_sm_weights(mu=mu, sigma=sigma)

    # Compute the second moment of each stimulus
    mu_normalized = mu / sigma
    # Get the outer product of the normalized stimulus, and multiply by weight
    second_moment = torch.einsum("...d,...b,...->...db",
                                 mu_normalized,
                                 mu_normalized,
                                 mean_w
                                )

    # Add noise term to the diagonal
    diag_idx = torch.arange(n_dim)

    is_batch = mu.dim() == 2
    if is_batch:
        n_batch = mu.shape[0]
        for i in range(n_batch):
            second_moment[i, diag_idx, diag_idx] += noise_w[i]
    else:
        second_moment[diag_idx, diag_idx] += noise_w

    return second_moment


def prnorm_sm_iso_batch(mu, sigma):
    """
    Get the second moment of the sum of projected normals. This saves a lot
    of computation time by not adding the indentity matrix to each second
    moment matrix.
    ----------------
    Arguments:
    ----------------
      - mu : Means of normal distributions X. (n_points x n_dim)
      - sigma : Standard deviation of the normal distributions (isotropic)
    ----------------
    Outputs:
    ----------------
      - second_moment : Second of projected gaussian sum. Shape (n_dim x n_dim)
    """
    n_points = mu.shape[0]
    n_dim = mu.shape[1]

    # Compute mean SM
    noise_w = torch.zeros(n_points, device=mu.device)
    mean_w = torch.zeros(n_points, device=mu.device)
    for i in range(n_points):
        noise_w[i], mean_w[i] = iso_sm_weights(mu=mu[i, :], sigma=sigma)

    # Compute the second moment of each stimulus
    mu_normalized = mu / sigma
    # Get the total weight of the identity across stim SM
    noise_w_mean = noise_w.mean()

    # Scale each stimulus by the sqrt of the outer prods weights
    mean_w_normalized = torch.sqrt(mean_w / (n_points)) / sigma
    mu_scaled = torch.einsum("nd,n->nd", mu, mean_w_normalized)

    # Add together
    second_moment = (
        torch.einsum("nd,nb->db", mu_scaled, mu_scaled)
        + torch.eye(n_dim, device=mu.device) * noise_w_mean
    )
    return second_moment


def iso_sm_weights(mu, sigma):
    """
    Compute the weights of the mean outer product and of the identity
    matrix in the formula for the second moment matrix of the
    isotropic projected normal.
    ----------------
    Arguments:
    ----------------
      - mu : Mean of X. (n_points x n_dim)
      - sigma : Standard deviation of the noise
      - nc : Non centrality parameter of each X. If None,
          then it's computed by the function(n_points)
    ----------------
    Outputs:
    ----------------
      - mean_w : Weigths for the outer products of the means for
          each random variable. (n_points)
      - noise_w : Weights for the identity matrices. (n_points)
    """
    n_dim = mu.shape[-1]
    nc = non_centrality(mu=mu, sigma=sigma)
    # Noise weights
    hyp_noise_val = sps.hyp1f1(1, n_dim / 2 + 1, -nc / 2)
    noise_w = hyp_noise_val * (1 / n_dim)
    # Mean weights
    hyp_mean_val = sps.hyp1f1(1, n_dim / 2 + 2, -nc / 2)
    mean_w = hyp_mean_val * (1 / (n_dim + 2))
    return noise_w, mean_w


#############
#### CASE WITH C50
#############

def invert_projection(y, c50):
    """
    Invert the projection y = X/(X'X + c50)^0.5
    ----------------
    Arguments:
    ----------------
      - y : Observed points in the ball. (n_points x n_dim)
      - c50 : Constant added to the denominator. Scalar
    ----------------
    Outputs:
    ----------------
      - X : Pre-projection points. (n_points x n_dim)
    """
    scaling = torch.sqrt(c50 / (1 - torch.sum(y**2, dim=-1)))
    X = torch.einsum("...d,...->...d", y, scaling)
    return X


def invert_projection_jacobian_matrix(y, c50):
    """
    Compute the jacobian matrix of the inverse projection.
    ----------------
    Arguments:
    ----------------
      - y : Observed points in the ball. (n_points x n_dim)
      - c50 : Constant added to the denominator. Scalar
    ----------------
    Outputs:
    ----------------
      - J : Jacobian matrix of the inverse projection. (n_points x n_dim x n_dim)
    """
    n_dim = y.shape[-1]
    y_sq_norm = torch.sum(y**2, dim=-1)
    J_multiplier = torch.sqrt(c50 / (1 - y_sq_norm))
    J_matrix = torch.einsum("...d,...e->...de", y, y / (1 - y_sq_norm.view(-1, 1)))
    # Add identity to the diagonal
    diag_idx = torch.arange(n_dim)
    is_batch = y.dim() == 2
    if is_batch:
        # Make identity matrix for each batch
        n_batch = y.shape[0]
        I = torch.eye(n_dim, device=y.device).unsqueeze(0).expand(n_batch, -1, -1)
        J_matrix += I
    else:
        J_matrix += torch.eye(n_dim, device=y.device)
    # Put multiplier and matrix together
    J = torch.einsum("n,nij->nij", J_multiplier, J_matrix)
    return J


def invert_projection_det(y, c50):
    """
    Efficiently compute the determinant of the jacobian matrix.
    Uses the matrix determinant lemman that states that
    det(I + uv') = 1 + v'u and det(cA) = c^n det(A) for a scalar c and
    matrix A.
    ----------------
    Arguments:
    ----------------
      - y : Observed points in the ball. (n_points x n_dim)
      - c50 : Constant added to the denominator. Scalar
    ----------------
    Outputs:
    ----------------
      - det : Determinant of the jacobian matrix. (n_points)
    """
    log_det = invert_projection_log_det(y, c50)
    det = torch.exp(log_det)
    return det


def invert_projection_log_det(y, c50):
    """
    Efficiently compute the determinant of the jacobian matrix.
    Uses the matrix determinant lemman that states that
    det(I + uv') = 1 + v'u and det(cA) = c^n det(A) for a scalar c and
    matrix A.
    ----------------
    Arguments:
    ----------------
      - y : Observed points in the ball. (n_points x n_dim)
      - c50 : Constant added to the denominator. Scalar
    ----------------
    Outputs:
    ----------------
      - det : Determinant of the jacobian matrix. (n_points)
    """
    n_dim = y.shape[-1]
    y_sq_norm = torch.sum(y**2, dim=-1)
    scalar = torch.sqrt(c50 / (1 - y_sq_norm)) # Scalar from Jacobian matrix formula
    det_1 = 1 + y_sq_norm / (1 - y_sq_norm) # Matrix determinant lemma
    det = n_dim * torch.log(scalar) + torch.log(det_1) # Scalar multiplication determinant property
    return det


##################################
##################################
#
## PROJECTED NORMAL SAMPLING
#
##################################
##################################

########## Change code to have B_diagonal, or not?

def sample_prnorm(mu, covariance, n_samples, B=None, c50=0):
    """
    Sample from the generalized projected normal distribution
    Y = X/(X'BX + c50)^0.5, where X~N(mu, covariance).
    -----------------
    Arguments:
    -----------------
      - mu : Mean of X. (n_dim)
      - covariance : Covariance matrix of X. (n_dim x n_dim)
      - n_samples : Number of samples.
      - B : Matrix in the denominator. (n_dim x n_dim)
      - c50 : Constant added to the denominator. Scalar
    -----------------
    Output:
    -----------------
      - samples_prnorm : Samples from the generalized projected normal. (n_samples x n_dim)
    """
    if B is None:
        B = torch.ones(len(mu))

    # Initialize Gaussian distribution to sample from
    dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)

    # Take n_samples
    X = dist.sample([n_samples])

    # Compute normalizing quadratic form
    if B.dim() == 1:
        q = torch.sqrt(torch.einsum("ni,i,in->n", X, B, X.t()) + c50)
    else:
        q = torch.sqrt(torch.einsum("ni,ij,jn->n", X, B, X.t()) + c50)

    # Normalize normal distribution samples
    samples_prnorm = torch.einsum("ni,n->ni", X, 1 / q)

    return samples_prnorm


def empirical_moments_prnorm(mu, covariance, n_samples, B=None, c50=0):
    """
    Compute the mean, covariance and second moment of the projected normal
    by sampling from the distribution.
    -----------------
    Arguments:
    -----------------
      - mu : Mean. (n_dim)
      - covariance : Covariance matrix. (n_dim x n_dim)
      - n_samples : Number of samples.
      - B : Matrix in the denominator. (n_dim x n_dim)
      - c50 : Constant added to the denominator. Scalar
    -----------------
    Output:
    -----------------
      Dictionary with the following keys:
      - gamma : Mean of the projected normal. (n_dim)
      - psi : Covariance of the projected normal. (n_dim x n_dim)
      - second_moment : Second moment of the projected normal. (n_dim x n_dim)
    """
    samples = sample_prnorm(mu, covariance, n_samples=n_samples, B=B, c50=c50)
    gamma = torch.mean(samples, dim=0)
    second_moment = torch.einsum("in,nj->ij", samples.t(), samples) / n_samples
    psi = second_moment_2_cov(second_moment, gamma)
    return {"gamma": gamma, "psi": psi, "second_moment": second_moment}

