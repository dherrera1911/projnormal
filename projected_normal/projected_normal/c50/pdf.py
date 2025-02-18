"""Probability density function (PDF) for the general projected normal distribution."""
import torch
import torch.distributions.multivariate_normal as mvn


__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mu, covariance, c50, y):
    """
    Compute the probability density function of projected
    normal Y = X/(X'X + c50)^0.5 at points y.

    Parameters
    ----------------
      - mu : Mean of the non-projected Gaussian. Shape (n_dim).
      - covariance : Covariance matrix of the non-projected Gaussian. Shape (n_dim x n_dim).
      - c50 : Constant added to the denominator. Scalar
      - y : Points where to evaluate the PDF. Shape (n_points x n_dim).

    Returns
    ----------------
      PDF evaluated at y. Shape (n_points).
    """
    log_pdf = log_pdf(mu, covariance, c50, y)
    pdf = torch.exp(log_pdf)
    return pdf


def prnorm_c50_log_pdf(mu, covariance, c50, y):
    """
    Compute the log probability density function of projected
    normal Y = X/(X'X + c50)^0.5 at points y.

    Parameters
    ----------------
      - mu : Mean of the non-projected Gaussian. Shape (n_dim).
      - covariance : Covariance matrix of the non-projected Gaussian. Shape (n_dim x n_dim).
      - c50 : Constant added to the denominator. Scalar
      - y : Points where to evaluate the PDF. Shape (n_points x n_dim).

    Returns
    ----------------
      Log-PDF evaluated at y. Shape (n_points).
    """
    # Verify that c50 is positive
    if c50 <= 0:
        raise ValueError("c50 must be a positive scalar value.")
    # Invert the projection
    X = _invert_projection(y, c50)
    # Compute the PDF under the normal distribution
    normal_dist = mvn.MultivariateNormal(loc=mu, covariance_matrix=covariance)
    log_pdf = normal_dist.log_prob(X)
    # Compute the jacobian of the inverse projection
    J_log_det = _invert_projection_log_det(y, c50)
    # Compute the PDF
    log_pdf = log_pdf + J_log_det
    return log_pdf


def _invert_projection(y, c50):
    """
    Invert the function projection f(X) = X/(X'X + c50)^0.5

    Parameters
    ----------------
      - y : Observed points in the ball. (n_points x n_dim)
      - c50 : Constant added to the denominator. Scalar

    Returns
    ----------------
      Pre-projection points. (n_points x n_dim)
    """
    scaling = torch.sqrt(c50 / (1 - torch.sum(y**2, dim=-1)))
    X = torch.einsum("...d,...->...d", y, scaling)
    return X


def _invert_projection_jacobian_matrix(y, c50):
    """
    Compute the jacobian matrix of the inverse projection.

    Parameters 
    ----------------
      - y : Observed points in the ball. (n_points x n_dim)
      - c50 : Constant added to the denominator. Scalar

    Returns
    ----------------
      Jacobian matrix of the inverse projection. (n_points x n_dim x n_dim)
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


def _invert_projection_det(y, c50):
    """
    Compute the determinant of the jacobian matrix for the transformation
    Y = X/(X'X + c50)^0.5 at each point y.

    Parameters
    ----------------
      - y : Observed points in the ball. (n_points x n_dim)
      - c50 : Constant added to the denominator. Scalar

    Returns
    ----------------
      Determinant of the jacobian matrix. (n_points)
    """
    log_det = _invert_projection_log_det(y, c50)
    det = torch.exp(log_det)
    return det


def _invert_projection_log_det(y, c50):
    """
    Compute the log determinant of the jacobian matrix for the
    transformation Y = X/(X'X + c50)^0.5 at each point y.

    Note: Uses the matrix determinant lemman that states that
    det(I + uv') = 1 + v'u and det(cA) = c^n det(A) for a scalar c and
    matrix A.

    Parameters
    ----------------
      - y : Observed points in the ball. (n_points x n_dim)
      - c50 : Constant added to the denominator. Scalar

    Returns
    ----------------
      - det : Determinant of the jacobian matrix. (n_points)
    """
    n_dim = y.shape[-1]
    y_sq_norm = torch.sum(y**2, dim=-1)
    scalar = torch.sqrt(c50 / (1 - y_sq_norm))  # Scalar from Jacobian matrix formula
    det_1 = 1 + y_sq_norm / (1 - y_sq_norm)  # Matrix determinant lemma
    det = n_dim * torch.log(scalar) + torch.log(
        det_1
    )  # Scalar multiplication determinant property
    return det
