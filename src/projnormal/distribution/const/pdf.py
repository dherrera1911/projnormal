"""Probability density function (PDF) for the general projected normal distribution with an additive constant const in the denominator ."""
import torch
import torch.distributions.multivariate_normal as mvn


__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mean_x, covariance_x, const, y):
    """
    Compute the probability density function at points y for the variable
    Y = X/(X'X+const)^0.5 where X ~ N(mean_x, covariance_x). Y has a general projected
    normal distribution with an extra additive constant const in the denominator.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      var_x : torch.Tensor, shape ()
          Variance of X elements.

      const : torch.Tensor, shape ()
          Constant added to the denominator.

      y : torch.Tensor, shape (n_points, n_dim)
          Points where to evaluate the PDF.

    Returns
    ----------------
      torch.Tensor, shape (n_points)
          PDF evaluated at each y.
    """
    lpdf = log_pdf(mean_x, covariance_x, const, y)
    pdf = torch.exp(lpdf)
    return pdf


def log_pdf(mean_x, covariance_x, const, y):
    """
    Compute the log probability density function at points y for the variable
    Y = X/(X'X+const)^0.5 where X ~ N(mean_x, covariance_x). Y has a general projected
    normal distribution with an extra additive constant const in the denominator.

    Parameters
    ----------------
      mean_x : torch.Tensor, shape (n_dim,)
          Mean of X.

      covariance_x : torch.Tensor, shape (n_dim, n_dim)
        Covariance matrix of X elements.

      const : torch.Tensor, shape ()
          Constant added to the denominator.

      y : torch.Tensor, shape (n_points, n_dim)
          Points where to evaluate the PDF.

    Returns
    ----------------
      torch.Tensor, shape (n_points)
          Log-PDF evaluated at each y.
    """
    # Verify that const is positive
    if const <= 0:
        raise ValueError("const must be a positive scalar value.")
    # Invert the projection
    X = _invert_projection(y, const)
    # Compute the PDF under the normal distribution
    normal_dist = mvn.MultivariateNormal(loc=mean_x, covariance_matrix=covariance_x)
    lpdf = normal_dist.log_prob(X)
    # Compute the jacobian of the inverse projection
    J_log_det = _invert_projection_log_det(y, const)
    # Compute the PDF
    lpdf = lpdf + J_log_det
    return lpdf


def _invert_projection(y, const):
    """
    Invert the function projection f(X) = X/(X'X + const)^0.5

    Parameters
    ----------------
      y : torch.Tensor, shape (n_points, n_dim)
          Observed points in the ball.

      const : torch.Tensor, shape ()
          Constant added to the denominator.

    Returns
    ----------------
      torch.Tensor, shape (n_points, n_dim)
          Pre-projection points.
    """
    scaling = torch.sqrt(const / (1 - torch.sum(y**2, dim=-1)))
    X = torch.einsum("...d,...->...d", y, scaling)
    return X


def _invert_projection_jacobian_matrix(y, const):
    """
    Compute the Jacobian matrix of the inverse projection.

    Parameters
    ----------------
      y : torch.Tensor, shape (n_points, n_dim)
          Observed points in the ball.

      const : torch.Tensor, shape ()
          Constant added to the denominator.

    Returns
    ----------------
      torch.Tensor, shape (n_points, n_dim, n_dim)
          Jacobian matrix of the inverse projection.
    """
    n_dim = y.shape[-1]
    y_sq_norm = torch.sum(y**2, dim=-1)
    J_multiplier = torch.sqrt(const / (1 - y_sq_norm))
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


def _invert_projection_det(y, const):
    """
    Compute the determinant of the Jacobian matrix for the transformation
    Y = X/(X'X + const)^0.5 at each point y.

    Parameters
    ----------------
      y : torch.Tensor, shape (n_points, n_dim)
          Observed points in the ball.

      const : torch.Tensor, shape ()
          Constant added to the denominator.

    Returns
    ----------------
      torch.Tensor, shape (n_points)
          Determinant of the Jacobian matrix of the inverse projection.
    """
    log_det = _invert_projection_log_det(y, const)
    det = torch.exp(log_det)
    return det


def _invert_projection_log_det(y, const):
    """
    Compute the log determinant of the jacobian matrix for the
    transformation Y = X/(X'X + const)^0.5 at each point y.

    Note: Uses the matrix determinant lemman that states that
    det(I + uv') = 1 + v'u and det(cA) = c^n det(A) for a scalar c and
    matrix A.

    Parameters
    ----------------
      y : torch.Tensor, shape (n_points, n_dim)
          Observed points in the ball.

      const : torch.Tensor, shape ()
          Constant added to the denominator.

    Returns
    ----------------
      torch.Tensor, shape (n_points)
          Log-determinant of the Jacobian matrix of the inverse projection.
    """
    n_dim = y.shape[-1]
    y_sq_norm = torch.sum(y**2, dim=-1)
    scalar = const / (1 - y_sq_norm)  # Scalar from Jacobian matrix formula
    det_1 = 1 + y_sq_norm / (1 - y_sq_norm)  # Matrix determinant lemma
    det = (n_dim / 2) * torch.log(scalar) + torch.log(
        det_1
    )  # Scalar multiplication determinant property
    return det
