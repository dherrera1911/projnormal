"""Probability density function (PDF) for the general projected normal distribution with an additive constant const in the denominator ."""
import torch
import torch.distributions.multivariate_normal as mvn

__all__ = ["pdf", "log_pdf"]


def __dir__():
    return __all__


def pdf(mean_x, covariance_x, const, y):
    r"""
    Compute the pdf at points y for the distribution of the variable
    :math:`y = x/\sqrt{x^T x + c}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`c` is a positive constant.


    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
        Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      y : ``torch.Tensor``
          Points where to evaluate the PDF. Shape is ``(n_points, n_dim)``.

      const : ``torch.Tensor``
          Constant added to the denominator. Must be positive. Shape is ``()``.

    Returns
    -------
      ``torch.Tensor``
          PDF evaluated at each y. Shape is ``(n_points,)``.
    """
    lpdf = log_pdf(mean_x, covariance_x, const, y)
    return torch.exp(lpdf)


def log_pdf(mean_x, covariance_x, const, y):
    r"""
    Compute the log-pdf at points y for the distribution of the variable
    :math:`y = x/\sqrt{x^T x + c}`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    and :math:`c` is a positive constant.


    Parameters
    ----------
      mean_x : ``torch.Tensor``
          Mean of `x`. Shape is ``(n_dim,)``.

      covariance_x : ``torch.Tensor``
        Covariance of `x`. Shape is ``(n_dim, n_dim)``.

      y : ``torch.Tensor``
          Points where to evaluate the PDF. Shape is ``(n_points, n_dim)``.

      const : ``torch.Tensor``
          Constant added to the denominator. Must be positive. Shape is ``()``.

    Returns
    -------
      ``torch.Tensor``
          Log-PDF evaluated at each y. Shape is ``(n_points,)``.
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
    Invert the function projection f(X) = X/(X'X + const)^0.5.

    Parameters
    ----------
      y : torch.Tensor, shape (n_points, n_dim)
          Observed points in the ball.

      const : torch.Tensor, shape ()
          Constant added to the denominator.

    Returns
    -------
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
    ----------
      y : torch.Tensor, shape (n_points, n_dim)
          Observed points in the ball.

      const : torch.Tensor, shape ()
          Constant added to the denominator.

    Returns
    -------
      torch.Tensor, shape (n_points, n_dim, n_dim)
          Jacobian matrix of the inverse projection.
    """
    n_dim = y.shape[-1]
    y_sq_norm = torch.sum(y**2, dim=-1)
    J_multiplier = torch.sqrt(const / (1 - y_sq_norm))
    J_matrix = torch.einsum("...d,...e->...de", y, y / (1 - y_sq_norm.view(-1, 1)))
    # Add identity to the diagonal
    is_batch = y.dim() == 2
    if is_batch:
        # Make identity matrix for each batch
        n_batch = y.shape[0]
        eye = torch.eye(n_dim, device=y.device).unsqueeze(0).expand(n_batch, -1, -1)
        J_matrix += eye
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
    ----------
      y : torch.Tensor, shape (n_points, n_dim)
          Observed points in the ball.

      const : torch.Tensor, shape ()
          Constant added to the denominator.

    Returns
    -------
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
    ----------
      y : torch.Tensor, shape (n_points, n_dim)
          Observed points in the ball.

      const : torch.Tensor, shape ()
          Constant added to the denominator.

    Returns
    -------
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
