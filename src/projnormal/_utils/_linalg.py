"""Linear algebra utilities to deal with denominator positive definite matrix B."""
import torch

__all__ = [
  "spd_sqrt",
  "make_B_matrix",
]


def __dir__():
    return __all__


def spd_sqrt(B, return_inverse=False):
    """
    Compute the square root of a symmetric positive definite matrix.
    Optionally return the inverse square root also.

    Computes the symmetric positive definite matrix S such that SS = B.

    Parameters
    ----------
    B : torch.Tensor
        Symmetric positive definite matrices. Shape (n_dim, n_dim).

    return_inverse : bool, optional
        Whether to return the inverse square root of B. Default is True.

    Returns
    -------
    B_sqrt : torch.Tensor
        The square root of B. Shape (n_dim, n_dim).

    B_sqrt_inv : torch.Tensor, optional
        The inverse square root of B. Shape (n_dim, n_dim).

    """
    eigvals, eigvecs = torch.linalg.eigh(B)
    B_sqrt = torch.einsum(
        "ij,j,kj->...ik", eigvecs, torch.sqrt(eigvals), eigvecs
    )
    if return_inverse:
        B_sqrt_inv = torch.einsum(
            "ij,j,kj->...ik", eigvecs, 1 / torch.sqrt(eigvals), eigvecs
        )
        return B_sqrt, B_sqrt_inv
    else:
        return B_sqrt


def make_B_matrix(eigvals, eigvecs, rad_sq):
    """
    For a set of n_eig eigenvalues and eigenvectors, make
    a symmetric positive definite matrix B that has those
    eigenvalues and eigenvectors. The rest of the
    eigenvalues are set to rad_sq.

    Parameters
    ----------
    eigvals : torch.Tensor
        Eigenvalues. Shape (n_eig,).

    eigvecs : torch.Tensor
        Eigenvectors. Shape (n_eig, n_dim).

    rad_sq : float
        The value of the remaining eigenvalues.

    Returns
    -------
    B : torch.Tensor
        Symmetric positive definite matrix. Shape (n_dim, n_dim).

    """
    n_dim = eigvecs.shape[1]
    B = torch.eye(n_dim, dtype=eigvals.dtype, device=eigvals.device) * rad_sq \
        + torch.einsum("ij,i,im->jm", eigvecs, (eigvals - rad_sq), eigvecs)
    return B
