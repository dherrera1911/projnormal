"""Test the class implementing the ellipse parametrization."""
import pytest
import torch
import projnormal.param_sampling as par_samp
import projnormal.matrix_checks as checks
from projnormal.models._ellipsoid import Ellipsoid, EllipsoidFixed


@pytest.fixture(scope="function")
def sample_B_sqrt(n_dim, n_dirs):
    """ Fixture to generate matrix B."""
    eigvecs = par_samp.make_ortho_vectors(n_dim=n_dim, n_vec=n_dirs)
    eigvals = torch.rand(n_dirs) + 1
    diag_val = torch.rand(1) + 0.5

    B_sqrt = torch.eye(n_dim) * diag_val + torch.einsum(
      "ij,i,im->jm", eigvecs, eigvals, eigvecs
    )

    return {
      "B_sqrt": B_sqrt,
      "eigvecs": eigvecs,
      "eigvals": eigvals,
      "diag_val": diag_val,
    }


@pytest.mark.parametrize("n_dim", [3, 5])
@pytest.mark.parametrize("n_dirs", [1, 2])
def test_ellipse(n_dim, n_dirs, sample_B_sqrt):

    B_sqrt = sample_B_sqrt["B_sqrt"]
    B = B_sqrt @ B_sqrt
    eigvecs = sample_B_sqrt["eigvecs"]
    eigvals = sample_B_sqrt["eigvals"]
    diag_val = sample_B_sqrt["diag_val"]

    # Initialize ellipsoid
    ellipse = Ellipsoid(
      n_dim=n_dim,
      n_dirs=n_dirs,
      sqrt_coefs=eigvals,
      sqrt_vecs=eigvecs,
      sqrt_diag=diag_val
    )

    # Check that matrix B is correct
    with torch.no_grad():
        B_class = ellipse.get_B()
        B_sqrt_class = ellipse.get_B_sqrt()
        B_sqrt_inv = ellipse.get_B_sqrt_inv()
        B_logdet = ellipse.get_B_logdet()

    assert torch.allclose(B_class, B, atol=1e-5), \
        "Matrix B is incorrect"
    # Check that classerated matrices are correct
    assert torch.allclose(B_sqrt_class, B_sqrt, atol=1e-5), \
        "B_sqrt is incorrect"
    assert torch.allclose(B_sqrt_class @ B_sqrt_inv, torch.eye(n_dim), atol=1e-5), \
        "B_sqrt_inv is incorrect"
    assert torch.allclose(B_logdet, torch.logdet(B)), \
        "B_logdet is incorrect"


@pytest.mark.parametrize("n_dim", [3, 5, 8])
def test_ellipse_fixed(n_dim):
    B = par_samp.make_spdm(n_dim=n_dim)

    # Initialize ellipsoid
    ellipse = EllipsoidFixed(B=B)

    assert torch.allclose(B, ellipse.B, atol=1e-5), \
        "B is incorrect"

    B_sqrt = ellipse.get_B_sqrt()
    B_sqrt_inv = ellipse.get_B_sqrt_inv()
    B_logdet = ellipse.get_B_logdet()

    assert torch.allclose(B, B_sqrt @ B_sqrt, atol=1e-5), \
        "B_sqrt is incorrect"
    assert torch.allclose(B @ B_sqrt_inv @ B_sqrt_inv, torch.eye(n_dim), atol=1e-5), \
        "B_sqrt_inv is incorrect"
    assert torch.allclose(B_logdet, torch.logdet(B)), \
        "B_logdet is incorrect"

    # Assign new B
    B_new = par_samp.make_spdm(n_dim=n_dim)
    ellipse.B = B_new

    assert torch.allclose(B_new, ellipse.B, atol=1e-5), \
        "B_sqrt is incorrect"

    B_sqrt = ellipse.get_B_sqrt()
    B_sqrt_inv = ellipse.get_B_sqrt_inv()

    assert torch.allclose(B_new, B_sqrt @ B_sqrt, atol=1e-5), \
        "B_sqrt is incorrect"
    assert torch.allclose(B_new @ B_sqrt_inv @ B_sqrt_inv, torch.eye(n_dim), atol=1e-5), \
        "B_sqrt_inv is incorrect"
