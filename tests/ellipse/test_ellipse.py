"""Test the class implementing the ellipse parametrization."""
import pytest
import torch
import projnormal.param_sampling as par_samp
import projnormal.matrix_checks as checks
from projnormal.models._ellipsoid import Ellipsoid


@pytest.fixture(scope="function")
def sample_B(n_dim, n_dirs):
    """ Fixture to generate matrix B."""
    eigvecs = par_samp.make_ortho_vectors(n_dim=n_dim, n_vec=n_dirs)
    eigvals = torch.rand(n_dirs) + 1
    rad_sq = torch.rand(1) + 0.5

    B = torch.eye(n_dim) * rad_sq + torch.einsum(
      "ij,i,im->jm", eigvecs, (eigvals - rad_sq), eigvecs
    )

    return {
      "B": B,
      "eigvecs": eigvecs,
      "eigvals": eigvals,
      "rad_sq": rad_sq
    }

@pytest.mark.parametrize("n_dim", [3, 5])
@pytest.mark.parametrize("n_dirs", [1, 2])
def test_ellipse(n_dim, n_dirs, sample_B):

    B = sample_B["B"]
    eigvecs = sample_B["eigvecs"]
    eigvals = sample_B["eigvals"]
    rad_sq = sample_B["rad_sq"]

    # Initialize ellipsoid
    ellipse = Ellipsoid(
      n_dim=n_dim, n_dirs=n_dirs, rad_sq=rad_sq
    )

    ellipse.eigvecs = eigvecs
    ellipse.eigvals = eigvals

    # Check that matrix B is correct
    with torch.no_grad():
        B_gen = ellipse.get_B()
        B_eigvals = torch.sort(
          torch.linalg.eigvalsh(B_gen)
        ).values
        B_sqrt = ellipse.get_B_sqrt()
        B_sqrt_inv = ellipse.get_B_sqrt_inv()
        B_logdet = ellipse.get_B_logdet()

    eigval_list = torch.sort(
      torch.cat((torch.ones(n_dim-n_dirs) * rad_sq, eigvals))
    ).values

    assert torch.allclose(B_gen, B, atol=1e-5), "Matrix B has incorrect eigenvalues"
    assert torch.allclose(B_gen, B), "Matrix B is incorrect"

    # Check that generated matrices are correct
    assert torch.allclose(B_gen, B_sqrt @ B_sqrt, atol=1e-5), "B_sqrt is incorrect"
    assert torch.allclose(B_sqrt @ B_sqrt_inv, torch.eye(n_dim), atol=1e-5), "B_sqrt_inv is incorrect"
    assert torch.allclose(B_logdet, torch.logdet(B)), "B_logdet is incorrect"

