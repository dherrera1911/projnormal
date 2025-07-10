"""Test the formulas of the ellipse_linalg module."""
import torch

import projnormal


def test_spd_sqrt():
    """Test the square root of a SPD matrix."""
    n_dim = 8
    B = projnormal.param_sampling.make_spdm(n_dim=n_dim)
    sqrt_B, sqrt_B_inv = projnormal._linalg.spd_sqrt(B, return_inverse=True)

    assert torch.allclose(sqrt_B @ sqrt_B, B, atol=1e-5), \
        "SPD square root is not correct."
    assert torch.allclose(sqrt_B @ sqrt_B_inv, torch.eye(n_dim), atol=1e-5), \
        "SPD square root inverse is not correct."
    assert projnormal._matrix_checks.is_symmetric(sqrt_B), \
        "SPD square root is not symmetric."
    assert projnormal._matrix_checks.is_symmetric(sqrt_B_inv), \
        "SPD square root inverse is not symmetric."

