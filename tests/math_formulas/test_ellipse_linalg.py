""" Test the formulas of the ellipse_linalg module."""
import pytest
import torch
import projnormal as pn


def test_spd_sqrt():
    """Test the square root of a SPD matrix."""
    n_dim = 8
    B = pn.param_sampling.make_spdm(n_dim=n_dim)
    sqrt_B, sqrt_B_inv = pn.ellipse_linalg.spd_sqrt(B)

    assert torch.allclose(sqrt_B @ sqrt_B, B, atol=1e-5), \
        "SPD square root is not correct."
    assert torch.allclose(sqrt_B @ sqrt_B_inv, torch.eye(n_dim), atol=1e-5), \
        "SPD square root inverse is not correct."
    assert pn.matrix_checks.is_symmetric(sqrt_B), \
        "SPD square root is not symmetric."
    assert pn.matrix_checks.is_symmetric(sqrt_B_inv), \
        "SPD square root inverse is not symmetric."

