"""Test the Taylor approximation to ellipse projected normal moments."""
import pytest
import torch

import projnormal._utils._matrix_checks as checks
import projnormal.formulas.projected_normal_B as pnb_formulas
import projnormal.param_sampling as par_samp


@pytest.fixture(scope="function")
def sample_parameters(n_dim, sigma):
    """Fixture to generate Gaussian parameters for tests."""
    # Initialize the mean of the gaussian
    # Parameters of distribution
    mean_x = par_samp.make_mean(n_dim=n_dim)
    covariance_x = par_samp.make_spdm(n_dim=n_dim) * sigma**2
    B = torch.diag(torch.rand(n_dim) + 0.5)

    return {
        "mean_x": mean_x,
        "covariance_x": covariance_x,
        "B": B,
    }

@pytest.mark.parametrize("n_dim", [5])
@pytest.mark.parametrize("sigma", [0.2])
def test_taylor_moments(sample_parameters):
    """Test the Taylor approximation to ellipse projected normal moments."""
    n_samples = 500000

    # Unpack parameters
    mean_x = sample_parameters["mean_x"]
    covariance_x = sample_parameters["covariance_x"]
    B = sample_parameters["B"]

    # Get taylor approximation moments
    gamma_taylor = pnb_formulas.mean(
      mean_x=mean_x, covariance_x=covariance_x, B=B
    )
    sm_taylor = pnb_formulas.second_moment(
        mean_x=mean_x, covariance_x=covariance_x, B=B
    )

    # Check that the means are close
    assert checks.is_symmetric(sm_taylor)
    assert checks.is_positive_definite(sm_taylor)

    moments = pnb_formulas.empirical_moments(
        mean_x=mean_x, covariance_x=covariance_x,
        B=B, n_samples=n_samples
    )
    gamma_emp = moments["mean"]
    sm_emp = moments["second_moment"]

    assert torch.allclose(gamma_taylor, gamma_emp, atol=1e-2)
    assert torch.allclose(sm_taylor, sm_emp, atol=1e-2)
