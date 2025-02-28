"""Test the Taylor approximation to projected normal moments"""
import pytest
import torch
import projected_normal.distribution.c50 as pnc
import projected_normal.param_sampling as par_samp
import projected_normal.matrix_checks as checks

torch.manual_seed(0)
TOLERANCE = 0.04


def relative_error(x, y):
    """Compute the relative error between two tensors."""
    return torch.norm(x - y) * 2 / (torch.norm(y) + torch.norm(x))


# Instantiate parameters
@pytest.fixture(scope="function")
def gaussian_parameters(n_dim, mean_type, eigvals, eigvecs, sigma, c50):
    """ Fixture to generate Gaussian parameters for tests."""

    # Initialize the mean of the gaussian
    # Parameters of distribution
    mean_x = par_samp.make_mean(
      n_dim=n_dim, shape=mean_type
    )
    covariance_x = par_samp.make_spdm(
      n_dim=n_dim, eigvals=eigvals, eigvecs=eigvecs
    ) * sigma**2

    return {
        "mean_x": mean_x,
        "covariance_x": covariance_x,
        "c50": c50,
    }


######### CHECK THAT THE OUTPUTS ARE NUMERICALLY VALID ############
@pytest.mark.parametrize("n_dim", [3, 5, 10])
@pytest.mark.parametrize("mean_type", ["sin", "sparse"])
@pytest.mark.parametrize("eigvals", ["uniform", "exponential"])
@pytest.mark.parametrize("eigvecs", ["random", "identity"])
@pytest.mark.parametrize("sigma", [0.1])
@pytest.mark.parametrize("c50", [0, 0.1])
def test_taylor_stability(gaussian_parameters):
    # Unpack parameters
    mean_x = gaussian_parameters["mean_x"]
    covariance_x = gaussian_parameters["covariance_x"]
    c50 = gaussian_parameters["c50"]

    # Get taylor approximation moments
    gamma_taylor = pnc.moments.mean(
      mean_x=mean_x, covariance_x=covariance_x, c50=c50
    )
    sm_taylor = pnc.moments.second_moment(
        mean_x=mean_x, covariance_x=covariance_x, c50=c50
    )

    # Check that outputs are not nan, and matrices are as expected
    assert not torch.isnan(
        gamma_taylor
    ).any(), "Taylor approximation of the mean is nan"
    assert not torch.isnan(
        sm_taylor
    ).any(), "Taylor approximation of the second moment is nan"
    assert checks.is_symmetric(
        sm_taylor
    ), "Taylor approximation of the covariance is not symmetric"
    assert checks.is_positive_definite(
        sm_taylor
    ), "Taylor approximation of the covariance is not positive definite"


######### COMPARE APPROXIMATION AND EMPIRICAL ############
@pytest.mark.parametrize("n_dim", [2, 3, 10])
@pytest.mark.parametrize("mean_type", ["sin", "sparse"])
@pytest.mark.parametrize("eigvals", ["uniform", "exponential"])
@pytest.mark.parametrize("eigvecs", ["random", "identity"])
@pytest.mark.parametrize("sigma", [0.01, 0.1])
@pytest.mark.parametrize("c50", [0, 1])
@pytest.mark.parametrize("n_samples", [200000])
def test_taylor_vs_empirical(gaussian_parameters, n_samples):
    # Unpack parameters
    mean_x = gaussian_parameters["mean_x"]
    covariance_x = gaussian_parameters["covariance_x"]
    c50 = gaussian_parameters["c50"]

    # Get taylor approximation moments
    gamma_taylor = pnc.moments.mean(
      mean_x=mean_x, covariance_x=covariance_x, c50=c50
    )
    sm_taylor = pnc.moments.second_moment(
        mean_x=mean_x, covariance_x=covariance_x, c50=c50
    )

    # Get empirical moments
    moments_empirical = pnc.sampling.empirical_moments(
        mean_x, covariance_x, n_samples=n_samples, c50=c50
    )
    gamma_empirical = moments_empirical["mean"]
    sm_empirical = moments_empirical["second_moment"]

    # Get Taylor approximation moments
    gamma_taylor = pnc.moments.mean(
        mean_x=mean_x, covariance_x=covariance_x, c50=c50
    )
    sm_taylor = pnc.moments.second_moment(
        mean_x=mean_x, covariance_x=covariance_x, c50=c50
    )

    # Relative error
    gamma_error = relative_error(gamma_empirical, gamma_taylor)
    sm_error = relative_error(sm_empirical, sm_taylor)

    # Check if the error is small
    assert (
        gamma_error < TOLERANCE
    ), f"Taylor expected value approximation has large error: {gamma_error}"
    assert (
        sm_error < TOLERANCE
    ), f"Taylor second moment approximation has large error: {sm_error}"
