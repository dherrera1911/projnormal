##################
#
# TESTS FOR THE FUNCTIONS THAT COMPUTE MOMENTS
# OF THE PROJECTED NORMAL DISTRIBUTION
#
##################

import time
import pytest
import torch
import projected_normal.prnorm_func as pnf
import projected_normal.auxiliary as pna
from utility_functions import make_mu, make_covariance


# Instantiate parameters
@pytest.fixture(scope="function")
def gaussian_parameters(n_dim, mean_type, cov_type, sigma, B_type, c50):

    # tolerance = 1e-2
    tolerance = 0.7

    # Initialize the mean of the gaussian
    mu = make_mu(n_dim=n_dim, mu_type=mean_type)

    # Initialize the covariance of the gaussian
    covariance = make_covariance(n_dim=n_dim, cov_scale=sigma, cov_type=cov_type)

    # Initialize the matrix B
    if B_type == "isotropic":
        B_diagonal = torch.ones(n_dim) * torch.rand(1)
    elif B_type == "random":
        B_diagonal = torch.rand(n_dim)
    elif B_type == "exponential":
        i = torch.arange(n_dim)
        B_diagonal = torch.exp(-i.float() * 3 / n_dim)
    elif B_type == "sparse":
        B_diagonal = torch.ones(n_dim) * 0.01
        B_diagonal[::3] = 1
    elif B_type == "identity":
        B_diagonal = torch.ones(n_dim)

    return {
        "mu": mu,
        "covariance": covariance,
        "B_diagonal": B_diagonal,
        "c50": c50,
        "tolerance": tolerance,
    }


######### CHECK THAT THE OUTPUTS ARE NUMERICALLY VALID ############


@pytest.mark.parametrize("n_dim", [3, 5, 20, 50])
@pytest.mark.parametrize("mean_type", ["sin", "sparse"])
@pytest.mark.parametrize("cov_type", ["random", "diagonal"])
@pytest.mark.parametrize("B_type", ["isotropic", "exponential", "sparse"])
@pytest.mark.parametrize("sigma", [0.01])
@pytest.mark.parametrize("c50", [0, 1])
def test_taylor_approximation(gaussian_parameters):
    # Unpack parameters
    mu = gaussian_parameters["mu"]
    covariance = gaussian_parameters["covariance"]
    B_diagonal = gaussian_parameters["B_diagonal"]
    c50 = gaussian_parameters["c50"]

    # Get taylor approximation moments
    gamma_taylor = pnf.prnorm_mean_taylor(
        mu=mu, covariance=covariance, B_diagonal=B_diagonal, c50=c50
    )

    sm_taylor = pnf.prnorm_sm_taylor(
        mu=mu, covariance=covariance, B_diagonal=B_diagonal, c50=c50
    )

    # Check that outputs are not nan
    assert not torch.isnan(
        gamma_taylor
    ).any(), "Taylor approximation of the mean is nan"
    assert not torch.isnan(
        sm_taylor
    ).any(), "Taylor approximation of the second moment is nan"
    assert pna.is_symmetric(
        sm_taylor
    ), "Taylor approximation of the covariance is not symmetric"
    assert pna.is_positive_definite(
        sm_taylor
    ), "Taylor approximation of the covariance is not positive definite"


######### COMPARE APPROXIMATION AND EMPIRICAL ############


@pytest.mark.parametrize("n_dim", [2, 3, 10, 25])
@pytest.mark.parametrize("mean_type", ["sin", "sparse"])
@pytest.mark.parametrize("cov_type", ["random", "diagonal"])
@pytest.mark.parametrize("B_type", ["identity", "exponential"])
@pytest.mark.parametrize("sigma", [0.01, 0.1, 1])
@pytest.mark.parametrize("c50", [0, 1])
@pytest.mark.parametrize("n_samples", [200000])
def test_taylor_vs_empirical(gaussian_parameters, n_samples):
    # Unpack parameters
    mu = gaussian_parameters["mu"]
    covariance = gaussian_parameters["covariance"]
    B_diagonal = gaussian_parameters["B_diagonal"]
    c50 = gaussian_parameters["c50"]
    tolerance = gaussian_parameters["tolerance"]
    B = torch.diag(B_diagonal)

    # Get empirical moments
    moments_empirical = pnf.empirical_moments_prnorm(
        mu, covariance, n_samples=n_samples, B=B, c50=c50
    )

    gamma_empirical = moments_empirical["gamma"]
    sm_empirical = moments_empirical["second_moment"]

    # Get taylor approximation moments
    gamma_taylor = pnf.prnorm_mean_taylor(
        mu=mu, covariance=covariance, B_diagonal=B_diagonal, c50=c50
    )

    sm_taylor = pnf.prnorm_sm_taylor(
        mu=mu, covariance=covariance, B_diagonal=B_diagonal, c50=c50
    )

    # Compare moments
    # Absolute error
    # gamma_error = torch.norm(gamma_empirical - gamma_taylor)
    # sm_error = torch.norm(sm_empirical - sm_taylor)
    # Relative error
    gamma_error = torch.norm(gamma_empirical - gamma_taylor) / torch.norm(
        gamma_empirical
    )
    sm_error = torch.norm(sm_empirical - sm_taylor) / torch.norm(sm_empirical)

    # Check if the error is small
    assert (
        gamma_error < tolerance
    ), f"Taylor expected value approximation has large error: {gamma_error}"
    assert (
        sm_error < tolerance
    ), f"Taylor second moment approximation has large error: {sm_error}"
