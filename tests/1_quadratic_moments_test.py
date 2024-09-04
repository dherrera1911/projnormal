##################
#
# TESTS FOR THE FUNCTIONS THAT COMPUTE MOMENTS
# OF QUADRATIC FORMS OF GAUSSIAN RANDOM VARIABLES
#
##################

import pytest
import torch
import projected_normal.qf_func as qf
from utility_functions import make_spdm, make_mu, make_covariance

@pytest.fixture(scope="function")
def quadratic_moments_data(n_dim, sigma):
    # Extract parameters from the request
    n_samples = 500000
    #tolerance = 1e-2
    tolerance = 0.02

    # Parameters of distribution
    mu = make_mu(n_dim=n_dim, mu_type='sin')
    covariance = make_covariance(n_dim=n_dim,
                                 cov_scale=sigma,
                                 cov_type='random')
    M1 = make_spdm(n_dim=n_dim)
    M2 = make_spdm(n_dim=n_dim)

    # Get empirical estimates
    momentsE = qf.empirical_moments_quadratic_form(mu, covariance, M1, n_samples=n_samples)
    mean_empirical = momentsE['mean']
    var_empirical = momentsE['var']
    cov_empirical = qf.empirical_covariance_quadratic_form(
        mu, covariance, M1, M2, n_samples=n_samples
    )

    # Return all relevant data
    return {
        'mu': mu.double(),
        'covariance': covariance.double(),
        'M1': M1.double(),
        'M2': M2.double(),
        'mean_empirical': mean_empirical,
        'var_empirical': var_empirical,
        'cov_empirical': cov_empirical,
        'tolerance': tolerance
    }

@pytest.mark.parametrize('n_dim', [2, 3, 5, 50])
@pytest.mark.parametrize('sigma', [0.01, 0.1, 1, 3])
def test_quadratic_moments(quadratic_moments_data, n_dim, sigma):
    # Unpack distribution parameters
    mu = quadratic_moments_data['mu']
    covariance = quadratic_moments_data['covariance']
    M1 = quadratic_moments_data['M1']
    M2 = quadratic_moments_data['M2']
    tolerance = quadratic_moments_data['tolerance']

    # Unpack empirical moments
    mean_empirical = quadratic_moments_data['mean_empirical']
    var_empirical = quadratic_moments_data['var_empirical']
    cov_empirical = quadratic_moments_data['cov_empirical']

    # Get theoretical moments
    # Mean
    mean_analytic = qf.quadratic_form_mean(mu, covariance, M1)
    var_analytic = qf.quadratic_form_var(mu, covariance, M1)
    cov_analytic = qf.quadratic_form_cov(mu, covariance, M1, M2)

    # Compute the relative errors
    mean_error = (mean_empirical - mean_analytic) / mean_analytic
    var_error = (var_empirical - var_analytic) / var_analytic
    cov_error = (cov_empirical - cov_analytic) / cov_analytic

    assert mean_error < tolerance, f"Mean error is too large: {mean_error}"
    assert var_error < tolerance, f"Variance error is too large: {var_error}" 
    assert cov_error < tolerance, f"Covariance error is too large: {cov_error}"


@pytest.mark.parametrize('n_dim', [2, 3, 5, 50])
@pytest.mark.parametrize('sigma', [0.01, 0.1, 1, 3])
def test_diagonal_quadratic_moments(quadratic_moments_data, n_dim, sigma):
    # Unpack distribution parameters
    mu = quadratic_moments_data['mu']
    covariance = quadratic_moments_data['covariance']
    M1 = torch.diag( torch.diagonal(quadratic_moments_data['M1']))
    M2 = torch.diag( torch.diagonal(quadratic_moments_data['M2']))
    tolerance = 1e-5

    # Get moments with general functions
    # Mean
    mean_analytic = qf.quadratic_form_mean(mu, covariance, M1)
    var_analytic = qf.quadratic_form_var(mu, covariance, M1)

    # Get moments with diagonal functions
    M_diagonal = torch.diagonal(M1)
    mean_diagonal = qf.quadratic_form_mean_diagonal(mu,
                                                    covariance,
                                                    M_diagonal=M_diagonal)
    var_diagonal = qf.quadratic_form_var_diagonal(mu,
                                                  covariance,
                                                  M_diagonal=M_diagonal)

    # Compute the absolute errors
    mean_error = (mean_diagonal - mean_analytic)
    var_error = (var_diagonal - var_analytic)

    assert mean_error < tolerance, f"Mean error is too large: {mean_error}"
    assert var_error < tolerance, f"Variance error is too large: {var_error}"

