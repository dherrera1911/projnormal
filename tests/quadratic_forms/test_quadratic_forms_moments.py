"""Tests for quadratic moments formulas of quadratic forms of random variables."""
import pytest
import torch

import projnormal.param_sampling as par_samp
import projnormal.quadratic_forms as qf


@pytest.fixture(scope="function")
def quadratic_moments_data(n_dim, sigma):
    """Generate parameters to test, and obtain empirical estimates to compare with."""
    # Extract parameters from the request
    n_samples = 100000
    tolerance = 0.05

    # Parameters of distribution
    mean_x = par_samp.make_mean(
      n_dim=n_dim, shape='sin'
    )
    covariance_x = par_samp.make_spdm(n_dim=n_dim) * sigma**2
    M1 = par_samp.make_spdm(n_dim=n_dim)
    M2 = par_samp.make_spdm(n_dim=n_dim)

    # Get empirical estimates
    momentsE = qf.sampling.empirical_moments(
      mean_x, covariance_x, M1, n_samples=n_samples
    )
    mean_empirical = momentsE['mean']
    var_empirical = momentsE['var']
    cov_empirical = qf.sampling.empirical_covariance(
        mean_x, covariance_x, M1, M2, n_samples=n_samples
    )

    # Return all relevant data
    return {
        'mean_x': mean_x.double(),
        'covariance_x': covariance_x.double(),
        'M1': M1.double(),
        'M2': M2.double(),
        'mean_empirical': mean_empirical,
        'var_empirical': var_empirical,
        'cov_empirical': cov_empirical,
        'tolerance': tolerance
    }

@pytest.mark.parametrize('n_dim', [2, 3, 10])
@pytest.mark.parametrize('sigma', [0.1, 1])
def test_qf_moments(quadratic_moments_data, n_dim, sigma):
    """Test the quadratic moments formulas."""
    # Unpack distribution parameters
    mean_x = quadratic_moments_data['mean_x']
    covariance_x = quadratic_moments_data['covariance_x']
    M1 = quadratic_moments_data['M1']
    M2 = quadratic_moments_data['M2']
    tolerance = quadratic_moments_data['tolerance']

    # Unpack empirical moments
    mean_empirical = quadratic_moments_data['mean_empirical']
    var_empirical = quadratic_moments_data['var_empirical']
    cov_empirical = quadratic_moments_data['cov_empirical']

    # Get theoretical moments
    # Mean
    mean_analytic = qf.moments.mean(mean_x, covariance_x, M1)
    var_analytic = qf.moments.variance(mean_x, covariance_x, M1)
    cov_analytic = qf.moments.qf_covariance(mean_x, covariance_x, M1, M2)

    # Compute the relative errors
    mean_error = (mean_empirical - mean_analytic) / mean_analytic
    var_error = (var_empirical - var_analytic) / var_analytic
    cov_error = (cov_empirical - cov_analytic) / cov_analytic

    assert mean_error < tolerance, f"Mean error is too large: {mean_error}"
    assert var_error < tolerance, f"Variance error is too large: {var_error}"
    assert cov_error < tolerance, f"Covariance error is too large: {cov_error}"


@pytest.mark.parametrize('n_dim', [2, 3, 10])
@pytest.mark.parametrize('sigma', [0.1, 1])
def test_diagonal_qf_moments(quadratic_moments_data, n_dim, sigma):
    """Test that functions also work with diagonal matrices M."""
    # Unpack distribution parameters
    mean_x = quadratic_moments_data['mean_x']
    covariance_x = quadratic_moments_data['covariance_x']
    M1 = torch.diag(torch.diagonal(quadratic_moments_data['M1']))
    tolerance = 1e-5

    # Get moments with full matrix M1
    # Mean
    mean_analytic = qf.moments.mean(mean_x, covariance_x, M1)
    var_analytic = qf.moments.variance(mean_x, covariance_x, M1)

    # Get moments with diagonal functions
    M_diagonal = torch.diagonal(M1)
    mean_diagonal = qf.moments.mean(mean_x, covariance_x, M_diagonal)
    var_diagonal = qf.moments.variance(mean_x, covariance_x, M_diagonal)

    # Compute the absolute errors
    mean_error = (mean_diagonal - mean_analytic)
    var_error = (var_diagonal - var_analytic)

    assert mean_error < tolerance, f"Mean error is too large: {mean_error}"
    assert var_error < tolerance, f"Variance error is too large: {var_error}"

