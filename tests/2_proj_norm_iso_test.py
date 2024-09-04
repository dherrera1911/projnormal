##################
#
# TESTS FOR THE FUNCTIONS THAT COMPUTE MOMENTS
# OF THE PROJECTED NORMAL DISTRIBUTION
#
##################

import pytest
import torch
import projected_normal.prnorm_func as pnf
from utility_functions import make_mu, make_covariance

### TEST THAT THE MEAN OF THE PROJECTED NORMAL DISTRIBUTION IS CORRECT

# Fixture to set up data for isotropic noise case
@pytest.fixture(scope='class')
def isotropic_noise_data(request):
    n_dim = request.param['n_dim']
    n_samples = 100000
    sigma = request.param['sigma']
    tolerance = 0.05

    # Parameters of distribution
    mu = make_mu(n_dim=n_dim, mu_type='sin')
    covariance = make_covariance(n_dim=n_dim, cov_scale=sigma, cov_type='isotropic')
    B = torch.eye(n_dim)

    # Get empirical estimates
    moments_empirical = pnf.empirical_moments_prnorm(
        mu, covariance, n_samples=n_samples, B=B
    )

    return {
        'mu': mu,
        'sigma': sigma,
        'gamma_empirical': moments_empirical['gamma'],
        'psi_empirical': moments_empirical['psi'],
        'second_m_empirical': moments_empirical['second_moment'],
        'tolerance': tolerance,
    }


@pytest.mark.parametrize(
    'isotropic_noise_data',
    [
        {'n_dim': 2, 'sigma': 0.1},
        {'n_dim': 2, 'sigma': 0.5},
        {'n_dim': 3, 'sigma': 0.1},
        {'n_dim': 3, 'sigma': 0.5},
        {'n_dim': 3, 'sigma': 1},
        {'n_dim': 10, 'sigma': 0.1},
        {'n_dim': 10, 'sigma': 0.5},
        {'n_dim': 20, 'sigma': 0.1},
        {'n_dim': 20, 'sigma': 0.5},
        {'n_dim': 20, 'sigma': 1},
        {'n_dim': 40, 'sigma': 0.2},
    ],
    indirect=True,
)
class TestIsotropicNoiseCase:

    def test_mean_error(self, isotropic_noise_data):
        # unpack data
        mu = isotropic_noise_data['mu']
        sigma = isotropic_noise_data['sigma']
        gamma_empirical = isotropic_noise_data['gamma_empirical']

        # Get analytical estimate
        gamma_analytic = pnf.prnorm_mean_iso(mu=mu, sigma=sigma)

        # Check error
        gamma_error = torch.norm(gamma_empirical - gamma_analytic) / torch.norm(
            gamma_analytic
        )
        tolerance = isotropic_noise_data['tolerance']
        assert gamma_error < tolerance

    def test_second_moment_error(self, isotropic_noise_data):
        # unpack data
        mu = isotropic_noise_data['mu']
        sigma = isotropic_noise_data['sigma']
        second_m_empirical = isotropic_noise_data['second_m_empirical']

        # Get analytical estimate
        second_m_analytic = pnf.prnorm_sm_iso(mu=mu, sigma=sigma)

        # Check error
        second_m_error = torch.norm(
            second_m_empirical - second_m_analytic
        ) / torch.norm(second_m_analytic)
        tolerance = isotropic_noise_data['tolerance']
        assert second_m_error < tolerance


### TEST THAT THE FUNCTIONS CAN TAKE MANY MEANS AT ONCE

# Fixture to set up data for isotropic noise case
@pytest.fixture(scope='function')
def isotropic_batch_data(n_dim, sigma):
    n_batch = 15
    tolerance = 1e-6

    # Generate many means and compute the second moment for each one
    mu = torch.zeros((n_batch, n_dim))
    gamma_individual = torch.zeros((n_batch, n_dim))
    second_m_individual = torch.zeros((n_batch, n_dim, n_dim))

    for i in range(n_batch):
        mu[i, :] = make_mu(n_dim=n_dim, mu_type='sin')
        gamma_individual[i, :] = pnf.prnorm_mean_iso(mu=mu[i], sigma=sigma)
        second_m_individual[i, :, :] = pnf.prnorm_sm_iso(mu=mu[i], sigma=sigma)

    return {
        'mu': mu,
        'sigma': sigma,
        'gamma_individual': gamma_individual,
        'second_m_individual': second_m_individual,
        'tolerance': tolerance,
    }


@pytest.mark.parametrize('n_dim', [2, 3, 5])
@pytest.mark.parametrize('sigma', [0.5, 1])
def test_isotropic_batch_data(isotropic_batch_data, n_dim, sigma):

    mu_batch = isotropic_batch_data['mu']
    sigma = isotropic_batch_data['sigma']
    gamma_individual = isotropic_batch_data['gamma_individual']
    second_m_individual = isotropic_batch_data['second_m_individual']
    tolerance = isotropic_batch_data['tolerance']

    # Get the mean using batch inut
    gamma_batch = pnf.prnorm_mean_iso(mu=mu_batch, sigma=sigma)
    second_m_batch = pnf.prnorm_sm_iso(mu=mu_batch, sigma=sigma)

    # Get the batch mean efficient computation

    # Compute gamma error
    gamma_error = torch.max(gamma_individual - gamma_batch)
    # Compute the error
    second_m_error = torch.max(second_m_individual - second_m_batch)

    assert gamma_error < tolerance, f'Error of batch input for prnorm_mean_iso too large'
    assert second_m_error < tolerance, f'Error of batch input for prnorm_sm_iso too large'


@pytest.mark.parametrize('n_dim', [2, 3, 5])
@pytest.mark.parametrize('sigma', [0.5, 1])
def test_isotropic_batch_mean(isotropic_batch_data, n_dim, sigma):

    mu_batch = isotropic_batch_data['mu']
    sigma = isotropic_batch_data['sigma']
    second_m_individual = isotropic_batch_data['second_m_individual']
    tolerance = isotropic_batch_data['tolerance']

    # Average individually computed second moments
    second_m_average = torch.mean(second_m_individual, dim=0)

    # Compute average second moment efficiently
    second_m_efficient = pnf.prnorm_sm_iso_batch(mu=mu_batch, sigma=sigma)

    # Compute the error
    second_m_error = torch.max(second_m_average - second_m_efficient)

    assert second_m_error < tolerance

