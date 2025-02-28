"""Test moments of isotropic projected normal distribution."""

import pytest
import torch
import projnormal.distribution.isotropic as iso
import projnormal.param_sampling as par_samp


def relative_error(x, y):
    """Compute the relative error between two tensors."""
    return torch.norm(x - y) * 2 / (torch.norm(y) + torch.norm(x))


# Fixture to set up data for isotropic noise case
@pytest.fixture(scope='class')
def iso_data(request):
    """Generate parameters to test, and obtain empirical estimates to compare with."""
    # Get parameters from the request
    n_dim = request.param['n_dim']
    n_samples = 100000
    sigma = request.param['sigma']
    tolerance = 0.05

    # Parameters of distribution
    mean_x = par_samp.make_mean(
      n_dim=n_dim, shape='sin'
    )

    # Get empirical estimates
    var_x = torch.as_tensor(sigma**2)
    moments_empirical = iso.sampling.empirical_moments(
        mean_x, var_x, n_samples=n_samples
    )

    return {
        'mean_x': mean_x,
        'sigma': sigma,
        'mean_empirical': moments_empirical['mean'],
        'covariance_empiricak': moments_empirical['covariance'],
        'second_moment_empirical': moments_empirical['second_moment'],
        'tolerance': tolerance,
    }


@pytest.mark.parametrize(
    'iso_data',
    [
        {'n_dim': 2, 'sigma': 0.1},
        {'n_dim': 2, 'sigma': 0.5},
        {'n_dim': 3, 'sigma': 0.1},
        {'n_dim': 3, 'sigma': 0.5},
        {'n_dim': 3, 'sigma': 1},
        {'n_dim': 10, 'sigma': 0.1},
        {'n_dim': 10, 'sigma': 0.5},
    ],
    indirect=True,
)
class TestIsotropicNoiseCase:

    def test_mean_error(self, iso_data):
        # unpack data
        mean_x = iso_data['mean_x']
        sigma = iso_data['sigma']
        mean_empirical = iso_data['mean_empirical']
        # Get analytical estimate
        var_x = torch.as_tensor(sigma**2)
        mean_analytic = iso.moments.mean(
          mean_x=mean_x, var_x=var_x
        )
        # Check error
        mean_error = relative_error(
          mean_empirical, mean_analytic
        )
        assert mean_error < iso_data['tolerance']

    def test_second_moment_error(self, iso_data):
        # unpack data
        mean_x = iso_data['mean_x']
        sigma = iso_data['sigma']
        second_m_empirical = iso_data['second_moment_empirical']
        # Get analytical estimate
        var_x = torch.as_tensor(sigma**2)
        second_m_analytic = iso.moments.second_moment(mean_x=mean_x, var_x=var_x)
        # Check error
        second_m_error = relative_error(
          second_m_empirical, second_m_analytic
        )
        assert second_m_error < iso_data['tolerance']


#### TEST THAT THE FUNCTIONS CAN TAKE MANY MEANS AT ONCE
#
## Fixture to set up data for isotropic noise case
#@pytest.fixture(scope='function')
#def isotropic_batch_data(n_dim, sigma):
#    n_batch = 15
#    tolerance = 1e-6
#
#    # Generate many means and compute the second moment for each one
#    mu = torch.zeros((n_batch, n_dim))
#    gamma_individual = torch.zeros((n_batch, n_dim))
#    second_m_individual = torch.zeros((n_batch, n_dim, n_dim))
#
#    for i in range(n_batch):
#        mu[i, :] = make_mu(n_dim=n_dim, mu_type='sin')
#        gamma_individual[i, :] = pnf.prnorm_mean_iso(mu=mu[i], sigma=sigma)
#        second_m_individual[i, :, :] = pnf.prnorm_sm_iso(mu=mu[i], sigma=sigma)
#
#    return {
#        'mu': mu,
#        'sigma': sigma,
#        'gamma_individual': gamma_individual,
#        'second_m_individual': second_m_individual,
#        'tolerance': tolerance,
#    }
#
#
#@pytest.mark.parametrize('n_dim', [2, 3, 5])
#@pytest.mark.parametrize('sigma', [0.5, 1])
#def test_isotropic_batch_data(isotropic_batch_data, n_dim, sigma):
#
#    mu_batch = isotropic_batch_data['mu']
#    sigma = isotropic_batch_data['sigma']
#    gamma_individual = isotropic_batch_data['gamma_individual']
#    second_m_individual = isotropic_batch_data['second_m_individual']
#    tolerance = isotropic_batch_data['tolerance']
#
#    # Get the mean using batch inut
#    gamma_batch = pnf.prnorm_mean_iso(mu=mu_batch, sigma=sigma)
#    second_m_batch = pnf.prnorm_sm_iso(mu=mu_batch, sigma=sigma)
#
#    # Get the batch mean efficient computation
#
#    # Compute gamma error
#    gamma_error = torch.max(gamma_individual - gamma_batch)
#    # Compute the error
#    second_m_error = torch.max(second_m_individual - second_m_batch)
#
#    assert gamma_error < tolerance, f'Error of batch input for prnorm_mean_iso too large'
#    assert second_m_error < tolerance, f'Error of batch input for prnorm_sm_iso too large'
#
#
#@pytest.mark.parametrize('n_dim', [2, 3, 5])
#@pytest.mark.parametrize('sigma', [0.5, 1])
#def test_isotropic_batch_mean(isotropic_batch_data, n_dim, sigma):
#
#    mu_batch = isotropic_batch_data['mu']
#    sigma = isotropic_batch_data['sigma']
#    second_m_individual = isotropic_batch_data['second_m_individual']
#    tolerance = isotropic_batch_data['tolerance']
#
#    # Average individually computed second moments
#    second_m_average = torch.mean(second_m_individual, dim=0)
#
#    # Compute average second moment efficiently
#    second_m_efficient = pnf.prnorm_sm_iso_batch(mu=mu_batch, sigma=sigma)
#
#    # Compute the error
#    second_m_error = torch.max(second_m_average - second_m_efficient)
#
#    assert second_m_error < tolerance
#
