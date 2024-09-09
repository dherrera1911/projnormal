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
from utility_functions import make_mu, make_covariance

# Instantiate parameters
@pytest.fixture(scope="function")
def gaussian_parameters(n_dim, mean_type, cov_type, sigma, B_type, c50):

    tolerance = 1e-2
    n_samples = 500000

    # Initialize the mean of the gaussian
    mu = make_mu(n_dim=n_dim, mu_type=mean_type)

    # Initialize the covariance of the gaussian
    covariance = make_covariance(n_dim=n_dim, cov_scale=sigma, cov_type=cov_type)

    # Initialize the matrix B
    if B_type == 'isotropic':
        B_diagonal = torch.ones(n_dim) * torch.rand(1)
    elif B_type == 'exponential':
        i = torch.arange(n_dim)
        B_diagonal = torch.exp(-i.float()*3/n_dim)

    return {
      'mu': mu,
      'covariance': covariance,
      'B_diagonal': B_diagonal,
      'c50': c50,
      'tolerance': tolerance,
      'n_samples': n_samples
    }


######### CHECK THAT DIAGONAL AND FULL B GIVE SAME SAMPLING RESULT ############

@pytest.mark.parametrize('n_dim', [3, 10])
@pytest.mark.parametrize('mean_type', ['sin', 'sparse'])
@pytest.mark.parametrize('cov_type', ['random', 'diagonal'])
@pytest.mark.parametrize('B_type', ['isotropic', 'exponential'])
@pytest.mark.parametrize('sigma', [0.1])
@pytest.mark.parametrize('c50', [0, 1])
def test_empirical_sampling(gaussian_parameters):
    # Unpack parameters
    mu = gaussian_parameters['mu']
    covariance = gaussian_parameters['covariance']
    B_diagonal = gaussian_parameters['B_diagonal']
    c50 = gaussian_parameters['c50']
    n_samples = gaussian_parameters['n_samples']
    tolerance = gaussian_parameters['tolerance']

    # Get full B
    B = torch.diag(B_diagonal)

    # Get empirical moments with full B
    moments_empirical = pnf.empirical_moments_prnorm(mu, covariance, B=B, c50=c50,
                                                    n_samples=n_samples)
    # Get empirical moments with diagonal B
    moments_empirical_diag = pnf.empirical_moments_prnorm(mu, covariance, B=B_diagonal, c50=c50,
                                                         n_samples=n_samples)

    # Check that outputs are close
    gamma_full = moments_empirical['gamma']
    gamma_diag = moments_empirical_diag['gamma']

    gamma_diff = torch.max(gamma_full - gamma_diag)

    assert gamma_diff < tolerance, 'Empirical moments not the same for diagonal and full B'

