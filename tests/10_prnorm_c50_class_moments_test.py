##################
#
# TESTS THAT THE PROJECTED NORMAL DISTRIBUTION CLASS
# IS ABLE TO PERFORM ITS BASIC FUNCTIONS
#
##################

import time
import pytest
import torch
from projected_normal.prnorm_class import ProjectedNormalC50
from projected_normal.auxiliary import is_symmetric, is_positive_definite
from utility_functions import make_mu, make_covariance


# Instantiate parameters
@pytest.fixture(scope='function')
def gaussian_parameters(n_dim, mean_type, cov_type, sigma):

    tolerance = 1e-3

    # Initialize the mean of the gaussian
    mu = make_mu(n_dim=n_dim, mu_type=mean_type)

    # Initialize the covariance of the gaussian
    covariance = make_covariance(n_dim=n_dim, cov_scale=sigma, cov_type=cov_type)

    return {'mu': mu, 'covariance': covariance, 'tolerance': tolerance}

######## TEST THAT SAMPLING RUNS ########

@pytest.mark.parametrize('n_dim', [2, 3, 5, 25])
@pytest.mark.parametrize('mean_type', ['sin', 'sparse'])
@pytest.mark.parametrize('cov_type', ['random', 'diagonal'])
@pytest.mark.parametrize('sigma', [0.01, 0.1, 0.5, 1])
@pytest.mark.parametrize('c50', [0.1, 1, 5, 10])
def test_sampling_works(gaussian_parameters, c50):

    # Unpack parameters
    mu = gaussian_parameters['mu']
    covariance = gaussian_parameters['covariance']
    n_dim = mu.shape[0]

    # Initialize the projected normal class
    prnorm = ProjectedNormalC50(n_dim=n_dim, mu=mu, covariance=covariance, c50=c50, requires_grad=False)

    # Sample from the distribution
    samples = prnorm.sample(n_samples=500)

    # compute the norm of the samples
    norm_samples = torch.norm(samples, dim=1)

    assert not torch.isnan(samples).any(), 'Samples are nan'
    assert not torch.any(norm_samples >= torch.tensor(1.0)), 'Samples norm is not smaller than 1.0'


######### CHECK THAT THE MOMENT FUNCTIONS RUN ############
def norm_leq_1(gamma):
    return torch.norm(gamma) <= 1

@pytest.mark.parametrize('n_dim', [2, 3, 5, 10, 25])
@pytest.mark.parametrize('mean_type', ['sin', 'sparse'])
@pytest.mark.parametrize('cov_type', ['random', 'diagonal'])
@pytest.mark.parametrize('sigma', [0.05, 0.1, 0.5, 1])
@pytest.mark.parametrize('c50', [0.1, 1, 5, 10])
def test_moments_work(gaussian_parameters, c50):

    # Unpack parameters
    mu = gaussian_parameters['mu']
    covariance = gaussian_parameters['covariance']
    n_dim = mu.shape[0]

    # Initialize the projected normal class
    prnorm = ProjectedNormalC50(n_dim=n_dim, mu=mu, covariance=covariance, c50=c50, requires_grad=False)

    # Get taylor approximation moments
    with torch.no_grad():
        moments_taylor = prnorm.moments_approx()

    # Get the empirical moments
    with torch.no_grad():
        moments_empirical = prnorm.moments_empirical(n_samples=10000)

    # Check that Taylor moments are not nan
    assert not torch.isnan(
        moments_taylor['gamma']
    ).any(), 'Taylor approximation of the mean is nan'
    assert not torch.isnan(
        moments_taylor['psi']
    ).any(), 'Taylor approximation of the covariance is nan'
    assert not torch.isnan(
        moments_taylor['second_moment']
    ).any(), 'Taylor approximation of the second moment is nan'

    # Check that Empirical moments are not nan
    assert not torch.isnan(moments_empirical['gamma']).any(), 'Empirical mean is nan'
    assert not torch.isnan(moments_empirical['psi']).any(), 'Empirical covariance is nan'

    # Check that gamma has norm <= 1
    assert norm_leq_1(moments_taylor['gamma']), 'Taylor approximation of the mean has norm > 1'
    assert norm_leq_1(moments_empirical['gamma']), 'Empirical mean has norm > 1'

    # Check that psi and second moment are symmetric and positive definite
    # Taylor covariance
    assert is_symmetric(
        moments_taylor['psi']
    ), 'Taylor approximation of the covariance is not symmetric'
    #assert is_positive_definite(
    #    moments_taylor['psi']
    #), 'Taylor approximation of the covariance is not positive definite'
    # Second moment
    assert is_symmetric(
        moments_taylor['second_moment']
    ), 'Taylor approximation of the second moment is not symmetric'
    assert is_positive_definite(
        moments_taylor['second_moment']
    ), 'Taylor approximation of the second moment is not positive definite'
    # Empirical
    assert is_symmetric(moments_empirical['psi']), 'Empirical covariance is not symmetric'
    assert is_positive_definite(
        moments_empirical['psi']
    ), 'Empirical covariance is not positive definite'


########## CHECK THAT THE MOMENT MATCHING WORKS ############
@pytest.mark.parametrize('n_dim', [2, 3, 20])
@pytest.mark.parametrize('mean_type', ['sin'])
@pytest.mark.parametrize('cov_type', ['random', 'diagonal'])
@pytest.mark.parametrize('sigma', [0.01, 0.1, 0.5])
@pytest.mark.parametrize('cov_param', ['Logarithm'])
@pytest.mark.parametrize('c50', [1, 5, 10])
def test_moment_matching_works(gaussian_parameters, cov_param, c50):

    # Unpack parameters
    mu = gaussian_parameters['mu']
    covariance = gaussian_parameters['covariance']
    n_dim = mu.shape[0]

    # Initialize the projected normal class to generate the data
    prnorm_data = ProjectedNormalC50(n_dim=n_dim, mu=mu, covariance=covariance,
                                  c50=c50, covariance_parametrization=cov_param)

    # Generate the taylor moments to fit to (instead of empirical moments)
    with torch.no_grad():
        moments_taylor = prnorm_data.moments_approx()

    # Initialize the projected normal to fit to the data
    covariance_initial = torch.eye(n_dim)   # Put a far off covariance
    prnorm_fit = ProjectedNormalC50(n_dim=n_dim, covariance=covariance_initial,
                             covariance_parametrization=cov_param, c50=2.0)

    # Get initial parameters
    mu_initial = prnorm_fit.mu.clone().detach()
    covariance_initial = prnorm_fit.covariance.clone().detach()
    c50_initial = prnorm_fit.c50.clone().detach()

    # Fit to the data with moment matching
    n_rep = 3
    loss_list = []
    for i in range(n_rep):
        # Initialize optimizer and scheduler
        lr = 0.05 * (0.5**i)
        optimizer, scheduler = prnorm_fit.initialize_optimizer_and_scheduler(
          lr=lr,
          decay_iter=5
        )

        # Fit to the data with moment_matching
        loss = prnorm_fit.moment_match(
            gamma_obs=moments_taylor['gamma'],
            psi_obs=moments_taylor['psi'],
            optimizer=optimizer,
            scheduler=scheduler,
            n_iter=20
        )
        loss_list.append(loss)
    loss = torch.cat(loss_list)

    # Estimated parameters
    mu_estimated = prnorm_fit.mu.detach()
    covariance_estimated = prnorm_fit.covariance.detach()
    c50_estimated = prnorm_fit.c50.detach()

    # Get errors
    mu_error_i = torch.norm(mu_initial - mu)
    mu_error_f = torch.norm(mu_estimated - mu)
    covariance_error_i = torch.norm(covariance_initial - covariance)
    covariance_error_f = torch.norm(covariance_estimated - covariance)
    c50_error_i = torch.abs(c50_initial - c50)
    c50_error_f = torch.abs(c50_estimated - c50)

    assert not torch.isnan(loss).any(), 'Loss is nan'
    assert not torch.isnan(mu_estimated).any(), 'Estimated mu is nan'
    assert not torch.isnan(covariance_estimated).any(), 'Estimated covariance is nan'
    assert not torch.isnan(c50_estimated).any(), 'Estimated c50 is nan'
    assert loss[0] > loss[-1], 'Loss did not decrease'
    assert torch.allclose(
        mu_estimated.norm(), torch.tensor(1.0)
    ), 'Estimated mean norm is not 1'
    assert is_symmetric(covariance_estimated), 'Estimated covariance is not symmetric'
    assert is_positive_definite(
        covariance_estimated
    ), 'Estimated covariance is not positive definite'

