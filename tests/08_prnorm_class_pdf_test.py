##################
#
# TESTS THAT THE PROJECTED NORMAL DISTRIBUTION CLASS
# IS ABLE TO PERFORM ITS BASIC FUNCTIONS
#
##################

import time
import logging
import pytest
import torch
from projected_normal.prnorm_class import ProjectedNormal
from projected_normal.auxiliary import is_symmetric, is_positive_definite
from utility_functions import make_mu, make_covariance, loss_function_pdf

log = logging.getLogger(__name__)

# Instantiate parameters
@pytest.fixture(scope='function')
def gaussian_parameters(n_dim, mean_type, cov_type, sigma):

    tolerance = 1e-3

    # Initialize the mean of the gaussian
    mu = make_mu(n_dim=n_dim, mu_type=mean_type)
    # Initialize the covariance of the gaussian
    covariance = make_covariance(n_dim=n_dim, cov_scale=sigma, cov_type=cov_type)

    return {'mu': mu, 'covariance': covariance, 'tolerance': tolerance}


######### CHECK THAT THE PDF FUNCTIONS RUN ############

@pytest.mark.parametrize('n_dim', [3, 5, 10, 20])
@pytest.mark.parametrize('mean_type', ['sin', 'sparse'])
@pytest.mark.parametrize('cov_type', ['random', 'diagonal'])
@pytest.mark.parametrize('sigma', [0.01, 0.1, 0.5, 1])
def test_pdf_works(gaussian_parameters):

    # Unpack parameters
    mu = gaussian_parameters['mu']
    covariance = gaussian_parameters['covariance']
    n_dim = mu.shape[0]

    # Initialize the projected normal class
    prnorm = ProjectedNormal(
        n_dim=n_dim, mu=mu, covariance=covariance, dtype=torch.float64
    )

    # Sample from the distribution
    samples = prnorm.sample(n_samples=1000)

    # Compute pdf of the samples
    with torch.no_grad():
        pdf = prnorm.pdf(y=samples)

    # Compute the log pdf of the samples
    with torch.no_grad():
        log_pdf = prnorm.log_pdf(y=samples)

    exp_log_pdf = torch.exp(log_pdf)

    assert not torch.isnan(pdf).any(), 'PDFs are nan'
    assert torch.all(pdf > 0), 'PDFs are non-positive'
    assert not torch.isnan(log_pdf).any(), 'Log-PDFs are nan'
    assert torch.allclose(exp_log_pdf, pdf), 'Log-PDFs are not consistent with PDFs'
    # Check that pdfs are not infinte
    assert not torch.isinf(log_pdf).any(), 'Log-PDFs are infinite'

######### CHECK THAT PDF MAXIMUM IS WHERE EXPECTED ############
# For isotropic gaussian, the maximum of the pdf should be at
# the value of mu, because of symmetry

@pytest.mark.parametrize('n_dim', [3, 5, 10, 20])
@pytest.mark.parametrize('mean_type', ['sin', 'sparse'])
@pytest.mark.parametrize('cov_type', ['isotropic'])
@pytest.mark.parametrize('sigma', [0.01, 0.1, 0.5, 1])
def test_pdf_maximum(gaussian_parameters):

    # Unpack parameters
    mu = gaussian_parameters['mu']
    covariance = gaussian_parameters['covariance']
    n_dim = mu.shape[0]

    # Initialize the projected normal class
    prnorm = ProjectedNormal(
        n_dim=n_dim, mu=mu, covariance=covariance, dtype=torch.float64
    )

    # Sample from the distribution
    samples = prnorm.sample(n_samples=1000)

    # Compute the log pdf of the samples
    with torch.no_grad():
        log_pdf = prnorm.log_pdf(y=samples)

    # Compute the log pdf of the mu
    log_pdf_mu = prnorm.log_pdf(y=mu)

    assert torch.all(log_pdf <= log_pdf_mu), 'Maximum of pdf is not at mu'


######### CHECK THAT ML FITTING WORKS ############

@pytest.mark.parametrize('n_dim', [3, 5, 10])
@pytest.mark.parametrize('mean_type', ['sin'])
@pytest.mark.parametrize('cov_type', ['random', 'diagonal'])
@pytest.mark.parametrize('sigma', [0.01, 0.1, 0.5])
@pytest.mark.parametrize('cov_param', ['Logarithm', 'Spectral'])
def test_ml_fitting_works(gaussian_parameters, cov_param):

    # Unpack parameters
    mu = gaussian_parameters['mu']
    covariance = gaussian_parameters['covariance']
    n_dim = mu.shape[0]

    # Initialize the projected normal class to generate the data
    dtype = torch.float32
    prnorm_data = ProjectedNormal(
        n_dim=n_dim, mu=mu, covariance=covariance,
        covariance_parametrization=cov_param, dtype=dtype
    )

    # Sample from the distribution
    with torch.no_grad():
        samples = prnorm_data.sample(n_samples=3000)

    # Initialize the projected normal to fit to the data
    # Use paramters close to the true to avoid numerical issues
    covariance_initial = covariance + 0.2 * torch.eye(n_dim)
    mu_initial = mu + torch.randn(n_dim) * 0.1
    mu_initial = mu_initial / torch.norm(mu_initial)

    # Initialize the projected normal class
    prnorm_fit = ProjectedNormal(
      n_dim=n_dim,
      mu=mu_initial,
      covariance=covariance_initial,
      covariance_parametrization=cov_param,
      dtype=dtype
    )

    # Fit to the data with maximum likelihood
    n_rep = 2
    loss_list = []
    for i in range(n_rep):
        # Initialize optimizer and scheduler
        lr = 0.05 * 0.5 ** i
        optimizer, scheduler = prnorm_fit.initialize_optimizer_and_scheduler(
          lr=lr,
          decay_iter=5
        )
        # Fit to the data
        loss = prnorm_fit.fit(
            data=samples, optimizer=optimizer, loss_function=loss_function_pdf,
          scheduler=scheduler, n_iter=20
        )
        loss_list.append(loss)
    loss = torch.cat(loss_list)

    # Get estimated parameters
    mu_estimated = prnorm_fit.mu.detach()
    covariance_estimated = prnorm_fit.covariance.detach()

    # Get errors
    mu_error_i = torch.norm(mu_initial - mu)
    mu_error_f = torch.norm(mu_estimated - mu)
    covariance_error_i = torch.norm(covariance_initial - covariance)
    covariance_error_f = torch.norm(covariance_estimated - covariance)

    # Check that the loss is a numer and that it decreases
    assert not torch.isnan(loss).any(), 'Loss is nan'
    assert loss[0] > loss[-1], 'Loss did not decrease'

    # Check that the estimated parameters adhere to the constraints
    assert torch.allclose(
        mu_estimated.norm(), torch.tensor(1.0, dtype=dtype)
    ), 'Estimated mu norm is not 1'
    assert is_symmetric(covariance_estimated), 'Estimated covariance is not symmetric'
    assert is_positive_definite(
        covariance_estimated
    ), 'Estimated covariance is not positive definite'

    # Check that the estimated parameters are closer to the true parameters
    if mu_error_i > mu_error_f:
        log.warning('Initial mu is closer to true mu than estimated mu')
    if covariance_error_i > covariance_error_f:
        log.warning('Initial covariance is closer to true covariance than estimated covariance')

