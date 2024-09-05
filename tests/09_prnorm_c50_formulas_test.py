##################
#
# TESTS the implementation of the formulas for the Projected
# normal with a c50 term. This is done by comparing the
# formulas for jacobian and determinant to the
# numerical derivatives.
#
##################

import time
import pytest
import torch
import projected_normal.prnorm_func as pnf
from utility_functions import make_mu, make_covariance

######### CHECK THAT THE INVERTED PROJECTION GIVES EXPECTED RESULT ############

# Instantiate parameters
@pytest.fixture(scope='function')
def projection_result(n_points, n_dim, scale, c50):

    tolerance = 1e-4
    x = torch.randn(n_points, n_dim) * scale
    norm_factor = 1 / torch.sqrt(x.pow(2).sum(dim=-1) + c50)
    y = torch.einsum('ij,i->ij', [x, norm_factor])
    return {'x': x, 'y': y, 'c50': c50, 'tolerance': tolerance}

@pytest.mark.parametrize('n_points', [1, 10])
@pytest.mark.parametrize('n_dim', [2, 3, 20])
@pytest.mark.parametrize('scale', [1])
@pytest.mark.parametrize('c50', [0.1, 1])
def test_inverted_projection(projection_result):

    x = projection_result['x']
    y = projection_result['y']
    c50 = projection_result['c50']

    tolerance = projection_result['tolerance']
    x_reconstructed = pnf.invert_projection(y, c50)
    assert torch.allclose(x, x_reconstructed, atol=tolerance), 'Inverted projection does not give the true result.'

######### CHECK THAT THE JACOBIAN IS CORRECT COMPARING TO AUTOGRAD ############

# Instantiate parameters, including Jacobian and determinant
@pytest.fixture(scope='function')
def projection_jacobian(n_points, n_dim, scale, c50):

    tolerance = 1e-6
    x = torch.randn(n_points, n_dim) * scale
    norm_factor = 1 / torch.sqrt(x.pow(2).sum(dim=-1) + c50)
    y = torch.einsum('ij,i->ij', [x, norm_factor])
    c50 = torch.tensor(3.0)

    # Compute the Jacobian matrix for each point
    jacobian = torch.zeros((n_points, n_dim, n_dim))
    determinants = torch.zeros(n_points)
    for i in range(n_points):
        jacobian[i,:,:] = torch.autograd.functional.jacobian(
          pnf.invert_projection, (y[i], c50)
        )[0]
        determinants[i] = torch.linalg.det(jacobian[i,:,:])

    return {'x': x, 'y': y, 'c50': c50, 'jacobian': jacobian,
            'determinants': determinants, 'tolerance': tolerance}


@pytest.mark.parametrize('n_points', [1, 10])
@pytest.mark.parametrize('n_dim', [2, 3, 20])
@pytest.mark.parametrize('scale', [1])
@pytest.mark.parametrize('c50', [0.5, 1, 10])
def test_jacobian(projection_jacobian):

    x = projection_jacobian['x']
    y = projection_jacobian['y']
    c50 = projection_jacobian['c50']
    jacobian_autograd = projection_jacobian['jacobian']
    determinants_autograd = projection_jacobian['determinants']
    tolerance = projection_jacobian['tolerance']

    # Compute the Jacobian matrix for each point
    jacobian = pnf.invert_projection_jacobian_matrix(y, c50)
    # Compute the determinant of the Jacobian matrix for each point
    determinants = pnf.invert_projection_det(y, c50)
    # Compute the log determinants
    log_determinants = pnf.invert_projection_log_det(y, c50)

    assert not torch.isinf(determinants).any(), 'Determinants are infinite'
    assert torch.allclose(jacobian, jacobian_autograd, atol=tolerance), \
        'Inverted projection does not give the true result.'
    assert torch.allclose(determinants, determinants_autograd, atol=tolerance), \
        'Inverted projection does not give the true result.'
    assert torch.allclose(log_determinants, torch.log(determinants_autograd), atol=tolerance), \
        'Determinant and log determinant do not match.'


######### CHECK THAT THE PDF WORKS AS EXPECTED ############

# Instantiate parameters and sample from distribution
@pytest.fixture(scope='function')
def gaussian_parameters(n_points, n_dim, mean_type, cov_type, sigma, c50):

    # Initialize the mean of the gaussian
    mu = make_mu(n_dim=n_dim, mu_type=mean_type)
    # Initialize the covariance of the gaussian
    covariance = make_covariance(n_dim=n_dim, cov_scale=sigma, cov_type=cov_type)

    # Make samples from the distribution
    y = pnf.sample_prnorm(mu=mu, covariance=covariance,
                          n_samples=n_points, c50=c50)

    return {'mu': mu, 'covariance': covariance, 'c50': c50, 'y':y}


@pytest.mark.parametrize('n_points', [1, 10, 1000])
@pytest.mark.parametrize('n_dim', [2, 3, 20])
@pytest.mark.parametrize('mean_type', ['sin', 'sparse'])
@pytest.mark.parametrize('cov_type', ['random', 'diagonal'])
@pytest.mark.parametrize('sigma', [0.1, 1])
@pytest.mark.parametrize('c50', [0.5, 1, 10])
def test_pdf(gaussian_parameters):

    # Unpack parameters
    mu = gaussian_parameters['mu']
    covariance = gaussian_parameters['covariance']
    c50 = gaussian_parameters['c50']
    n_dim = mu.shape[0]
    # Unpack samples
    y = gaussian_parameters['y']

    # Compute the pdf
    pdf = pnf.prnorm_c50_pdf(mu=mu, covariance=covariance, c50=c50, y=y)
    # Compute the log pdf
    log_pdf = pnf.prnorm_c50_log_pdf(mu=mu, covariance=covariance, c50=c50, y=y)

    assert not torch.isnan(pdf).any(), 'PDFs are nan'
    assert not torch.isinf(pdf).any(), 'Log-PDFs are infinite'
    assert torch.all(pdf > 0), 'PDFs are non-positive'
    assert not torch.isnan(log_pdf).any(), 'Log-PDFs are nan'
    assert torch.allclose(torch.exp(log_pdf), pdf), 'Log-PDFs are not consistent with PDFs'
    # Check that pdfs are not infinte
    assert not torch.isinf(log_pdf).any(), 'Log-PDFs are infinite'

