"""Test the c50 PDF implementation in the projected normal distribution."""
import pytest
import torch
import projected_normal.distribution.c50 as pnc
import projected_normal.param_sampling as par_samp

torch.manual_seed(1)


def relative_error(x, y):
    """Compute the relative error between two tensors."""
    return torch.norm(x - y) * 2 / (torch.norm(y) + torch.norm(x))


######### CHECK THAT THE INVERTED PROJECTION GIVES EXPECTED RESULT ############
# Instantiate parameters
@pytest.fixture(scope='function')
def projection_result(n_points, n_dim, scale, c50):
    """
    Take a random x and divide by sqrt(x'x + c50) to get y.
    Return input and output.
    """
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
    x_reconstructed = pnc.pdf._invert_projection(y, c50)
    assert torch.allclose(x, x_reconstructed, atol=tolerance), \
        'Inverted projection does not give the true result.'


######### CHECK THAT THE JACOBIAN IS CORRECT COMPARING TO AUTOGRAD ############
@pytest.fixture(scope='function')
def projection_jacobian(n_points, n_dim, scale, c50):
    """Compute the Jacobian matrix for the inverse projection using autograd."""
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
          pnc.pdf._invert_projection, (y[i], c50)
        )[0]
        determinants[i] = torch.linalg.det(jacobian[i,:,:])

    return {'x': x, 'y': y, 'c50': c50, 'jacobian': jacobian,
            'determinants': determinants, 'tolerance': tolerance}


@pytest.mark.parametrize('n_points', [1, 10])
@pytest.mark.parametrize('n_dim', [2, 3, 10])
@pytest.mark.parametrize('scale', [1])
@pytest.mark.parametrize('c50', [0.5, 1, 3])
def test_jacobian(projection_jacobian):
    """Test the computation of the Jacobian matrix for the inverse projection."""
    x = projection_jacobian['x']
    y = projection_jacobian['y']
    c50 = projection_jacobian['c50']
    jacobian_autograd = projection_jacobian['jacobian']
    determinants_autograd = projection_jacobian['determinants']
    tolerance = projection_jacobian['tolerance']

    # Compute the Jacobian matrix for each point
    jacobian = pnc.pdf._invert_projection_jacobian_matrix(y, c50)
    # Compute the determinant of the Jacobian matrix for each point
    determinants = pnc.pdf._invert_projection_det(y, c50)
    # Compute the log determinants
    log_determinants = pnc.pdf._invert_projection_log_det(y, c50)

    assert not torch.isinf(determinants).any(), 'Determinants are infinite'
    assert torch.allclose(jacobian, jacobian_autograd, atol=tolerance), \
        'Inverted projection does not give the true result.'
    assert torch.allclose(determinants, determinants_autograd, atol=tolerance), \
        'Inverted projection does not give the true result.'
    assert torch.allclose(log_determinants, torch.log(determinants_autograd), atol=tolerance), \
        'Determinant and log determinant do not match.'


######### CHECK THAT THE PDF WORKS AS EXPECTED ############
# Instantiate parameters
@pytest.fixture(scope="function")
def gaussian_parameters(n_points, n_dim, mean_type, eigvals, eigvecs, sigma, c50):
    """ Fixture to generate Gaussian parameters for tests."""
    # Initialize the mean of the gaussian
    # Parameters of distribution
    mean_x = par_samp.make_mean(
      n_dim=n_dim, shape=mean_type
    )
    covariance_x = par_samp.make_spdm(
      n_dim=n_dim, eigvals=eigvals, eigvecs=eigvecs
    ) * sigma**2

    y = pnc.sampling.sample(mean_x, covariance_x, n_points, c50=c50)

    return {
        "mean_x": mean_x,
        "covariance_x": covariance_x,
        "c50": c50,
        "y": y,
    }

@pytest.mark.parametrize('n_points', [1, 10])
@pytest.mark.parametrize('n_dim', [2, 3, 10])
@pytest.mark.parametrize('mean_type', ['sin', 'sparse'])
@pytest.mark.parametrize("eigvals", ["uniform", "exponential"])
@pytest.mark.parametrize("eigvecs", ["random", "identity"])
@pytest.mark.parametrize('sigma', [0.1, 1])
@pytest.mark.parametrize('c50', [0.5, 1, 10])
def test_pdf(gaussian_parameters):
    """Test that the pdf of the projected gaussian with additive constant
    does not return nan or inf and is consistent with the log pdf."""
    # Unpack parameters
    mean_x = gaussian_parameters['mean_x']
    covariance_x = gaussian_parameters['covariance_x']
    c50 = gaussian_parameters['c50']
    # Unpack samples
    y = gaussian_parameters['y']

    # Compute the pdf
    pdf = pnc.pdf.pdf(
      mean_x=mean_x, covariance_x=covariance_x, c50=c50, y=y
    )
    # Compute the log pdf
    log_pdf = pnc.pdf.log_pdf(
      mean_x=mean_x, covariance_x=covariance_x, c50=c50, y=y
    )

    assert not torch.isnan(pdf).any(), 'PDFs are nan'
    assert not torch.isinf(pdf).any(), 'Log-PDFs are infinite'
    assert torch.all(pdf > 0), 'PDFs are non-positive'
    assert not torch.isnan(log_pdf).any(), 'Log-PDFs are nan'
    assert torch.allclose(torch.exp(log_pdf), pdf), 'Log-PDFs are not consistent with PDFs'
    # Check that pdfs are not infinte
    assert not torch.isinf(log_pdf).any(), 'Log-PDFs are infinite'

