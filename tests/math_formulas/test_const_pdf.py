"""Test the PDF implementation for projected normal with denominator constant."""
import pytest
import torch

import projnormal.formulas.projected_normal_c as pnc_formulas
import projnormal.param_sampling as par_samp

torch.manual_seed(1)


def relative_error(x, y):
    """Compute the relative error between two tensors."""
    return torch.norm(x - y) * 2 / (torch.norm(y) + torch.norm(x))


######### CHECK THAT THE INVERTED PROJECTION GIVES EXPECTED RESULT ############
# Instantiate parameters
@pytest.fixture(scope='function')
def projection_result(n_points, n_dim, scale, const):
    """
    Take a random x and divide by sqrt(x'x + const) to get y.
    Return input and output.
    """
    tolerance = 1e-4
    x = torch.randn(n_points, n_dim) * scale
    norm_factor = 1 / torch.sqrt(x.pow(2).sum(dim=-1) + const)
    y = torch.einsum('ij,i->ij', [x, norm_factor])
    return {'x': x, 'y': y, 'const': const, 'tolerance': tolerance}


@pytest.mark.parametrize('n_points', [1, 10])
@pytest.mark.parametrize('n_dim', [2, 3, 20])
@pytest.mark.parametrize('scale', [1])
@pytest.mark.parametrize('const', [0.1, 1])
def test_inverted_projection(const, projection_result):
    """Test that the inverted projection gives the expected result."""
    x = projection_result['x']
    y = projection_result['y']

    tolerance = projection_result['tolerance']
    x_reconstructed = pnc_formulas.probability._invert_projection(y, const)
    assert torch.allclose(x, x_reconstructed, atol=tolerance), \
        'Inverted projection does not give the true result.'


######### CHECK THAT THE JACOBIAN IS CORRECT COMPARING TO AUTOGRAD ############
@pytest.fixture(scope='function')
def projection_jacobian(n_points, n_dim, scale, const):
    """Compute the Jacobian matrix for the inverse projection using autograd."""
    tolerance = 1e-6
    x = torch.randn(n_points, n_dim) * scale
    norm_factor = 1 / torch.sqrt(x.pow(2).sum(dim=-1) + const)
    y = torch.einsum('ij,i->ij', [x, norm_factor])
    const = torch.tensor(3.0)

    # Compute the Jacobian matrix for each point
    jacobian = torch.zeros((n_points, n_dim, n_dim))
    determinants = torch.zeros(n_points)
    for i in range(n_points):
        jacobian[i,:,:] = torch.autograd.functional.jacobian(
          pnc_formulas.probability._invert_projection, (y[i], const)
        )[0]
        determinants[i] = torch.linalg.det(jacobian[i,:,:])

    return {'x': x, 'y': y, 'const': const, 'jacobian': jacobian,
            'determinants': determinants, 'tolerance': tolerance}


@pytest.mark.parametrize('n_points', [1, 10])
@pytest.mark.parametrize('n_dim', [2, 3, 10])
@pytest.mark.parametrize('scale', [1])
@pytest.mark.parametrize('const', [0.5, 1, 3])
def test_jacobian(projection_jacobian):
    """Test the computation of the Jacobian matrix for the inverse projection."""
    y = projection_jacobian['y']
    const = projection_jacobian['const']
    jacobian_autograd = projection_jacobian['jacobian']
    determinants_autograd = projection_jacobian['determinants']
    tolerance = projection_jacobian['tolerance']

    # Compute the Jacobian matrix for each point
    jacobian = pnc_formulas.probability._invert_projection_jacobian_matrix(y, const)
    # Compute the determinant of the Jacobian matrix for each point
    determinants = pnc_formulas.probability._invert_projection_det(y, const)
    # Compute the log determinants
    log_determinants = pnc_formulas.probability._invert_projection_log_det(y, const)

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
def gaussian_parameters(n_points, n_dim, mean_type, eigvals, eigvecs, sigma, const):
    """Fixture to generate Gaussian parameters for tests."""
    # Initialize the mean of the gaussian
    # Parameters of distribution
    mean_x = par_samp.make_mean(
      n_dim=n_dim, shape=mean_type
    )
    covariance_x = par_samp.make_spdm(
      n_dim=n_dim, eigvals=eigvals, eigvecs=eigvecs
    ) * sigma**2

    y = pnc_formulas.sample(mean_x, covariance_x, const=const, n_samples=n_points)

    return {
        "mean_x": mean_x,
        "covariance_x": covariance_x,
        "y": y,
    }

@pytest.mark.parametrize('n_points', [1, 10])
@pytest.mark.parametrize('n_dim', [2, 3, 10])
@pytest.mark.parametrize('mean_type', ['sin', 'sparse'])
@pytest.mark.parametrize("eigvals", ["uniform", "exponential"])
@pytest.mark.parametrize("eigvecs", ["random", "identity"])
@pytest.mark.parametrize('sigma', [0.1, 1])
@pytest.mark.parametrize('const', [0.5, 1, 10])
def test_pdf(const, gaussian_parameters):
    """Test that the pdf of the projected gaussian with additive constant
    does not return nan or inf and is consistent with the log pdf.
    """
    # Unpack parameters
    mean_x = gaussian_parameters['mean_x']
    covariance_x = gaussian_parameters['covariance_x']
    # Unpack samples
    y = gaussian_parameters['y']

    # Compute the pdf
    pdf = pnc_formulas.pdf(
      mean_x=mean_x, covariance_x=covariance_x, const=const, y=y
    )
    # Compute the log pdf
    log_pdf = pnc_formulas.log_pdf(
      mean_x=mean_x, covariance_x=covariance_x, const=const, y=y
    )

    assert not torch.isnan(pdf).any(), 'PDFs are nan'
    assert not torch.isinf(pdf).any(), 'Log-PDFs are infinite'
    assert torch.all(pdf > 0), 'PDFs are non-positive'
    assert not torch.isnan(log_pdf).any(), 'Log-PDFs are nan'
    assert torch.allclose(torch.exp(log_pdf), pdf), 'Log-PDFs are not consistent with PDFs'
    # Check that pdfs are not infinte
    assert not torch.isinf(log_pdf).any(), 'Log-PDFs are infinite'

