"""Test the projected normal class."""
import pytest
import torch
import projnormal.param_sampling as par_samp
import projnormal.matrix_checks as checks
import projnormal.models as models
import projnormal.distribution.ellipse as pne


torch.manual_seed(1)
TOLERANCE = 0.025
MAX_ITER = 30
N_DIRS = 2

# Instantiate parameters, get empirical moments
@pytest.fixture(scope="function")
def gaussian_parameters(n_dim, sigma):
    """ Fixture to generate Gaussian parameters for tests."""

    # Initialize the mean of the gaussian
    # Parameters of distribution
    mean_x = par_samp.make_mean(n_dim=n_dim)
    covariance_x = par_samp.make_spdm(
      n_dim=n_dim
    ) * sigma**2

    return {
        "mean_x": mean_x,
        "covariance_x": covariance_x,
    }


@pytest.mark.parametrize('n_dim', [3, 7])
def test_init(n_dim):
    """Test the initialization of the ProjNormal class."""
    # Initialize without input parameters
    prnorm = models.ProjNormalEllipse(n_dim=n_dim)

    # Initialize parameters
    mean_x = torch.ones(n_dim) / torch.sqrt(torch.as_tensor(n_dim))
    covariance_x = torch.eye(n_dim)
    eigvecs = par_samp.make_ortho_vectors(n_dim, N_DIRS)
    eigvals = torch.tensor([0.4, 2.5])
    rad_sq = torch.tensor(1.0)

    prnorm = models.ProjNormalEllipse(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B_eigvecs=eigvecs,
      B_eigvals=eigvals,
      B_rad_sq=rad_sq,
    )

    assert prnorm.mean_x.shape[0] == n_dim, \
        'Mean has wrong dimension'
    assert torch.allclose(prnorm.mean_x, mean_x), \
        'Mean is not initialized correctly'

    # Check B is correctly initialized
    B = prnorm.B.detach().clone()
    B_eigval, B_eigvec = torch.linalg.eigh(B)

    B_eigval_expected = torch.sort(
      torch.cat((eigvals, torch.ones(n_dim - 2) * rad_sq))
    )[0]

    assert torch.allclose(B_eigval, B_eigval_expected), \
        'Eigenvalues are not initialized correctly'

    inner_prod = torch.abs(eigvecs @ B_eigvec)
    assert torch.allclose(inner_prod[inner_prod > 1e-3], torch.tensor([1.0, 1.0])), \
        'Eigenvectors are not initialized correctly'

    # Check that value error is raised if n_dim doesn't match the statistics
    with pytest.raises(ValueError):
        prnorm = models.ProjNormalEllipse(
          n_dim=n_dim,
          B_eigvecs=par_samp.make_ortho_vectors(n_dim+1, N_DIRS),
        )


######### TEST BASIC METHODS

@pytest.mark.parametrize('n_dim', [3, 7])
@pytest.mark.parametrize('sigma', [0.1])
def test_empirical_moments(n_dim, gaussian_parameters):
    """Test the sampling of the ProjNormal class."""
    n_samples = 200000

    # Unpack parameters
    mean_x = gaussian_parameters['mean_x']
    covariance_x = gaussian_parameters['covariance_x']

    eigvecs = par_samp.make_ortho_vectors(n_dim, N_DIRS)
    eigvals = torch.tensor([0.4, 2.5])
    rad_sq = torch.tensor(1.0)

    # Initialize the projected normal class
    prnorm = models.ProjNormalEllipse(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B_eigvecs=eigvecs,
      B_eigvals=eigvals,
      B_rad_sq=rad_sq,
    )

    # Sample using the class
    emp_moments_class = prnorm.moments_empirical(n_samples)

    # Sample using the function
    B = prnorm.B.detach().clone()
    emp_moments_other = pne.sampling.empirical_moments(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B=B,
      n_samples=n_samples
    )

    # Compare results
    assert torch.allclose(emp_moments_class['mean'], emp_moments_other['mean'], atol=TOLERANCE), \
        'Class empirical moments not correct'
    assert torch.allclose(emp_moments_class['second_moment'], emp_moments_other['second_moment'], atol=TOLERANCE), \
        'Class empirical moments not correct'


@pytest.mark.parametrize('n_dim', [3, 10])
@pytest.mark.parametrize('sigma', [0.1])
def test_moments(n_dim, gaussian_parameters):
    """Test the moment computation of the ProjNormal class."""
    # Unpack parameters
    mean_x = gaussian_parameters['mean_x']
    covariance_x = gaussian_parameters['covariance_x']

    eigvecs = par_samp.make_ortho_vectors(n_dim, N_DIRS)
    eigvals = torch.tensor([0.4, 2.5])
    rad_sq = torch.tensor(1.0)

    prnorm = models.ProjNormalEllipse(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B_eigvecs=eigvecs,
      B_eigvals=eigvals,
      B_rad_sq=rad_sq,
    )

    # Sample using the class
    with torch.no_grad():
        moments_class = prnorm.moments()

    # Sample using the function
    B = prnorm.B.detach().clone()
    gamma = pne.moments.mean(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B=B,
    )
    second_moment = pne.moments.second_moment(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B=B,
    )

    # Compare results
    assert torch.allclose(moments_class['mean'], gamma), \
        'Class empirical moments not correct'
    assert torch.allclose(moments_class['second_moment'], second_moment), \
        'Class empirical moments not correct'

