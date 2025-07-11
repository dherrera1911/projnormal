"""Test the projected normal class."""
import pytest
import torch

import projnormal.classes as classes
import projnormal.formulas.projected_normal_B as pnb_formulas
import projnormal.param_sampling as par_samp

torch.manual_seed(1)
TOLERANCE = 0.025

# Instantiate parameters, get empirical moments
@pytest.fixture(scope="function")
def gaussian_parameters(n_dim, sigma):
    """Fixture to generate Gaussian parameters for tests."""
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
    # Initialize parameters
    mean_x = torch.ones(n_dim) / torch.sqrt(torch.as_tensor(n_dim))
    covariance_x = torch.eye(n_dim)
    B = par_samp.make_spdm(n_dim)

    prnorm = classes.ProjNormalEllipse(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B=B,
    )

    assert prnorm.mean_x.shape[0] == n_dim, \
        'Mean has wrong dimension'
    assert torch.allclose(prnorm.mean_x, mean_x), \
        'Mean is not initialized correctly'

    # Check B is correctly initialized
    B_init = prnorm.B.detach().clone()

    assert torch.allclose(B_init, B, atol=1e-6), \
        'B is not initialized correctly'


######### TEST BASIC METHODS

@pytest.mark.parametrize('n_dim', [3, 7])
@pytest.mark.parametrize('sigma', [0.1])
def test_empirical_moments(n_dim, gaussian_parameters):
    """Test the sampling of the ProjNormal class."""
    n_samples = 200000

    # Unpack parameters
    mean_x = gaussian_parameters['mean_x']
    covariance_x = gaussian_parameters['covariance_x']
    B = par_samp.make_spdm(n_dim)

    # Initialize the projected normal class
    prnorm = classes.ProjNormalEllipse(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B=B,
    )

    # Sample using the class
    emp_moments_class = prnorm.moments_empirical(n_samples)

    # Sample using the function
    B2 = prnorm.B.detach().clone()
    emp_moments_other = pnb_formulas.empirical_moments(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B=B2,
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
    B = par_samp.make_spdm(n_dim)

    prnorm = classes.ProjNormalEllipse(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B=B,
    )

    # Sample using the class
    with torch.no_grad():
        moments_class = prnorm.moments()

    # Sample using the function
    B2 = prnorm.B.detach().clone()
    gamma = pnb_formulas.mean(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B=B2,
    )
    second_moment = pnb_formulas.second_moment(
      mean_x=mean_x,
      covariance_x=covariance_x,
      B=B2,
    )

    # Compare results
    assert torch.allclose(moments_class['mean'], gamma), \
        'Class taylor mean not correct'
    assert torch.allclose(moments_class['second_moment'], second_moment, atol=TOLERANCE), \
        'Class taylor second moment not correct'

    # Compare to empirical
    #n_samples = 1000000
    #emp_moments = prnorm.moments_empirical(n_samples)
