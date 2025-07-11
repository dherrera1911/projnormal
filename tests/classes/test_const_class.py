"""Test the const projected normal class."""
import pytest
import torch

import projnormal._utils._matrix_checks as checks
import projnormal.classes as classes
import projnormal.formulas.projected_normal_c as pnc_formulas
import projnormal.param_sampling as par_samp

torch.manual_seed(1)
TOLERANCE = 0.025
MAX_ITER = 30

def norm_leq_1(gamma):
    """Check if the norm of gamma is less than or equal to 1."""
    return torch.norm(gamma) <= 1

# Instantiate parameters, get empirical moments
@pytest.fixture(scope="function")
def gaussian_parameters(n_dim, mean_type, sigma):
    """Fixture to generate Gaussian parameters for tests."""
    # Initialize the mean of the gaussian
    # Parameters of distribution
    mean_x = par_samp.make_mean(
      n_dim=n_dim, shape=mean_type
    )
    covariance_x = par_samp.make_spdm(
      n_dim=n_dim
    ) * sigma**2
    const = torch.rand(1).squeeze() + 0.05

    return {
        "mean_x": mean_x,
        "covariance_x": covariance_x,
        "const": const
    }


######### TEST INITIALIZATION

@pytest.mark.parametrize('n_dim', [2, 3, 10])
def test_init(n_dim):
    """Test the initialization of the ProjNormalConst class."""
    # Initialize without input parameters
    prnorm = classes.ProjNormalConst(n_dim=n_dim)

    # Initialize parameters
    mean_x = torch.ones(n_dim) / torch.sqrt(torch.as_tensor(n_dim))
    covariance_x = torch.eye(n_dim)
    const = torch.tensor(0.1)

    prnorm = classes.ProjNormalConst(
      mean_x=mean_x,
      covariance_x=covariance_x,
      const=const
    )

    assert prnorm.mean_x.shape[0] == n_dim, \
        'Mean has wrong dimension'
    assert torch.allclose(prnorm.mean_x, mean_x), \
        'Mean is not initialized correctly'

    # Check that value error is raised if n_dim doesn't match the statistics
    with pytest.raises(ValueError):
        prnorm = classes.ProjNormalConst(
          n_dim=n_dim,
          const=torch.tensor(-0.01),
        )

######### TEST BASIC METHODS

@pytest.mark.parametrize('n_dim', [2, 3, 10])
@pytest.mark.parametrize('mean_type', ['sparse'])
@pytest.mark.parametrize('sigma', [0.1])
def test_sampling(n_dim, gaussian_parameters):
    """Test the sampling of the ProjNormalConst class."""
    # Unpack parameters
    mean_x = gaussian_parameters['mean_x']
    covariance_x = gaussian_parameters['covariance_x']

    # Initialize the projected normal class
    prnorm = classes.ProjNormalConst(
      mean_x=mean_x,
      covariance_x=covariance_x,
      const=gaussian_parameters['const']
    )
    # Sample from the distribution
    samples = prnorm.sample(n_samples=500)
    # compute the norm of the samples
    norm_samples = torch.norm(samples, dim=1)

    assert torch.all(norm_samples < torch.tensor(1.0)), \
        'Samples norm are not smaller than 1'


@pytest.mark.parametrize('n_dim', [2, 3, 10, 30])
@pytest.mark.parametrize('mean_type', ['sin', 'sparse'])
@pytest.mark.parametrize('sigma', [0.05, 0.1, 1])
def test_moments(n_dim, gaussian_parameters):
    """Test the moment computation of the ProjNormalConst class."""
    # Unpack parameters
    mean_x = gaussian_parameters['mean_x']
    covariance_x = gaussian_parameters['covariance_x']
    const = gaussian_parameters['const']

    # Initialize the projected normal class
    prnorm = classes.ProjNormalConst(
      mean_x=mean_x,
      covariance_x=covariance_x,
      const=const
    )

    # Compute Taylor approximation of the moments
    with torch.no_grad():
        moments_y = prnorm.moments()

    # Check that Taylor moments are not nan
    assert not torch.isnan(moments_y['mean']).any(), \
        'Taylor approximation of the mean is nan'
    assert not torch.isnan(moments_y['second_moment']).any(), \
        'Taylor approximation of the second moment is nan'

    # Check that gamma has norm <= 1
    assert norm_leq_1(moments_y['mean']), \
        'Taylor approximation of the mean has norm > 1'

    # Check that second moment matrix is SPD
    # Taylor covariance
    assert checks.is_symmetric(moments_y['second_moment']), \
        'Taylor approximation of the second moments is not symmetric'
    assert checks.is_positive_definite(moments_y['second_moment']), \
        'Taylor approximation of the second moments is not positive definite'


@pytest.mark.parametrize('n_dim', [2, 3, 10])
@pytest.mark.parametrize('mean_type', ['sin', 'sparse'])
@pytest.mark.parametrize('sigma', [0.05, 0.1])
def test_pdf(n_dim, gaussian_parameters):
    """Test the moment computation of the ProjNormalConst class."""
    # Unpack parameters
    mean_x = gaussian_parameters['mean_x']
    covariance_x = gaussian_parameters['covariance_x']
    const = gaussian_parameters['const']

    # Initialize the projected normal class
    prnorm = classes.ProjNormalConst(
      mean_x=mean_x,
      covariance_x=covariance_x,
      const=const
    )
    # Sample from the distribution
    with torch.no_grad():
        samples = prnorm.sample(100)

    # Compute pdf of the samples
    pdf_samples = prnorm.pdf(samples)

    # Check that the pdf is not nan or inf
    assert not torch.isnan(pdf_samples).any(), \
        'Pdf of the samples is nan'
    assert not torch.isinf(pdf_samples).any(), \
        'Pdf of the samples is inf'


######## TEST FITTING PROCEDURES 

@pytest.mark.parametrize('n_dim', [2, 3])
@pytest.mark.parametrize('mean_type', ['sin'])
@pytest.mark.parametrize('sigma', [0.25])
@pytest.mark.parametrize('optimizer', ['NAdam', 'LBFGS'])
def test_moment_matching(n_dim, optimizer, gaussian_parameters):
    """Test moment matching algorithm."""
    # Unpack parameters
    mean_x = gaussian_parameters['mean_x']
    covariance_x = gaussian_parameters['covariance_x']
    const = gaussian_parameters['const']

    # Make observed moments
    moments_target = pnc_formulas.empirical_moments(
      mean_x=mean_x,
      covariance_x=covariance_x,
      const=const,
      n_samples=10000
    )

    # Initialize the projected normal class
    prnorm = classes.ProjNormalConst(
      n_dim=n_dim
    )

    # Initialize parameters to observed moments
    prnorm.moment_init(moments_target)

    # Fit to the data with moment_matching
    loss = prnorm.moment_match(
      data_moments=moments_target,
      optimizer=optimizer,
      max_epochs=MAX_ITER,
      n_cycles=2,
      cycle_gamma=0.2,
      show_progress=False,
      return_loss=True,
    )
    loss = loss['loss']

    # Get estimated moments
    fit_mean_x = prnorm.mean_x.detach()
    fit_covariance_x = prnorm.covariance_x.detach()

    assert not torch.isnan(loss).any(), 'Loss is nan'
    assert not torch.isnan(fit_mean_x).any(), 'Estimated mu is nan'
    assert not torch.isnan(fit_covariance_x).any(), 'Estimated covariance is nan'
    assert loss[0] > loss[-1], 'Loss did not decrease'
    assert torch.allclose(
        fit_mean_x.norm(), torch.tensor(1.0)
    ), 'Estimated mean norm is not 1'
    assert checks.is_symmetric(fit_covariance_x), 'Estimated covariance is not symmetric'
    assert checks.is_positive_definite(
        fit_covariance_x
    ), 'Estimated covariance is not positive definite'

