"""Test moments of isotropic projected normal distribution."""

import pytest
import torch

import projnormal.formulas.projected_normal_iso as pniso_formulas
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
    moments_empirical = pniso_formulas.empirical_moments(
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
    """Test moments of isotropic projected normal distribution."""

    def test_mean_error(self, iso_data):
        """Test the mean of isotropic projected normal distribution."""
        # unpack data
        mean_x = iso_data['mean_x']
        sigma = iso_data['sigma']
        mean_empirical = iso_data['mean_empirical']
        # Get analytical estimate
        var_x = torch.as_tensor(sigma**2)
        mean_analytic = pniso_formulas.mean(
          mean_x=mean_x, var_x=var_x
        )
        # Check error
        mean_error = relative_error(
          mean_empirical, mean_analytic
        )
        assert mean_error < iso_data['tolerance']

    def test_second_moment_error(self, iso_data):
        """Test the second moment of isotropic projected normal distribution."""
        # unpack data
        mean_x = iso_data['mean_x']
        sigma = iso_data['sigma']
        second_m_empirical = iso_data['second_moment_empirical']
        # Get analytical estimate
        var_x = torch.as_tensor(sigma**2)
        second_m_analytic = pniso_formulas.second_moment(mean_x=mean_x, var_x=var_x)
        # Check error
        second_m_error = relative_error(
          second_m_empirical, second_m_analytic
        )
        assert second_m_error < iso_data['tolerance']
