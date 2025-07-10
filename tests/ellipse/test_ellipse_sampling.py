"""Test the Taylor approximation to ellipsoid projected normal moments."""
import pytest
import torch
import torch.distributions.multivariate_normal as mvn

import projnormal.formulas.projected_normal_B as pnb_formulas
import projnormal.param_sampling as par_samp


def ellipse_prnorm_sampling(mean_x, covariance_x, B, n_samples=1000):
    """Sample from the projected normal distribution."""
    # Initialize Gaussian distribution to sample from
    dist = mvn.MultivariateNormal(
      loc=mean_x, covariance_matrix=covariance_x
    )
    # Take n_samples
    X = dist.sample([n_samples])
    q = torch.sqrt(torch.einsum("ni,ij,jn->n", X, B, X.t()))
    # Normalize normal distribution samples
    samples_prnorm = torch.einsum("ni,n->ni", X, 1 / q)
    return samples_prnorm


@pytest.fixture(scope="function")
def sample_parameters(n_dim, sigma):
    """Fixture to generate Gaussian parameters for tests."""
    # Initialize the mean of the gaussian
    # Parameters of distribution
    mean_x = par_samp.make_mean(n_dim=n_dim)
    covariance_x = par_samp.make_spdm(n_dim=n_dim) * sigma**2
    B = par_samp.make_spdm(n_dim=n_dim)

    return {
        "mean_x": mean_x,
        "covariance_x": covariance_x,
        "B": B,
    }


@pytest.mark.parametrize("n_dim", [5])
@pytest.mark.parametrize("sigma", [0.2])
def test_ellipse_sampling(sample_parameters):
    """Test that the ellipse sampling matches the package sampling."""
    n_samples = 500000
    # Unpack parameters
    mean_x = sample_parameters["mean_x"]
    covariance_x = sample_parameters["covariance_x"]
    B = sample_parameters["B"]

    samples_pkg = pnb_formulas.sample(
      mean_x, covariance_x, B=B, n_samples=n_samples
    )

    moments_pkg = pnb_formulas.empirical_moments(
      mean_x, covariance_x, B=B, n_samples=n_samples
    )

    samples_ellipse = ellipse_prnorm_sampling(
      mean_x, covariance_x, B=B, n_samples=n_samples
    )

    mean_samples_pkg = samples_pkg.mean(dim=0)
    mean_moments_pkg = moments_pkg["mean"]
    mean_samples_ctrl = samples_ellipse.mean(dim=0)

    # Check that the means are close
    assert torch.allclose(mean_samples_pkg, mean_samples_ctrl, atol=1e-2)
    assert torch.allclose(mean_moments_pkg, mean_samples_ctrl, atol=1e-2)

