"""Test the Taylor approximation to ellipsoid projected normal moments."""
import pytest
import torch
import projnormal.distribution.ellipse as pne
import projnormal.distribution.general as png
import projnormal.param_sampling as par_samp
import projnormal.matrix_checks as checks


@pytest.fixture(scope="function")
def sample_parameters(n_dim, sigma):
    """ Fixture to generate Gaussian parameters for tests."""

    # Initialize the mean of the gaussian
    # Parameters of distribution
    mean_x = par_samp.make_mean(n_dim=n_dim)
    covariance_x = par_samp.make_spdm(n_dim=n_dim) * sigma**2
    B = torch.diag(torch.rand(n_dim) + 0.5)

    return {
        "mean_x": mean_x,
        "covariance_x": covariance_x,
        "B": B,
    }

@pytest.mark.parametrize("n_dim", [5])
@pytest.mark.parametrize("sigma", [0.2])
def test_taylor_moments(sample_parameters):
    n_samples = 500000

    # Unpack parameters
    mean_x = sample_parameters["mean_x"]
    covariance_x = sample_parameters["covariance_x"]
    B = sample_parameters["B"]

    # Get taylor approximation moments
    gamma_taylor = pne.moments.mean(
      mean_x=mean_x, covariance_x=covariance_x, B=B
    )
    sm_taylor = pne.moments.second_moment(
        mean_x=mean_x, covariance_x=covariance_x, B=B
    )

    # Check that the means are close
    assert checks.is_symmetric(sm_taylor)
    assert checks.is_positive_definite(sm_taylor)

    moments = pne.sampling.empirical_moments(
        mean_x=mean_x, covariance_x=covariance_x,
        B=B, n_samples=n_samples
    )
    gamma_emp = moments["mean"]
    sm_emp = moments["second_moment"]

    assert torch.allclose(gamma_taylor, gamma_emp, atol=1e-2)
    assert torch.allclose(sm_taylor, sm_emp, atol=1e-2)


@pytest.mark.parametrize("n_dim", [5])
@pytest.mark.parametrize("sigma", [0.2])
def test_pdf(n_dim, sample_parameters):
    """Test the pdf of the ellipse distribution."""
    n_samples = 500

    # Unpack parameters
    mean_x = sample_parameters["mean_x"]
    covariance_x = sample_parameters["covariance_x"]
    B = sample_parameters["B"]

    # Get samples from the ellipse distribution
    samples_ellipse = pne.sampling.sample(
        mean_x=mean_x, covariance_x=covariance_x, B=B, n_samples=n_samples
    )

    # Compute pdf
    pdf_samples = pne.pdf.pdf(
      mean_x=mean_x, covariance_x=covariance_x, y=samples_ellipse, B=B
    )

    assert torch.all(pdf_samples >= 0)

    # Check that the values are the same when B=I
    B = torch.eye(mean_x.shape[0])

    samples = png.sampling.sample(
        mean_x=mean_x, covariance_x=covariance_x, n_samples=n_samples
    )

    # Compute pdf
    pdf_gen = png.pdf.pdf(
      mean_x=mean_x, covariance_x=covariance_x, y=samples,
    )

    pdf_ellipse = pne.pdf.pdf(
      mean_x=mean_x, covariance_x=covariance_x, y=samples, B=B
    )

    assert torch.allclose(pdf_gen, pdf_ellipse, atol=1e-2)

    # Check that values are just scaled when B=a*I
    # Original variable satisfies y'y = 1
    # Scale variable by q=yA and make it satisfy
    # q'(A^{-1}A^{-1})q = y'y = 1
    # In transformed space, area of sphere is scaled
    # a^(n-1), so pdf is scaled by 1/a^(1-n)

    # Sample from sphere
    samples = png.sampling.sample(
        mean_x=mean_x, covariance_x=covariance_x, n_samples=n_samples
    )
    # Compute pdf in sphere
    pdf_gen = png.pdf.pdf(
      mean_x=mean_x, covariance_x=covariance_x, y=samples,
    )

    # Scale up the sphere we're projecting to
    a = 2
    scaling = torch.eye(mean_x.shape[0]) * a
    B = scaling.inverse() @ scaling.inverse()

    samples_trans = samples @ scaling
    pdf_ellipse = pne.pdf.pdf(
      mean_x=mean_x, covariance_x=covariance_x, y=samples_trans, B=B
    )

    assert torch.allclose(
      pdf_ellipse * a**(n_dim-1), pdf_gen, atol=1e-2
    )

