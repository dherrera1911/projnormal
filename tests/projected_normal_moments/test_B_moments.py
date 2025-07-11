"""Test transform/untransform approach to approximate X/X'BX moments."""
import pytest
import torch

import projnormal.formulas.projected_normal_Bc as pnbc_formulas
import projnormal.formulas.projected_normal_c as pnc_formulas
import projnormal.param_sampling as par_samp

# Make double the default precision
torch.set_default_dtype(torch.float64)

torch.manual_seed(0)
tol_gamma = 3e-2 # Numerical difference between cholesky and B_sqrt is large
tol_sm = 1e-4

# Instantiate parameters
@pytest.fixture(scope="function")
def gaussian_parameters(n_dim, mean_type, eigvals, eigvecs, sigma, const):
    """Fixture to generate Gaussian parameters for tests."""
    # Initialize the mean of the gaussian
    # Parameters of distribution
    mean_x = par_samp.make_mean(
      n_dim=n_dim, shape=mean_type
    )
    covariance_x = par_samp.make_spdm(
      n_dim=n_dim, eigvals=eigvals, eigvecs=eigvecs
    ) * sigma**2
    B_sqrt = par_samp.make_spdm(
      n_dim=n_dim
    )

    return {
        "mean_x": mean_x,
        "covariance_x": covariance_x,
        "const": const,
        "B_sqrt": B_sqrt,
    }


######### CHECK THAT THE OUTPUTS ARE NUMERICALLY VALID ############
@pytest.mark.parametrize("n_dim", [2, 3, 5, 10])
@pytest.mark.parametrize("mean_type", ["sin", "sparse"])
@pytest.mark.parametrize("eigvals", ["exponential"])
@pytest.mark.parametrize("eigvecs", ["random", "identity"])
@pytest.mark.parametrize("sigma", [0.1])
@pytest.mark.parametrize("const", [0, 0.1])
def test_B_mapping(gaussian_parameters):
    """Test the transform/untransform approach to approximate X/X'BX moments."""
    # Unpack parameters
    mean_x = gaussian_parameters["mean_x"]
    covariance_x = gaussian_parameters["covariance_x"]
    const = gaussian_parameters["const"]
    B_sqrt = gaussian_parameters["B_sqrt"]

    B = B_sqrt @ B_sqrt
    B_sqt_inv = torch.linalg.inv(B_sqrt)

    # Get taylor approximation moments to Y=X'BX with package functions
    gamma_taylor = pnbc_formulas.mean(
      mean_x=mean_x, covariance_x=covariance_x, const=const, B=B,
    )
    sm_taylor = pnbc_formulas.second_moment(
        mean_x=mean_x, covariance_x=covariance_x, const=const, B=B
    )

    # Compute approximations manually
    # First compute moments in transformed space
    mean_z = mean_x @ B_sqrt
    covariance_z = B_sqrt @ covariance_x @ B_sqrt.T
    gamma_prime = pnc_formulas.mean(
      mean_x=mean_z, covariance_x=covariance_z, const=const
    )
    sm_prime = pnc_formulas.second_moment(
        mean_x=mean_z, covariance_x=covariance_z, const=const
    )
    # Now transform back to original space
    gamma_manual = B_sqt_inv @ gamma_prime
    sm_manual = B_sqt_inv @ sm_prime @ B_sqt_inv.T

    # Check that outputs are not nan, and matrices are as expected
    diff = gamma_taylor - gamma_manual
    assert torch.norm(diff) / torch.norm(gamma_manual) < tol_gamma, \
        "Mismatch in mean for X/X'BX"
    assert torch.allclose(
        sm_taylor, sm_manual, atol=tol_sm
    ), "Mismatch in second moment for X/X'BX"
