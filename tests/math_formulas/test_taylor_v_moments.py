"""Test the computation of the moments of auxiliary variable in Taylor approximation."""
import pytest
import torch

import projnormal.formulas.projected_normal_c as pnc_formulas
import projnormal.param_sampling as par_samp
import projnormal.quadratic_forms as qf


def v_moments_naive(mean_x, covariance_x, method='analytic'):
    """Compute v moments in naive more expensive way."""
    n_dim = mean_x.shape[0]
    # Compute naively
    v_mean_naive = torch.zeros(n_dim)
    v_var = torch.zeros(n_dim)
    v_cov = torch.zeros(n_dim)

    # For each i, keep only the elements that are not i, and
    # compute the mean and variance of the resulting quadratic form
    for i in range(n_dim):
        # Get the indices of non-i elements
        keep_inds = list(range(n_dim))
        keep_inds.remove(i)

        # Remove i-th element from mean, covariance and B_diagonal
        covariance_sub = covariance_x.clone()
        covariance_sub = covariance_sub[keep_inds,:]
        covariance_sub = covariance_sub[:,keep_inds]
        mean_sub = mean_x[keep_inds]

        # Mean
        v_mean_naive[i] = qf.moments.mean(
          mean_x=mean_sub,
          covariance_x=covariance_sub,
        )
        # Variance
        v_var[i] = qf.moments.variance(
          mean_x=mean_sub,
          covariance_x=covariance_sub,
        )

        # Covariance
        a = torch.zeros(n_dim)  # Linear form vector
        a[i] = 1  # Set the i-th element to 1
        A = torch.eye(n_dim)  # Quadratic form matrix
        A[i, i] = 0
        v_cov[i] = qf.moments.qf_linear_covariance(
          mean_x=mean_x, covariance_x=covariance_x, M=A, b=a
        )

    return v_mean_naive, v_var, v_cov


# Fixture to set up the parameters and compute the moments naively
@pytest.fixture(scope='function')
def taylor_moments_data(n_dim, sigma):  # Add 'request' as a parameter
    """Fixture to set up the parameters and compute the moments naively."""
    # Tolerance
    tolerance = 0.001

    # Instantiate parameters
    mean_x = par_samp.make_mean(
      n_dim=n_dim, shape='sin'
    )
    covariance_x = par_samp.make_spdm(n_dim=n_dim) * sigma**2

    # Compute moments of auxiliary variables v_i
    v_mean_naive, v_var, v_cov = v_moments_naive(
      mean_x, covariance_x
    )

    return {
        'mean_x': mean_x,
        'covariance_x': covariance_x,
        'v_mean': v_mean_naive,
        'v_var': v_var,
        'v_cov': v_cov,
        'tolerance': tolerance
    }


@pytest.mark.parametrize('n_dim', [2, 3, 10])
@pytest.mark.parametrize('sigma', [0.1, 1])
@pytest.mark.parametrize('cov_type', ['random', 'diagonal'])
def test_taylor_v_variable_moments(taylor_moments_data, n_dim, sigma, cov_type):
    """Test the computation of the moments of auxiliary variable in Taylor approximation."""
    # Unpack data

    # Distribution parameters
    mean_x = taylor_moments_data['mean_x']
    covariance_x = taylor_moments_data['covariance_x']
    tolerance = taylor_moments_data['tolerance']

    # Naive computation results
    v_mean_naive = taylor_moments_data['v_mean']
    v_var_naive = taylor_moments_data['v_var']
    v_cov_naive = taylor_moments_data['v_cov']

    # Efficient computation results
    v_mean = pnc_formulas.moments._get_v_mean(
      mean_x=mean_x, covariance_x=covariance_x
    )
    v_var = pnc_formulas.moments._get_v_var(
      mean_x=mean_x, covariance_x=covariance_x
    )
    v_cov = pnc_formulas.moments._get_v_cov(
      mean_x=mean_x, covariance_x=covariance_x
    )

    # Compute the error
    error_mean = torch.max(torch.abs(v_mean - v_mean_naive))
    error_var = torch.max(torch.abs(v_var - v_var_naive))
    error_cov = torch.max(torch.abs(v_cov - v_cov_naive))

    # Print and assert
    print(f'Error in computing the mean of V = {error_mean}')
    print(f'Error in computing the variance of V = {error_var}')
    print(f'Error in computing the covariance of V = {error_cov}')

    assert error_mean < tolerance, \
        'Error in computing the mean of V is too large'
    assert error_var < tolerance, \
        'Error in computing the variance of V is too large'
    assert error_cov < tolerance, \
        'Error in computing the covariance of V is too large'

