##################
#
# TESTS THE MATHEMATICAL COMPUTATIONS IN THE TAYLOR APPROXIMATION
#
##################

import time
import pytest
import torch
import projected_normal.prnorm_func as pnf
import projected_normal.qf_func as qf
from utility_functions import make_mu, make_covariance


# Make function to compute the moments of the auxiliary variable the naive way,
# by removing the i-th element from the mean and covariance, to use for testing

def v_moments_naive(mu, covariance, B_diagonal, method='analytic'):
    n_dim = mu.shape[0]
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
        B_diagonal_sub = B_diagonal[keep_inds]
        covariance_sub = covariance.clone()
        covariance_sub = covariance_sub[keep_inds,:]
        covariance_sub = covariance_sub[:,keep_inds]
        mu_sub = mu[keep_inds]

        # Compute moments with non-i elements
        if method == 'analytic':
            # Mean
            v_mean_naive[i] = qf.quadratic_form_mean_diagonal(
              mu=mu_sub,
              covariance=covariance_sub,
              M_diagonal=B_diagonal_sub
            )
            # Variance
            v_var[i] = qf.quadratic_form_var_diagonal(
              mu=mu_sub,
              covariance=covariance_sub,
              M_diagonal=B_diagonal_sub
            )

        elif method == 'empirical':
            moments = qf.empirical_moments_quadratic_form(
              mu=mu_sub,
              covariance=covariance_sub,
              M=torch.diag(B_diagonal_sub),
              n_samples=100000)
            v_mean_naive[i] = moments['mean']
            v_var[i] = moments['var']

        # Covariance
        a = torch.zeros(n_dim) # Linear form vector
        a[i] = 1 # Set the i-th element to 1
        A = torch.diag(B_diagonal) # Quadratic form matrix
        A[i,i] = 0
        v_cov[i] = qf.quadratic_linear_cov(mu=mu, covariance=covariance, M=A, b=a)

    return v_mean_naive, v_var, v_cov


# Fixture to set up the parameters and compute the moments naively
@pytest.fixture(scope='function')
def taylor_moments_data(n_dim, sigma, cov_type):  # Add 'request' as a parameter

    # Tolerance
    tolerance = 0.01

    # Make the B_diagonal of the quadratic form (diagonal matrix B)
    i = torch.arange(n_dim)
    B_diagonal = torch.exp(-i.float() * 2 / n_dim)
    #B_diagonal = torch.ones(n_dim)

    # Instantiate parameters
    mu = make_mu(n_dim=n_dim, mu_type='sin')
    covariance = make_covariance(n_dim=n_dim, cov_scale=sigma, cov_type=cov_type)
    variances = torch.diag(covariance)

    # Compute moments of auxiliary variables v_i
    start = time.time()
    v_mean_naive, v_var, v_cov = v_moments_naive(mu, covariance, B_diagonal)
    end = time.time()

    return {
        'mu': mu,
        'covariance': covariance,
        'B_diagonal': B_diagonal,
        'v_mean': v_mean_naive,
        'v_var': v_var,
        'v_cov': v_cov,
        'time_naive': end - start,
        'tolerance': tolerance
    }


@pytest.mark.parametrize('n_dim', [2, 3, 5, 10, 20, 50])
@pytest.mark.parametrize('sigma', [0.01, 0.1, 1, 5])
@pytest.mark.parametrize('cov_type', ['random', 'diagonal'])
def test_taylor_v_variable_moments(taylor_moments_data, n_dim, sigma, cov_type):
    # Unpack data

    # Distribution parameters
    mu = taylor_moments_data['mu']
    covariance = taylor_moments_data['covariance']
    B_diagonal = taylor_moments_data['B_diagonal']
    tolerance = taylor_moments_data['tolerance']

    # Naive computation results
    v_mean_naive = taylor_moments_data['v_mean']
    v_var_naive = taylor_moments_data['v_var']
    v_cov_naive = taylor_moments_data['v_cov']
    time_naive = taylor_moments_data['time_naive']

    # Efficient computation results
    start = time.time()
    v_mean = pnf.get_v_mean(mu=mu, covariance=covariance, B_diagonal=B_diagonal)
    v_var = pnf.get_v_var(mu=mu, covariance=covariance, B_diagonal=B_diagonal)
    v_cov = pnf.get_v_cov(mu=mu, covariance=covariance, B_diagonal=B_diagonal)
    end = time.time()
    time_efficient = end - start

    # Compute the relative error
    error_mean = torch.max(torch.abs(v_mean - v_mean_naive))
    error_var = torch.max(torch.abs(v_var - v_var_naive))
    error_cov = torch.max(torch.abs(v_cov - v_cov_naive))

    # Print and assert
    print(f'Error in computing the mean of V = {error_mean}')
    print(f'Error in computing the variance of V = {error_var}')
    print(f'Error in computing the covariance of V = {error_cov}')

    assert error_mean < tolerance, 'Error in computing the mean of V is too large'
    assert error_var < tolerance, 'Error in computing the variance of V is too large'
    assert error_cov < tolerance, 'Error in computing the covariance of V is too large'
    #assert time_efficient < time_naive, 'Efficient computation is slower than naive computation'

