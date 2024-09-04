##################
#
# TESTS THE MATHEMATICAL COMPUTATIONS IN THE TAYLOR APPROXIMATION
#
##################

import pytest
import torch
import projected_normal.prnorm_func as pnf
from utility_functions import make_mu, make_covariance

# Function of taylor approximation
def f(u, v, b=1, c50=0):
    ' f = u/sqrt(b*u^2+v) '
    return u / ( torch.sqrt( b*u**2 + v + c50 ) )

# Function to compute d2f/du2 using autograd
def d2f_du2_autograd(u, v, b=1, c50=0):
    df_du = torch.autograd.grad(f(u, v, b, c50), u, create_graph=True)[0]
    d2f_du2 = torch.autograd.grad(df_du, u)[0]
    return d2f_du2

# Function to compute d2f/dv2 using autograd
def d2f_dv2_autograd(u, v, b=1, c50=0):
    df_dv = torch.autograd.grad(f(u, v, b, c50), v, create_graph=True)[0]
    d2f_dv2 = torch.autograd.grad(df_dv, v)[0]
    return d2f_dv2

# Function to compute d2f/dudv using autograd
def d2f_dudv_autograd(u, v, b=1, c50=0):
    df_du = torch.autograd.grad(f(u, v, b, c50), u, create_graph=True)[0]
    d2f_dudv = torch.autograd.grad(df_du, v)[0]
    return d2f_dudv


# Fixture to set up the parameters and compute gradients
@pytest.fixture(scope='function')
def taylor_derivatives_data(n_dim, sigma, c50, cov_type):  # Add 'request' as a parameter

    # Make the weights of the quadratic form (diagonal matrix B)
    i = torch.arange(n_dim)
    B_diagonal = torch.exp(-i.float() * 2 / n_dim)

    # Instantiate parameters
    mu = make_mu(n_dim=n_dim, mu_type='sin')
    covariance = make_covariance(n_dim=n_dim, cov_scale=sigma, cov_type=cov_type)
    variances = torch.diag(covariance)

    # Compute mean of auxiliary variables
    v_mean = pnf.get_v_mean(mu=mu, covariance=covariance, B_diagonal=B_diagonal)

    # Compute the derivatives using autograd
    du2_autograd = torch.zeros(n_dim)
    dv2_autograd = torch.zeros(n_dim)
    dudv_autograd = torch.zeros(n_dim)
    for i in range(n_dim):
        x = mu[i].clone().detach().requires_grad_(True)
        y = v_mean[i].clone().detach().requires_grad_(True)
        du2_autograd[i] = d2f_du2_autograd(x, y, B_diagonal[i], c50)
        dv2_autograd[i] = d2f_dv2_autograd(x, y, B_diagonal[i], c50)
        dudv_autograd[i] = d2f_dudv_autograd(x, y, B_diagonal[i], c50)

    return {
        'n_dim': n_dim,
        'mu': mu,
        'v_mean': v_mean,
        'c50': c50,
        'B_diagonal': B_diagonal,
        'du2_autograd': du2_autograd,
        'dv2_autograd': dv2_autograd,
        'dudv_autograd': dudv_autograd
    }

@pytest.mark.parametrize('n_dim', [2, 3, 5, 10, 50])
@pytest.mark.parametrize('sigma', [0.01, 0.5, 1])
@pytest.mark.parametrize('c50', [0, 0.1, 1])
@pytest.mark.parametrize('cov_type', ['random', 'diagonal'])
def test_taylor_approximation_derivatives(taylor_derivatives_data, n_dim, sigma, c50, cov_type):
    # Unpack data

    # Distribution parameters
    mu = taylor_derivatives_data['mu']
    v_mean = taylor_derivatives_data['v_mean']
    B_diagonal = taylor_derivatives_data['B_diagonal']
    c50 = taylor_derivatives_data['c50']

    # Autograd results
    du2_autograd = taylor_derivatives_data['du2_autograd']
    dv2_autograd = taylor_derivatives_data['dv2_autograd']
    dudv_autograd = taylor_derivatives_data['dudv_autograd']

    # Compute derivatives using the function being tested
    du2 = pnf.dfdu2(u=mu, v=v_mean, b=B_diagonal, c50=c50)
    dv2 = pnf.dfdv2(u=mu, v=v_mean, b=B_diagonal, c50=c50)
    dudv = pnf.dfdudv(u=mu, v=v_mean, b=B_diagonal, c50=c50)

    # Compute the relative error
    du2_error = torch.norm(du2 - du2_autograd) / torch.norm(du2_autograd)
    dv2_error = torch.norm(dv2 - dv2_autograd) / torch.norm(dv2_autograd)
    dudv_error = torch.norm(dudv - dudv_autograd) / torch.norm(dudv_autograd)

    # Print and assert
    print(f'Error in d2f/du2 = {du2_error}')
    # Assertions
    assert du2_error < 1e-5, f'Error in d2f/du2 is too large: {du2_error}'
    assert dv2_error < 1e-5, f'Error in d2f/dv2 is too large: {dv2_error}'
    assert dudv_error < 1e-5, f'Error in d2f/dudv is too large: {dudv_error}'

