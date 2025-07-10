"""Test the formulas of the Taylor approximation to the moments."""
import pytest
import torch

import projnormal.formulas.projected_normal_c as pnc_formulas
import projnormal.param_sampling as par_samp


def relative_error(x, y):
    """Compute the relative error between two tensors."""
    return torch.norm(x - y) * 2 / (torch.norm(y) + torch.norm(x))


# Function of taylor approximation
def f(u, v):
    """F = u/sqrt(b*u^2+v)."""
    return u / (torch.sqrt( u**2 + v))


def d2f_du2_autograd(u, v):
    """Compute derivatives with autograd."""
    df_du = torch.autograd.grad(f(u, v), u, create_graph=True)[0]
    d2f_du2 = torch.autograd.grad(df_du, u)[0]
    return d2f_du2


def d2f_dv2_autograd(u, v):
    """Compute derivatives with autograd."""
    df_dv = torch.autograd.grad(f(u, v), v, create_graph=True)[0]
    d2f_dv2 = torch.autograd.grad(df_dv, v)[0]
    return d2f_dv2


def d2f_dudv_autograd(u, v):
    """Compute derivatives with autograd."""
    df_du = torch.autograd.grad(f(u, v), u, create_graph=True)[0]
    d2f_dudv = torch.autograd.grad(df_du, v)[0]
    return d2f_dudv


# Fixture to set up the parameters and compute gradients
@pytest.fixture(scope='function')
def taylor_derivatives_data(n_dim, sigma, cov_type):
    """Sample parameters and compute the taylor derivatives using autograd."""
    # Instantiate parameters
    mean_x = par_samp.make_mean(
      n_dim=n_dim, shape='sin'
    )
    covariance_x = par_samp.make_spdm(n_dim=n_dim) * torch.as_tensor(sigma**2)
    # Compute mean of auxiliary variables
    v_mean = pnc_formulas.moments._get_v_mean(
      mean_x=mean_x, covariance_x=covariance_x
    )

    # Compute the derivatives using autograd
    du2_autograd = torch.zeros(n_dim)
    dv2_autograd = torch.zeros(n_dim)
    dudv_autograd = torch.zeros(n_dim)
    for i in range(n_dim):
        x = mean_x[i].clone().detach().requires_grad_(True)
        y = v_mean[i].clone().detach().requires_grad_(True)
        du2_autograd[i] = d2f_du2_autograd(x, y)
        dv2_autograd[i] = d2f_dv2_autograd(x, y)
        dudv_autograd[i] = d2f_dudv_autograd(x, y)

    return {
        'n_dim': n_dim,
        'mean_x': mean_x,
        'v_mean': v_mean,
        'du2_autograd': du2_autograd,
        'dv2_autograd': dv2_autograd,
        'dudv_autograd': dudv_autograd
    }


@pytest.mark.parametrize('n_dim', [2, 3, 5, 10, 50])
@pytest.mark.parametrize('sigma', [0.01, 0.5, 1])
@pytest.mark.parametrize('cov_type', ['random', 'diagonal'])
def test_taylor_approximation_derivatives(taylor_derivatives_data, n_dim, sigma, cov_type):
    """Test the Taylor approximation derivatives against autograd results."""
    # Unpack data

    # Distribution parameters
    mean_x = taylor_derivatives_data['mean_x']
    v_mean = taylor_derivatives_data['v_mean']

    # Autograd results
    du2_autograd = taylor_derivatives_data['du2_autograd']
    dv2_autograd = taylor_derivatives_data['dv2_autograd']
    dudv_autograd = taylor_derivatives_data['dudv_autograd']

    # Compute derivatives using the function being tested
    du2 = pnc_formulas.moments._get_dfdu2(u=mean_x, v=v_mean)
    dv2 = pnc_formulas.moments._get_dfdv2(u=mean_x, v=v_mean)
    dudv = pnc_formulas.moments._get_dfdudv(u=mean_x, v=v_mean)

    # Compute the relative error
    du2_error = relative_error(du2, du2_autograd)
    dv2_error = relative_error(dv2, dv2_autograd)
    dudv_error = relative_error(dudv, dudv_autograd)

    # Print and assert
    print(f'Error in d2f/du2 = {du2_error}')
    # Assertions
    assert du2_error < 1e-5, f'Error in d2f/du2 is too large: {du2_error}'
    assert dv2_error < 1e-5, f'Error in d2f/dv2 is too large: {dv2_error}'
    assert dudv_error < 1e-5, f'Error in d2f/dudv is too large: {dudv_error}'

