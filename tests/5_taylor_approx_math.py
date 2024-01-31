##################
#
# TESTS FOR THE FUNCTIONS THAT COMPUTE MOMENTS
# OF THE PROJECTED NORMAL DISTRIBUTION
#
##################

#import pytest
import torch
import numpy as np
import qr_library as qr
from einops import reduce
from test_functions import *

####################################
####### INDEPENDENT NOISE CASE #####
#######  TAYLOR APPROXIMATION ######
####################################

# Parameters of distribution
sigma = 1
muMult = 4

# Instantiate parameters
mu = make_mu_sin(nDim=nDim) * muMult
mu = torch.abs(mu)
covariance = make_covariance(nDim=nDim, sigmaScale=sigma, covType=covType)
variances = torch.diag(covariance)
B = torch.eye(nDim)

## TEST THE MATH OF THE TAYLOR APPROXIMATION

# Compute the mean and variance of the variable V = X^2 - Xi^2
# for each Xi
meanX2 = qr.quadratic_form_mean(mu=mu, covariance=covariance, M=B)
meanXi2 = mu**2 + variances
meanV = meanX2 - meanXi2
varX2 = qr.quadratic_form_var(mu=mu, covariance=covariance, M=B)
varSubtract = 2 * variances**2 + 4 * mu**2 * variances

## TEST THAT THE METHOD FOR COMPUTING DERIVATIVES WORKS
def f(x, y):
    " f = x/sqrt(x^2+y) "
    return x/(torch.sqrt(x**2+y))

# Function to compute the second derivative of f with respect to x
# using the autograd package
def d2f_dx2_autograd(x, y):
    # First derivative with respect to x
    df_dx = torch.autograd.grad(f(x, y), x, create_graph=True)[0]
    # Second derivative with respect to x (the gradient of df_dx with respect to x)
    d2f_dx2 = torch.autograd.grad(df_dx, x)[0]
    return d2f_dx2

# Function to compute the second derivative of f with respect to y
# using the autograd package
def d2f_dy2_autograd(x, y):
    # First derivative with respect to y
    df_dy = torch.autograd.grad(f(x, y), y, create_graph=True)[0]
    # Second derivative with respect to y (the gradient of df_dy with respect to y)
    d2f_dy2 = torch.autograd.grad(df_dy, y)[0]
    return d2f_dy2

du2_autograd = torch.zeros(nDim)
dv2_autograd = torch.zeros(nDim)

for i in range(nDim):
    x = mu[i].clone().detach().requires_grad_(True)
    y = meanV[i].clone().detach().requires_grad_(True)
    # Compute the derivative of f with respect to x
    du2_autograd[i] = d2f_dx2_autograd(x, y)
    # Compute the derivative of f with respect to y
    dv2_autograd[i] = d2f_dy2_autograd(x, y)

# Function to compute the second derivative of f with respect to x
du2_fun = qr.proj_normal_dx2(u=mu, v=meanV)
dv2_fun = qr.proj_normal_dv2(u=mu, v=meanV)

# Compute the error
error = torch.max(torch.abs(du2_autograd - du2_fun))
error2 = torch.max(torch.abs(dv2_autograd - dv2_fun))

print(f'Error in computing the second derivative of f with respect to x = {error}')
print(f'Error in computing the second derivative of f with respect to y = {error2}')


