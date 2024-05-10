##################
#
# TESTS THE MATHEMATICAL COMPUTATIONS IN THE TAYLOR APPROXIMATION
#
##################

#import pytest
import torch
import numpy as np
import projected_normal as pn
from einops import reduce
from test_functions import *
import time

# Parameters of distribution
nDim = 10
sigma = 1
muMult = 4
c50 = 10
covType = 'random'

# Instantiate parameters
mu = make_mu_sin(nDim=nDim) * muMult
mu = torch.abs(mu)
covariance = make_covariance(nDim=nDim, sigmaScale=sigma, covType=covType)
variances = torch.diag(covariance)
#B = torch.eye()
B = torch.zeros((nDim, nDim))
i = torch.arange(nDim)
B[i,i] = torch.exp(-i.float()*2/nDim)

# Compute the mean the variable V = X^2 - Xi^2 for each Xi
meanV = pn.v_mean(mu=mu, covariance=covariance, weights=B.diagonal())

############################
## TEST FUNCTIONS FOR COMPUTING DERIVATIVES
############################

def f(u, v, b=1, c50=0):
    " f = u/sqrt(b*u^2+v) "
    return u/(torch.sqrt(b*u**2 + v + c50))

# Function to compute d2f/du2 using autograd
def d2f_du2_autograd(u, v, b=1, c50=0):
    # First derivative with respect to x
    df_du = torch.autograd.grad(f(u, v, b, c50), u, create_graph=True)[0]
    # Second derivative with respect to u (the gradient of df_du with respect to u)
    d2f_du2 = torch.autograd.grad(df_du, u)[0]
    return d2f_du2

# Function to compute d2f/dv2 using autograd
def d2f_dv2_autograd(u, v, b, c50=0):
    # First derivative with respect to v
    df_dv = torch.autograd.grad(f(u, v, b, c50), v, create_graph=True)[0]
    # Second derivative with respect to v (the gradient of df_dv with respect to v)
    d2f_dv2 = torch.autograd.grad(df_dv, v)[0]
    return d2f_dv2

# Function to compute d2f/dudv using autograd
def d2f_dudv_autograd(u, v, b, c50=0):
    # First derivative with respect to v
    df_du = torch.autograd.grad(f(u, v, b, c50), u, create_graph=True)[0]
    # Second derivative with respect to v (the gradient of df_dv with respect to v)
    d2f_dudv = torch.autograd.grad(df_du, v)[0]
    return d2f_dudv

# Constant weighting u in f(u,v)
b = 0.5

du2_autograd = torch.zeros(nDim)
dv2_autograd = torch.zeros(nDim)
dudv_autograd = torch.zeros(nDim)

for i in range(nDim):
    x = mu[i].clone().detach().requires_grad_(True)
    y = meanV[i].clone().detach().requires_grad_(True)
    # Compute the second derivatives of f
    du2_autograd[i] = d2f_du2_autograd(x, y, b, c50)
    dv2_autograd[i] = d2f_dv2_autograd(x, y, b, c50)
    dudv_autograd[i] = d2f_dudv_autograd(x, y, b, c50)

# Function to compute the second derivative of f with respect to x
du2_fun = pn.dfdu2(u=mu, v=meanV, b=b, c50=c50)
dv2_fun = pn.dfdv2(u=mu, v=meanV, b=b, c50=c50)
dudv_fun = pn.dfdudv(u=mu, v=meanV, b=b, c50=c50)

# Compute the relative error
error = torch.max(torch.abs((du2_autograd - du2_fun)/(du2_autograd)))
error2 = torch.max(torch.abs((dv2_autograd - dv2_fun)/(dv2_autograd)))
error3 = torch.max(torch.abs((dudv_autograd - dudv_fun)/(dudv_autograd)))

print(f'Error in d2f/du2 = {error}')
print(f'Error in d2f/dv2 = {error2}')
print(f'Error in d2f/dudv = {error3}')


