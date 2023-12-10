import pytest
import torch
import numpy as np
import qr_library as qr
from test_functions import *


def f(x, y):
    " f = x/sqrt(x^2+y) "
    return x/(torch.sqrt(x**2+y))

def d2f_dx2(x, y):
    " Second derivative of f wrt x"
    out = x*(3*x**2/(x**2+y) - 3) / (x**2+y)**(3/2)
    return out

def d2f_dy2(x, y):
    " Second derivative of f wrt y"
    out = 3*x/(4*(x**2+y)**(5/2))
    return out

def rm_el(vec, ind):
    " Remove element from vector"
    out = torch.cat((vec[:ind], vec[ind+1:]))
    return out

def mean_y(mu, var, ind_out):
    " Expected value of y"
    muTemp = rm_el(mu, ind_out)
    varTemp = rm_el(var, ind_out)
    out = torch.sum(muTemp**2) + torch.sum(varTemp)
    return out

def variance_y(mu, var, ind_out):
    " Variance of y"
    muTemp = rm_el(mu, ind_out)
    varTemp = rm_el(var, ind_out)
    out = 2*torch.sum(varTemp**2) + 4*torch.einsum('i,i,i->', muTemp,
                                                 muTemp, varTemp)
    return out

def taylor_approx_Ef(mu, var, ind):
    " Taylor approximation of E[f(x,y)]"
    mu_x = mu[ind]
    var_x = var[ind]
    mu_y = mean_y(mu, var, ind)
    var_y = variance_y(mu, var, ind)
    dx2 = d2f_dx2(mu_x, mu_y)
    dy2 = d2f_dy2(mu_x, mu_y)
    term0 = f(mu_x, mu_y)
    out = term0 + 0.5*(dx2*var_x + dy2*var_y)
    return out

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

# Check gradients
xTst = torch.tensor([3.0], requires_grad=True)
yTst = torch.tensor([100.0], requires_grad=True)
df_dx2_ag = d2f_dx2_autograd(xTst, yTst)
df_dy2_ag = d2f_dy2_autograd(xTst, yTst)
# Get second derivatives with formulas
dz_dx2f = d2f_dx2(xTst, yTst)
dz_dy2f = d2f_dy2(xTst, yTst)
# Print results
print(f'df_dx2: Autograd = {df_dx2_ag}, Formula = {dz_dx2f}')
print(f'df_dy2: Autograd = {df_dy2_ag}, Formula = {dz_dy2f}')


# Parameters of simulation
nX = 1
nDim = 5
nSamples = 1000000
meanType = 'ones'
covType = 'diagonal'
sigma = 2

# Parameters of distribution
if meanType == 'sin':
    mu = make_mu_sin(nX=nX, nDim=nDim)
elif meanType == 'ones':
    mu = torch.ones((1,nDim))
elif meanType == 'sparse':
    mu = torch.zeros((1,nDim)) + 0.1
    # Select random indices
    ind = np.random.randint(0, nDim, size=5)
    # Set values to 2
    mu[0,ind] = 2

if covType != 'poisson':
    print('not pois')
    covariance = make_covariance(nDim=nDim, sigmaScale=sigma, covType=covType)
    variance = torch.diag(covariance)
elif covType == 'poisson':
    print('pois')
    variance = mu.clone().squeeze() * sigma
    variance = torch.abs(variance)
    covariance = torch.diag(variance)
B = torch.eye(nDim)


# Compute Taylor approximation
meanTaylor = torch.zeros(nDim)
for ind in range(nDim):
    meanTaylor[ind] = taylor_approx_Ef(mu.squeeze(), variance, ind)

# Get empirical estimates
meanE, covE, smE = empirical_moments_projected_gaussian(mu, covariance,
                                                        nSamples=nSamples, B=B)

# Plot results
import matplotlib.pyplot as plt
plt.figure()
plt.plot(meanTaylor, 'b', label='Taylor')
plt.plot(meanE.squeeze(), 'r', label='Empirical')
plt.legend()
plt.show()


