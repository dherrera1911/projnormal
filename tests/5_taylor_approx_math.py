##################
#
# TESTS THE MATHEMATICAL COMPUTATIONS IN THE TAYLOR APPROXIMATION
#
##################

#import pytest
import torch
import numpy as np
import qr_library as qr
from einops import reduce
from test_functions import *
import time

# Parameters of distribution
nDim = 1000
sigma = 1
muMult = 4
covType = 'random'

# Instantiate parameters
mu = make_mu_sin(nDim=nDim) * muMult
mu = torch.abs(mu)
covariance = make_covariance(nDim=nDim, sigmaScale=sigma, covType=covType)
variances = torch.diag(covariance)
B = torch.eye(nDim)

# Compute the mean the variable V = X^2 - Xi^2 for each Xi
meanV = qr.v_mean(mu=mu, covariance=covariance)

############################
## TEST FUNCTIONS FOR COMPUTING DERIVATIVES
############################

def f(u, v):
    " f = u/sqrt(u^2+v) "
    return u/(torch.sqrt(u**2+v))

# Function to compute d2f/du2 using autograd
def d2f_du2_autograd(u, v):
    # First derivative with respect to x
    df_du = torch.autograd.grad(f(u, v), u, create_graph=True)[0]
    # Second derivative with respect to u (the gradient of df_du with respect to u)
    d2f_du2 = torch.autograd.grad(df_du, u)[0]
    return d2f_du2

# Function to compute d2f/dv2 using autograd
def d2f_dv2_autograd(u, v):
    # First derivative with respect to v
    df_dv = torch.autograd.grad(f(u, v), v, create_graph=True)[0]
    # Second derivative with respect to v (the gradient of df_dv with respect to v)
    d2f_dv2 = torch.autograd.grad(df_dv, v)[0]
    return d2f_dv2

# Function to compute d2f/dudv using autograd
def d2f_dudv_autograd(u, v):
    # First derivative with respect to v
    df_du = torch.autograd.grad(f(u, v), u, create_graph=True)[0]
    # Second derivative with respect to v (the gradient of df_dv with respect to v)
    d2f_dudv = torch.autograd.grad(df_du, v)[0]
    return d2f_dudv

du2_autograd = torch.zeros(nDim)
dv2_autograd = torch.zeros(nDim)
dudv_autograd = torch.zeros(nDim)

for i in range(nDim):
    x = mu[i].clone().detach().requires_grad_(True)
    y = meanV[i].clone().detach().requires_grad_(True)
    # Compute the second derivatives of f
    du2_autograd[i] = d2f_du2_autograd(x, y)
    dv2_autograd[i] = d2f_dv2_autograd(x, y)
    dudv_autograd[i] = d2f_dudv_autograd(x, y)

# Function to compute the second derivative of f with respect to x
du2_fun = qr.prnorm_du2(u=mu, v=meanV)
dv2_fun = qr.prnorm_dv2(u=mu, v=meanV)
dudv_fun = qr.prnorm_dudv(u=mu, v=meanV)

# Compute the error
error = torch.max(torch.abs(du2_autograd - du2_fun))
error2 = torch.max(torch.abs(dv2_autograd - dv2_fun))
error3 = torch.max(torch.abs(dudv_autograd - dudv_fun))

print(f'Error in d2f/du2 = {error}')
print(f'Error in d2f/dv2 = {error2}')
print(f'Error in d2f/dudv = {error3}')


############################
## TEST EFFICIENT SUB-COMPUTATIONS OF V-VARIABLE STATISTICS
############################

### EFFICIENT COMPUTATION OF TRACE OF SUB-MATRICES
# Trace of full covariance product
fullTrace = torch.trace(torch.matmul(covariance, covariance))
# Inner products of rows and columns of covariance
innerProds = torch.einsum('ij,ij->i', covariance, covariance)
# Trace of covariance products removing ith row and column
subTraces = fullTrace - innerProds * 2 + covariance.diagonal()**2
# Compute the sub-traces by explicitly removing the ith row and column
subTraces2 = torch.zeros(nDim)
for i in range(nDim):
    keepInds = list(range(nDim))
    keepInds.remove(i)
    subCov = covariance.clone()
    subCov = subCov[keepInds,:]
    subCov = subCov[:,keepInds]
    subTraces2[i] = torch.trace(torch.matmul(subCov, subCov))

# Compute the error
error = torch.max(torch.abs(subTraces - subTraces2))
print(f'Error in computing the sub-traces = {error}')

### EFFICIENT COMPUTATION OF X'CX of submatrices
fullQ = torch.einsum('i,ij,j->', mu, covariance, mu)
linearF = torch.einsum('i,ij,j->i', mu, covariance, mu)
subQ = fullQ - 2 * linearF + mu**2 * torch.diag(covariance)

# Compute the sub-traces by explicitly removing the ith row and column
subQ2 = torch.zeros(nDim)
for i in range(nDim):
    keepInds = list(range(nDim))
    keepInds.remove(i)
    subCov = covariance.clone()
    subCov = subCov[keepInds,:]
    subCov = subCov[:,keepInds]
    subMu = mu[keepInds]
    subQ2[i] = torch.einsum('i,ij,j->', subMu, subCov, subMu)

# Compute the error
error = torch.max(torch.abs(subQ - subQ2))
print(f'Error in computing the sub-Q = {error}')

### EFFICIENT COMPUTATION OF COV(U,V)
covUV = 2 * (torch.einsum('i,ij->j', mu, covariance) - \
    mu * covariance.diagonal())

covUV2 = torch.zeros(nDim)
for i in range(nDim):
    a = torch.zeros(nDim)
    A = torch.eye(nDim, nDim)
    a[i] = 1
    A[i,i] = 0
    covUV2[i] = 2 * torch.einsum('i,ij,jk,k->', mu, A, covariance, a)

error = torch.max(torch.abs(covUV - covUV2))
print(f'Error in computing the sub-covUV = {error}')


############################
## TEST THE COMPUTING OF V-VARIABLE STATISTICS
############################

# Efficient functions
start = time.time()
vMean = qr.v_mean(mu=mu, covariance=covariance)
vVar = qr.v_var(mu=mu, covariance=covariance)
vCov = qr.v_cov(mu=mu, covariance=covariance)
end = time.time()
time1 = end - start

# Compute naively
vMean2 = torch.zeros(nDim)
vVar2 = torch.zeros(nDim)
vCov2 = torch.zeros(nDim)

start = time.time()
for i in range(nDim):
    keepInds = list(range(nDim))
    keepInds.remove(i)
    # Compute mean and variance
    # First remove i-th element from variable
    subB = torch.eye(nDim-1)
    subCov = covariance.clone()
    subCov = subCov[keepInds,:]
    subCov = subCov[:,keepInds]
    subMu = mu[keepInds]
    # Compute moments with remaining elements
    vMean2[i] = qr.quadratic_form_mean(mu=subMu, covariance=subCov, M=subB)
    vVar2[i] = qr.quadratic_form_var(mu=subMu, covariance=subCov, M=subB)
    # Compute covariance
    a = torch.zeros(nDim)
    A = torch.eye(nDim, nDim)
    a[i] = 1
    A[i,i] = 0
    vCov2[i] = qr.quadratic_linear_cov(mu=mu, covariance=covariance, M=A, b=a)
end = time.time()
time2 = end - start

errorMean = torch.max(torch.abs(vMean - vMean2))
errorVar = torch.max(torch.abs(vVar - vVar2))
errorCov = torch.max(torch.abs(vCov - vCov2))
print(f'Error in computing the mean of V = {errorMean}')
print(f'Error in computing the variance of V = {errorVar}')
print(f'Error in computing the covariance of V = {errorCov}')

print(f'Time for efficient computation = {time1}')
print(f'Time for naive computation = {time2}')

