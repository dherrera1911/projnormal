##################
#
# TESTS THE MATHEMATICAL COMPUTATIONS IN THE TAYLOR APPROXIMATION
#
##################

import pytest
import torch
import numpy as np
import qr_library as qr
from einops import reduce
from test_functions import *
import time

# Parameters of distribution
nDim = 10
sigma = 1
muMult = 4
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
meanV = qr.v_mean(mu=mu, covariance=covariance, weights=B.diagonal())

############################
## TEST THE COMPUTING OF V-VARIABLE STATISTICS
############################

# Efficient functions
weights = B.diag()
start = time.time()
vMean = qr.v_mean(mu=mu, covariance=covariance, weights=weights)
vVar = qr.v_var(mu=mu, covariance=covariance, weights=weights)
vCov = qr.v_cov(mu=mu, covariance=covariance, weights=weights)
end = time.time()
time1 = end - start

qr.v_mean(mu=mu, covariance=covariance, weights=weights)

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
    subB = B.clone()
    subB = subB[keepInds,:]
    subB = subB[:,keepInds]
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
    A = B.clone()
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
print(f'Daniel\'s trick is {time2/time1} times faster')


############################
## TEST EFFICIENT SUB-COMPUTATIONS OF V-VARIABLE STATISTICS
############################

### EFFICIENT COMPUTATION OF TRACE OF SUB-MATRICES
# Trace of full covariance product
Bcov = torch.einsum('i,ij->ij', B.diagonal(), covariance)
fullTrace = qr.product_trace(Bcov, Bcov)
# Inner products of rows and columns of covariance
innerProds = torch.einsum('ij,ji->i', Bcov, Bcov)
# Trace of covariance products removing ith row and column
subTraces = fullTrace - innerProds * 2 + Bcov.diagonal()**2
# Compute the sub-traces by explicitly removing the ith row and column
subTraces2 = torch.zeros(nDim)
for i in range(nDim):
    keepInds = list(range(nDim))
    keepInds.remove(i)
    subCov = covariance.clone()
    subCov = subCov[keepInds,:]
    subCov = subCov[:,keepInds]
    subB = B.clone()
    subB = subB[keepInds,:]
    subB = subB[:,keepInds]
    BsubCov = torch.matmul(subB, subCov)
    subTraces2[i] = torch.trace(torch.matmul(BsubCov, BsubCov))

# Compute the error
error = torch.max(torch.abs(subTraces - subTraces2))
print(f'Error in computing the sub-traces = {error}')

### EFFICIENT COMPUTATION OF X'CX of submatrices
Bcov = torch.einsum('i,ij->ij', B.diag(), covariance)
fullQ = torch.einsum('i,ij,j->', mu, Bcov, mu * B.diag())
linearF = torch.einsum('i,ij,j->i', mu, Bcov, mu * B.diag())
subQ = fullQ - 2 * linearF + mu**2 * torch.diag(Bcov) * B.diag()

# Compute the quadratic forms explicitly by removing the ith row and column
subQ2 = torch.zeros(nDim)
for i in range(nDim):
    keepInds = list(range(nDim))
    keepInds.remove(i)
    subCov = covariance.clone()
    subCov = subCov[keepInds,:]
    subCov = subCov[:,keepInds]
    subB = B.clone()
    subB = subB[keepInds,:]
    subB = subB[:,keepInds]
    subMu = mu[keepInds]
    subQ2[i] = torch.einsum('i,ij,jk,km,m->', subMu, subB, subCov, subB, subMu)

# Compute the error
error = torch.max(torch.abs(subQ - subQ2))
print(f'Error in computing the sub-Q = {error}')

### EFFICIENT COMPUTATION OF COV(U,V)
covUV = 2 * (torch.einsum('i,ij->j', mu*B.diag(), covariance) - \
    mu * covariance.diagonal() * B.diag())

covUV2 = torch.zeros(nDim)
for i in range(nDim):
    a = torch.zeros(nDim)
    A = B.clone()
    a[i] = 1
    A[i,i] = 0
    covUV2[i] = 2 * torch.einsum('i,ij,jk,k->', mu, A, covariance, a)

error = torch.max(torch.abs(covUV - covUV2))
print(f'Error in computing the sub-covUV = {error}')

