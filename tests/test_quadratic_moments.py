##################
#
# TESTS FOR THE FUNCTIONS THAT COMPUTE MOMENTS
# OF QUADRATIC FORMS OF GAUSSIAN RANDOM VARIABLES
#
##################

import pytest
import torch
import numpy as np
import qr_library as qr
import torch.distributions.multivariate_normal as mvn
from test_functions import *


# Parameters of simulation
nX = 5
nDim = 30
nSamples = 500000

### MOMENTS OF NON-CENTRAL CHI-SQUARED DISTRIBUTION (ISOTROPIC VARIANCE)

# Parameters of distribution
mu = make_mu_sin(nX=nX, nDim=nDim)
sigma = 0.7
covariance = make_covariance(nDim=nDim, sigmaScale=sigma, covType='isotropic')

# Sample moments empirically
momentsE = empirical_moments_isotropic_gaussian_norm(mu, covariance,
                                                       nSamples=nSamples)
# Compute moments theoretically
normT = qr.nc_X2_moments(mu, sigma, 1/2)
norm2T = qr.nc_X2_moments(mu, sigma, 1)
invNormT = qr.nc_X2_moments(mu, sigma, -1/2)
invNorm2T = qr.nc_X2_moments(mu, sigma, -1)

# Compare moments
error1 = (momentsE['norm'] - normT)/normT
error2 = (momentsE['norm2'] - norm2T)/norm2T
error3 = (momentsE['invNorm'] - invNormT)/invNormT
error4 = (momentsE['invNorm2'] - invNorm2T)/invNorm2T

maxError = torch.max(torch.cat([error1, error2, error3, error4]))
assert maxError < 0.02


### MOMENTS OF QUADRATIC FORMS OF RANDOM VARIABLES

# Parameters of distribution
mu = make_mu_sin(nX=nX, nDim=nDim)
sigma = 0.7
covariance = make_covariance(nDim=nDim, sigmaScale=sigma, covType='random')
randMat1 = torch.randn(nDim, nDim)
M1 = torch.matmul(randMat1, randMat1.transpose(0, 1))
randMat2 = torch.randn(nDim, nDim)
M2 = torch.matmul(randMat2, randMat2.transpose(0, 1))

# Get empirical estimates
momentsE = empirical_moments_quadratic_form(mu, covariance, M1,
                                               nSamples=nSamples)
covE = empirical_covariance_quadratic_form(mu, covariance, M1, M2,
                                           nSamples=nSamples)
# Get theoretical moments
meanT = qr.quadratic_form_mean(mu, covariance, M1)
varT = qr.quadratic_form_var(mu, covariance, M1)
covT = qr.quadratic_form_cov(mu, covariance, M1, M2)

# Compare moments
error1 = (momentsE['mean'] - meanT)/meanT
error2 = (momentsE['var'] - varT)/varT
error3 = (covE - covT)/covT

maxError = torch.max(torch.cat([error1, error2, error3]))
assert maxError < 0.02

