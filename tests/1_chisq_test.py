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
nDim = 30
nSamples = 500000

### MOMENTS OF NON-CENTRAL CHI-SQUARED DISTRIBUTION (ISOTROPIC VARIANCE)

# Parameters of distribution
mu = make_mu_sin(nDim=nDim)
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

maxError = torch.max(torch.tensor([error1, error2, error3, error4]))
assert maxError < 0.02

