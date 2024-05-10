##################
#
# TESTS FOR THE FUNCTIONS THAT COMPUTE MOMENTS
# OF QUADRATIC FORMS OF GAUSSIAN RANDOM VARIABLES
#
##################

import pytest
import torch
import numpy as np
import projected_normal as pn
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
qfSamples = pn.sample_quadratic_form(mu=mu, covariance=covariance,
                                     M=torch.eye(nDim), nSamples=nSamples)

normE = torch.mean(torch.sqrt(qfSamples))
norm2E = torch.mean(qfSamples)
invNormE = torch.mean(1/torch.sqrt(qfSamples))
invNorm2E = torch.mean(1/qfSamples)

# Compute moments theoretically
normT = pn.nc_X2_moments(mu, sigma, 1/2)
norm2T = pn.nc_X2_moments(mu, sigma, 1)
invNormT = pn.nc_X2_moments(mu, sigma, -1/2)
invNorm2T = pn.nc_X2_moments(mu, sigma, -1)

# Compare moments
error1 = (normE - normT)/normT
error2 = (norm2E - norm2T)/norm2T
error3 = (invNormE - invNormT)/invNormT
error4 = (invNorm2E - invNorm2T)/invNorm2T

maxError = torch.max(torch.tensor([error1, error2, error3, error4]))
assert maxError < 0.02

