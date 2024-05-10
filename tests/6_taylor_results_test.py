##################
#
# TESTS FOR THE FUNCTIONS THAT COMPUTE MOMENTS
# OF THE PROJECTED NORMAL DISTRIBUTION
#
##################

#import pytest
import torch
import numpy as np
import projected_normal as pn
from einops import reduce
from test_functions import *

####################################
####### INDEPENDENT NOISE CASE #####
#######  TAYLOR APPROXIMATION ######
####################################

nDim = 10
nSamples = 1000000

# Parameters of distribution
meanType = 'ones'
covType = 'random'
sigma = 1
muMult = 1
c50 = 5

# Instantiate parameters
if meanType == 'sin':
    mu = make_mu_sin(nDim=nDim) * muMult
    mu = torch.abs(mu)
elif meanType == 'ones':
    mu = torch.ones(nDim) * muMult
elif meanType == 'sparse':
    mu = torch.rand(nDim) + 0.1
    # Select random indices
    ind = np.random.randint(0, nDim, size=5)
    # Set values to 2
    mu[ind] = 2
    mu = mu * muMult

if covType != 'poisson':
    covariance = make_covariance(nDim=nDim, sigmaScale=sigma, covType=covType)
    variances = torch.diag(covariance)
elif covType == 'poisson':
    variances = mu.clone().squeeze() * sigma**2
    variances = torch.abs(variances)
    covariance = torch.diag(variances)

# Option 1
#B = torch.eye(nDim) * 0.1
# Option 2
B = torch.zeros((nDim, nDim))
i = torch.arange(nDim)
B[i,i] = torch.exp(-i.float()*3/nDim)
# Option 3
#B = torch.eye(nDim) * 0.5
#i = torch.arange(3)
#B[i,i] = 0

# Get empirical moments
momentsE = pn.empirical_moments_prnorm(mu, covariance, nSamples=nSamples,
                                       B=B, c50=c50)
meanE = momentsE['mean']
covE = momentsE['covariance']
smE = momentsE['secondM']

# Get analytic moments
meanA = pn.prnorm_mean_taylor(mu=mu, covariance=covariance, B=B, c50=c50)
smA = pn.prnorm_sm_taylor(mu=mu, covariance=covariance, B=B, c50=c50)
covA = pn.secondM_2_cov(smA, meanA)

# Plot covariances
plotCovariances = True
if plotCovariances:
    maxVal = np.max([torch.max(covE), torch.max(covA)])
    minVal = np.min([torch.min(covE), torch.min(covA)])
    # Set min and max values to the color scale
    plt.subplot(1,4,1)
    plt.imshow(covE, vmin=minVal, vmax=maxVal)
    # Subplot title on top
    plt.title('Empirical')
    plt.subplot(1,4,2)
    plt.imshow(covA, vmin=minVal, vmax=maxVal)
    plt.title('Theoretical')
    plt.subplot(1,4,3)
    plt.imshow(covE - covA, vmin=minVal, vmax=maxVal)
    plt.title('Difference')
    plt.subplot(1,4,4)
    plt.plot(meanE, label='Empirical')
    plt.plot(meanA, label='Theoretical')
    plt.title('Means')
    plt.legend()
    # Set figure size
    plt.gcf().set_size_inches(14, 4)
    plt.show()


