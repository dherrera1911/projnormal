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

plotCovariances = False

# Parameters of simulation
nDim = 20
nSamples = 500000

####################################
####### ISOTROPIC NOISE CASE #######
#######    EXACT FORMULAS    #######
####################################

# Parameters of distribution
mu = make_mu_sin(nDim=nDim)
sigma = 0.5
covariance = make_covariance(nDim=nDim, sigmaScale=sigma, covType='isotropic')
B = torch.eye(nDim)

# Get analytical estimates
meanA = qr.prnorm_mean_iso(mu=mu, sigma=sigma)
smA = qr.prnorm_sm_iso(mu=mu, sigma=sigma)
covA = qr.secondM_2_cov(secondM=smA, mean=meanA)

# Get empirical estimates
meanE, covE, smE = empirical_moments_prnorm(mu, covariance,
                                                        nSamples=nSamples, B=B)

# Get errors
meanErr = torch.max(torch.abs(meanE - meanA)/(torch.abs(meanE)+0.02))
covErr = torch.max(torch.abs(covE - covA)/(torch.abs(covE)+0.02))
smErr = torch.max(torch.abs(smE - smA)/(torch.abs(smE)+0.02))

# Print with 2 decimals
print(f'Mean error (max) = {meanErr*100:.2f}%')
print(f'Cov error (max) = {covErr*100:.2f}%')

# Plot covariances
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


# Check that the efficiently computed average second moment
nX = 10

# Make many means
mu = torch.zeros((nX, nDim))
smA = torch.zeros((nX, nDim, nDim))
for i in range(nX):
    mu[i,:] = make_mu_sin(nDim=nDim)
    smA[i,:,:] = qr.prnorm_sm_iso(mu=mu[i], sigma=sigma)

# Average individually computed second moments
avSMA = reduce(smA, 'n d b -> d b', 'mean')
# Compute average second moment efficiently
avSMA2 = qr.prnorm_sm_iso_batch(mu=mu, sigma=sigma)
# Compute the error and print
smADiff = torch.max(torch.abs(avSMA - avSMA2)/(torch.abs(avSMA)))
print(f'SM batch error (max) = {smADiff*100:.2f}%')

