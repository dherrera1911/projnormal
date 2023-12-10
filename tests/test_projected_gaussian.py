##################
#
# TESTS FOR THE FUNCTIONS THAT COMPUTE MOMENTS
# OF THE PROJECTED NORMAL DISTRIBUTION
#
##################

import pytest
import torch
import numpy as np
import qr_library as qr
from test_functions import *

# Parameters of simulation
nX = 5
nDim = 20
nSamples = 500000

####################################
####### ISOTROPIC NOISE CASE #######
#######    EXACT FORMULAS    #######
####################################

# Parameters of distribution
mu = make_mu_sin(nX=nX, nDim=nDim)
sigma = 0.5
covariance = make_covariance(nDim=nDim, sigmaScale=sigma, covType='isotropic')
B = torch.eye(nDim)

# Get theoretical estimates
meanT = qr.projected_normal_mean_iso(mu=mu, sigma=sigma)
smT = qr.projected_normal_sm_iso(mu=mu, sigma=sigma)
covT = qr.secondM_2_cov(secondM=smT, mean=meanT)

# Get empirical estimates
meanE, covE, smE = empirical_moments_projected_gaussian(mu, covariance,
                                                        nSamples=nSamples, B=B)

# Get errors
meanErr = torch.max(torch.abs(meanE - meanT)/(meanE+0.02))
covErr = torch.max(torch.abs(covE - covT)/(covE+0.02))
smErr = torch.max(torch.abs(smE - smT)/(smE+0.02))

# Plot covariances
maxN = 5
cols = np.min([nX, maxN])
for i in range(cols):
    maxVal = np.max([torch.max(covE[i, :, :]), torch.max(covT[i, :, :])])
    minVal = np.min([torch.min(covE[i, :, :]), torch.min(covT[i, :, :])])
    # Set min and max values to the color scale
    plt.subplot(4, cols, i+1)
    plt.imshow(covE[i, :, :], vmin=minVal, vmax=maxVal)
    plt.yticks(ticks=[])
    plt.xticks(ticks=[])
    if i == 0:
        plt.ylabel('Empirical')
    plt.subplot(4, cols, i+1+cols)
    plt.imshow(covT[i, :, :], vmin=minVal, vmax=maxVal)
    plt.yticks(ticks=[])
    plt.xticks(ticks=[])
    if i == 0:
        plt.ylabel('Theoretical')
    plt.subplot(4, cols, i+1+2*cols)
    plt.imshow(covE[i, :, :] - covT[i, :, :], vmin=minVal, vmax=maxVal)
    plt.yticks(ticks=[])
    plt.xticks(ticks=[])
    if i == 0:
        plt.ylabel('Difference')
    plt.subplot(4, cols, i+1+3*cols)
    plt.plot(meanE[i, :], label='Empirical')
    plt.plot(meanT[i, :], label='Theoretical')
    plt.legend()
    plt.yticks(ticks=[])
    plt.xticks(ticks=[])
plt.show()


####################################
####### INDEPENDENT NOISE CASE #####
#######  TAYLOR APPROXIMATION ######
####################################

nX = 1
nDim = 10
nSamples = 1000000

# Parameters of distribution
meanType = 'sin'
covType = 'diagonal'
sigma = 1
muMult = 4
approxMethod = 'taylor'

# Instantiate parameters
if meanType == 'sin':
    mu = make_mu_sin(nX=nX, nDim=nDim) * muMult
    mu = torch.abs(mu)
elif meanType == 'ones':
    mu = torch.ones((nX,nDim)) * muMult
elif meanType == 'sparse':
    mu = torch.rand((nX,nDim)) + 0.1
    # Select random indices
    ind = np.random.randint(0, nDim, size=5)
    # Set values to 2
    mu[:,ind] = 2
    mu = mu * muMult

if covType != 'poisson':
    print('not pois')
    covariance = make_covariance(nDim=nDim, sigmaScale=sigma, covType=covType)
    variances = torch.diag(covariance)
elif covType == 'poisson':
    print('pois')
    variances = mu.clone().squeeze() * sigma**2
    variances = torch.abs(variances)
    covariance = torch.diag(variances)
B = torch.eye(nDim)

# Get empirical moments
meanE, covE, smE = empirical_moments_projected_gaussian(mu, covariance,
                                                        nSamples=nSamples, B=B)

# Get theoretical moments
if approxMethod == 'taylor':
    meanT = qr.projected_normal_mean_taylor(mu=mu, variances=variances, order=2)
elif approxMethod == '0th':
    meanT = qr.projected_normal_mean_taylor(mu=mu, variances=variances, order=0)
elif approxMethod == 'empirical':
    meanT = meanE
smT = qr.projected_normal_sm(mu=mu, covariance=covariance, B=B)
covT = qr.secondM_2_cov(smT, meanT)


# Plot covariances
maxN = 5
cols = np.min([nX, maxN])
for i in range(cols):
    maxVal = np.max([torch.max(covE[i, :, :]), torch.max(covT[i, :, :])])
    minVal = np.min([torch.min(covE[i, :, :]), torch.min(covT[i, :, :])])
    # Set min and max values to the color scale
    plt.subplot(4, cols, i+1)
    plt.imshow(covE[i, :, :], vmin=minVal, vmax=maxVal)
    plt.yticks(ticks=[])
    plt.xticks(ticks=[])
    if i == 0:
        plt.ylabel('Empirical')
    plt.subplot(4, cols, i+1+cols)
    plt.imshow(covT[i, :, :], vmin=minVal, vmax=maxVal)
    plt.yticks(ticks=[])
    plt.xticks(ticks=[])
    if i == 0:
        plt.ylabel('Theoretical')
    plt.subplot(4, cols, i+1+2*cols)
    plt.imshow(covE[i, :, :] - covT[i, :, :], vmin=minVal, vmax=maxVal)
    plt.yticks(ticks=[])
    plt.xticks(ticks=[])
    if i == 0:
        plt.ylabel('Difference')
    plt.subplot(4, cols, i+1+3*cols)
    plt.plot(meanE[i, :], label='Empirical')
    plt.plot(meanT[i, :], label='Theoretical')
    plt.legend()
    plt.yticks(ticks=[])
    plt.xticks(ticks=[])
plt.show()


# Plot second moments
#maxN = 5
#cols = np.min([nX, maxN])
#for i in range(cols):
#    maxVal = np.max([torch.max(smE[i, :, :]), torch.max(smT[i, :, :])])
#    minVal = np.min([torch.min(smE[i, :, :]), torch.min(smT[i, :, :])])
#    # Set min and max values to the color scale
#    plt.subplot(4, cols, i+1)
#    plt.imshow(smE[i, :, :], vmin=minVal, vmax=maxVal)
#    plt.yticks(ticks=[])
#    plt.xticks(ticks=[])
#    if i == 0:
#        plt.ylabel('Empirical')
#    plt.subplot(4, cols, i+1+cols)
#    plt.imshow(smT[i, :, :], vmin=minVal, vmax=maxVal)
#    plt.yticks(ticks=[])
#    plt.xticks(ticks=[])
#    if i == 0:
#        plt.ylabel('Theoretical')
#    plt.subplot(4, cols, i+1+2*cols)
#    plt.imshow(smE[i, :, :] - smT[i, :, :], vmin=minVal, vmax=maxVal)
#    plt.yticks(ticks=[])
#    plt.xticks(ticks=[])
#    if i == 0:
#        plt.ylabel('Difference')
#    plt.subplot(4, cols, i+1+3*cols)
#    plt.plot(meanE[i, :], label='Empirical')
#    plt.plot(meanT[i, :], label='Theoretical')
#    plt.legend()
#    plt.yticks(ticks=[])
#    plt.xticks(ticks=[])
#plt.show()



####################################
#######  TAYLOR APPROXIMATION ######
#######     MATHEMATICS       ######
####################################

## TEST THAT THE METHOD FOR COMPUTING MOMENTS WORKS
meanX2 = qr.quadratic_form_mean(mu=mu, covariance=covariance, M=B)
meanXi2 = mu**2 + variances
meanV = meanX2 - meanXi2
varX2 = qr.quadratic_form_var(mu=mu, covariance=covariance, M=B)
varSubtract = 2 * variances**2 + 4 * mu**2 * variances
varV = varX2 - varSubtract

meanV2 = torch.zeros(nDim)
varV2 = torch.zeros(nDim)
for i in range(nDim):
    muRemainder = torch.cat((mu[0,:i], mu[0,i+1:]))
    muRemainder = muRemainder.unsqueeze(0)
    varRemainder = torch.cat((variances[:i], variances[i+1:]))
    covRemainder = torch.diag(varRemainder)
    B2 = torch.eye(nDim-1)
    meanV2[i] = qr.quadratic_form_mean(mu=muRemainder, covariance=covRemainder, M=B2)
    varV2[i] = qr.quadratic_form_var(mu=muRemainder, covariance=covRemainder, M=B2)


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
    x = mu[0,i].clone().detach().requires_grad_(True)
    y = meanV[0,i].clone().detach().requires_grad_(True)
    # Compute the derivative of f with respect to x
    du2_autograd[i] = d2f_dx2_autograd(x, y)
    # Compute the derivative of f with respect to y
    dv2_autograd[i] = d2f_dy2_autograd(x, y)

# Function to compute the second derivative of f with respect to x
du2_fun = qr.proj_normal_dx2(u=mu, v=meanV)
dv2_fun = qr.proj_normal_dv2(u=mu, v=meanV)

print(f'du2_autograd = {du2_autograd.numpy()}')
print(f'du2_fun = {du2_fun.numpy()}')
print(f'dv2_autograd = {dv2_autograd.numpy()}')
print(f'dv2_fun = {dv2_fun.numpy()}')


