##################
#
# TESTS THAT THE PROJECTED NORMAL DISTRIBUTION CLASS
# IS ABLE TO PERFORM ITS BASIC FUNCTIONS
#
##################

import time
import pytest
import torch
from projected_normal.prnorm_class import ProjectedNormal
from projected_normal.auxiliary import is_symmetric, is_positive_definite
from utility_functions import make_mu_sin, make_covariance

mean_type = "sin"
cov_type = "diagonal"
cov_param = "Logarithm"
n_dim = 20
sigma = 0.01
tolerance = 1e-3

# Initialize the mean of the gaussian
if mean_type == "sin":
    mu = make_mu_sin(n_dim=n_dim)
elif mean_type == "ones":
    mu = torch.ones(n_dim)
elif mean_type == "sparse":
    mu = torch.zeros(n_dim)
    # Set values to 1
    mu[::3] = 1
mu = mu / torch.norm(mu)

# Initialize the covariance of the gaussian
covariance = make_covariance(n_dim=n_dim, cov_scale=sigma, cov_type=cov_type)

# Initialize the projected normal class to generate the data
dtype = torch.float64
prnorm_data = ProjectedNormal(
    n_dim=n_dim, mu=mu, covariance=covariance,
    covariance_parametrization=cov_param,
    dtype=dtype
)

# Sample from the distribution
with torch.no_grad():
    samples = prnorm_data.sample(n_samples=10000)

# Initialize the projected normal to fit to the data
#covariance_initial = torch.eye(n_dim) * 5  # Use a far off covariance
covariance_initial = torch.cov(samples.T) * torch.sqrt(torch.tensor(n_dim))

prnorm_fit = ProjectedNormal(
  n_dim=n_dim,
  mu=mu_initial,
  covariance=covariance_initial,
  covariance_parametrization=cov_param,
  dtype=dtype
)

# Get initial parameters
mu_initial = prnorm_fit.mu.detach().clone()
covariance_initial = prnorm_fit.covariance.detach().clone()

# Fit to the data with maximum likelihood
n_rep = 10
loss_list = []
for i in range(n_rep):
    # Initialize optimizer and scheduler
    lr = 0.1 * (0.9**i)
    optimizer, scheduler = prnorm_fit.initialize_optimizer_and_scheduler(lr=lr)
    loss = prnorm_fit.ml_fit(
        y=samples, optimizer=optimizer, scheduler=scheduler, n_iter=1
    )
    loss_list.append(loss)
loss = torch.cat(loss_list)

# Estimated parameters
mu_estimated = prnorm_fit.mu.detach().clone()
covariance_estimated = prnorm_fit.covariance.detach().clone()

# Get individual pdf's
lpdfs = prnorm_fit.log_pdf(samples)
pdfs = prnorm_fit.pdf(samples)

# Get errors
mu_error_i = torch.norm(mu_initial - mu)
mu_error_f = torch.norm(mu_estimated - mu)
covariance_error_i = torch.norm(covariance_initial - covariance)
covariance_error_f = torch.norm(covariance_estimated - covariance)


