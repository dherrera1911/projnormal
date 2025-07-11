---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: python3
  language: python
  name: python3
---

# Fitting distributions to data

This tutorial explains how to fit the distributions provided by
the `projnormal` package to data. 
The classes and methods used for fitting the distributions
data are available in `projnormal.models`.

We'll demonstrate fitting two different projected normal distributions:

* The standard projected normal distribution (`ProjNormal`).
* The projected normal distribution with a constant term (`ProjNormalConst`).

## Fitting the Projected Normal Distribution

Let's start by defining some (true) parameters for the distribution, and
generating synthetic data that we can then use to fit the distribution.


```{code-cell} ipython3
import projnormal
import torch

# Distribution parameters
N_DIM = 3
mean_x = projnormal.param_sampling.make_mean(N_DIM)
cov_x = projnormal.param_sampling.make_spdm(N_DIM)

# Initialize the projected normal distribution class for sampling
pn_sample = projnormal.classes.ProjNormal(
    n_dim=N_DIM,
    mean_x=mean_x,
    covariance_x=cov_x,
)

# Generate samples
samples = pn_sample.sample(2000)
```

Next, we initialize a new instance of the `ProjNormal` class, but without
the true parameters. This instance will be used to fit the model to the
generated samples via Maximum Likelihood Estimation (MLE).

```{code-cell} ipython3
# Fit the model using Maximum Likelihood Estimation (MLE)
pn_fit_mle = projnormal.classes.ProjNormal(n_dim=N_DIM)
pn_fit_mle.max_likelihood(samples, show_progress=False)

# Print the fitted parameters
print("True mean:\n", mean_x.numpy())
print("Fitted mean (MLE):\n", pn_fit_mle.mean_x.detach().numpy())

print("True covariance:\n", cov_x.numpy())
print("Fitted covariance (MLE):\n", pn_fit_mle.covariance_x.detach().numpy())
```

Now, we can also fit the model using Moment Matching, which uses the sample moments
to estimate the parameters of the distribution.

```{code-cell} ipython3
# Fit the model using Moment Matching
mean_data = torch.mean(samples, dim=0)
cov_data = torch.cov(samples.T)
data_moments = {'mean': mean_data, 'covariance': cov_data}

# Fit the model using Moment Matching
pn_fit_mm = projnormal.classes.ProjNormal(n_dim=N_DIM)
pn_fit_mm.moment_match(data_moments=data_moments, show_progress=False)

# Print the fitted parameters
print("Fitted mean (Moment Matching):\n", pn_fit_mm.mean_x.detach().numpy())
print("Fitted covariance (Moment Matching):\n", pn_fit_mm.covariance_x.detach().numpy())
```


## Fitting the Projected Normal Distribution with a Constant Term

For illustration purposes, the example code below shows how to
fit one of the variants of the projected normal distribution, that
includes a constant term in the denominator.

```{code-cell} ipython3
# Define the constant term
const = torch.tensor(3.0)

# Initialize the projected normal distribution with constant term for sampling
pnc_sample = projnormal.classes.ProjNormalConst(
    n_dim=N_DIM,
    mean_x=mean_x,
    covariance_x=cov_x,
    const=const,
)

# Generate samples
samples_const = pnc_sample.sample(2000)

# Fit the model using Maximum Likelihood Estimation (MLE)
pnc_fit_mle = projnormal.classes.ProjNormalConst(n_dim=N_DIM)
pnc_fit_mle.max_likelihood(samples_const, show_progress=False)

# Print the fitted parameters
print("True mean:\n", mean_x.numpy())
print("Fitted mean (MLE):\n", pnc_fit_mle.mean_x.detach().numpy())

print("True covariance:\n", cov_x.numpy())
print("Fitted covariance (MLE):\n", pnc_fit_mle.covariance_x.detach().numpy())

print("True constant:\n", const.numpy())
print("Fitted constant (MLE):\n", pnc_fit_mle.const.detach().numpy())
```

Finally, we can fit the `ProjNormalConst` model using Moment Matching as well.

```{code-cell} ipython3
# Fit the model using Moment Matching
mean_data_const = torch.mean(samples_const, dim=0)
cov_data_const = torch.cov(samples_const.T)
data_moments_const = {'mean': mean_data_const, 'covariance': cov_data_const}

# Fit the model using Moment Matching
pnc_fit_mm = projnormal.classes.ProjNormalConst(n_dim=N_DIM)
pnc_fit_mm.moment_match(data_moments=data_moments_const, show_progress=False)

# Print the fitted parameters
print("Fitted mean (Moment Matching):\n", pnc_fit_mm.mean_x.numpy())
print("Fitted covariance (Moment Matching)\n:", pnc_fit_mm.covariance_x.detach().numpy())
print("Fitted constant (Moment Matching)\n:", pnc_fit_mm.const.detach().numpy())
```
