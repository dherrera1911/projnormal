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

# Formulas available in projnormal

This document describes the organization of the formulas
available in `projnormal` .

`projnormal` implements formulas for working with the
projected normal distribution, and some generalized versions
of this distribution.
A random variable following the projected normal distribution,
$\mathbf{y} \sim \mathcal{PN}(\boldsymbol{\mu}, \Sigma)$,
is obtained by radially projecting a multivariate normal variable $\mathbf{x}$
onto the unit sphere, i.e.,
$\mathbf{y} = \frac{\mathbf{x}}{\|\mathbf{x}\|}$
where $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$.

The generalizations of the projected normal included
in the package are of the form

$\mathbf{y} = \frac{\mathbf{x}}{\sqrt{\mathbf{x} \mathbf{B} \mathbf{x} + c}}$

where $\mathbf{B}$ is a positive definite matrix and $c$
is a non-negative constant.

Also, `projnormal` includes a separate implementation
of the special case of the projected normal distribution
where $\Sigma = \mathbf{I} \sigma^2$.

For each of these distributions, `projnormal` provides formulas
to obtain the log-PDF, PDF, and the first
and second moments of the distribution.

The formulas for each distribution are available
as separate modules in `projnormal.formulas`. Lets first 
focus in the basic projected normal distribution.

## Projected Normal Distribution formulas

The formulas for the projected normal distribution are available
at `projnormal.formulas.projected_normal`. This module also includes
sampling functions. Lets generate some samples from the
distribution and compute their PDFs.


```{code-cell} ipython3
import projnormal
import torch

# PROJECTED NORMAL DISTRIBUTION FORMULAS MODULE
import projnormal.formulas.projected_normal as projnormal_dist

# Distribution parameters. projnormal has functions to generate distribution parameters
n_dim = 5  # The formulas work for any dimensionality
mean_x = projnormal.param_sampling.make_mean(n_dim)
cov_x = projnormal.param_sampling.make_spdm(n_dim)

# Sample distribution
samples = projnormal_dist.sample(
  mean_x=mean_x,
  covariance_x=cov_x,
  n_samples=2000,
)

# Compute PDF values for the samples
pdfs = projnormal_dist.pdf(
  mean_x=mean_x,
  covariance_x=cov_x,
  y=samples,
)
```

`projnormal` also provides analytic formulas for the mean and
second moment matrix of the projected normal distribution,
as obtained using a second-order Taylor approximation.
Lets compute these values and compare the approximated
mean to the sample mean.


```{code-cell} ipython3
# Compute the approximation to the distribution moments
y_mean = projnormal_dist.mean(
  mean_x=mean_x,
  covariance_x=cov_x,
)

y_sm = projnormal_dist.second_moment(
  mean_x=mean_x,
  covariance_x=cov_x,
)

print(f"Sample mean: {samples.mean(dim=0)}")
print(f"Approximated mean: {y_mean}")
```

The list of functions available for the projected normal distribution
can be found in the API reference.


## Other distributions

The available distributions with formulas in `projnormal` are
organized as modules in `projnormal.formulas`. All of these
modules provide the same set of functions. The available
modules are:

- `projnormal.formulas.projected_normal`: The basic projected normal distribution.
- `projnormal.formulas.projected_normal_iso`: The projected normal distribution
with isotropic covariance matrix. Unlike the other distributions, this one
has exact formulas for the mean and second moment matrix.
- `projnormal.formulas.projected_normal_Bc`: The projected normal distribution
with a positive definite matrix $\mathbf{B}$ and a constant $c>0$
in the denominator.
- `projnormal.formulas.projected_normal_B`: The projected normal distribution
with a positive definite matrix $\mathbf{B}$ in the denominator and $c = 0$.
- `projnormal.formulas.projected_normal_c`: The projected normal distribution
with a constant $c>0$ in the denominator and $\mathbf{B} = \mathbf{I}$.

It might be noted above that all the distributions can be obtained
by setting the parameters $\mathbf{B}$ and $c$ to specific values.
However, different implementations for the cases where
$c = 0$ and $\mathbf{B} = \mathbf{I}$ are provided for different reasons. 

Different implementations with $c=0$ and $c>0$ are provided because
these two cases are qualitatively different. When $c=0$, the
variable $\mathbf{y}$ is constrained to an $n-1$ dimensional surface
(the sphere in the case of $\mathbf{B} = \mathbf{I}$),
while when $c>0$, the variable $\mathbf{y}$ is defined on an
$n$ dimensional subset of the space. Different formulas
are required to compute the PDFs in these two cases.

Then, different implementations with $\mathbf{B} = \mathbf{I}$
and general $\mathbf{B}$ are provided because of
efficiency.

For completeness, lets show how to sample, compute the PDF, and
the moments, for the distribution with  $c > 0$ and general $\mathbf{B}$.

```{code-cell} ipython3
# PROJECTED NORMAL with B and c > 0
import projnormal.formulas.projected_normal_Bc as pnbc_dist

const = 1.0
B = torch.diag(torch.rand(n_dim) + 0.1)

# Sample distribution
samples_ellipse = pnbc_dist.sample(
  mean_x=mean_x,
  covariance_x=cov_x,
  B=B,
  const=const,
  n_samples=2000,
)

# Compute PDF values for the samples
pdfs_ellipse = pnbc_dist.pdf(
  mean_x=mean_x,
  covariance_x=cov_x,
  y=samples_ellipse,
  B=B,
  const=const,
)


# Compute the approximation to the distribution moments
y_mean = pnbc_dist.mean(
  mean_x=mean_x,
  covariance_x=cov_x,
  B=B,
  const=const,
)

y_sm = pnbc_dist.second_moment(
  mean_x=mean_x,
  covariance_x=cov_x,
  B=B,
  const=const,
)
```
