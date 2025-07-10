# `projnormal`: Python implementation of the Projected Normal Distribution

`projnormal` is a Python package for working with the
projected normal (also known as the angular Gaussian)
and related distributions. It uses a PyTorch backend
to provide efficient computations and fitting procedures.

Given a variable $\mathbf{x} \in \mathbb{R}^n$ following
a multivariate normal distribution
$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$,
the variable $\mathbf{y} = \frac{\mathbf{x}}{||\mathbf{x}||}$
follows a projected normal distribution, denoted
as $\mathbf{y} \sim \mathcal{PN}(\boldsymbol{\mu}, \Sigma)$.
That is, the projected normal distribution is obtained
by projecting a Gaussian random variable
onto the unit sphere $\mathbb{S}^{n-1}$.

The package was introduced in the preprint
["Projected Normal Distribution: Moment Approximations and Generalizations"](https://arxiv.org/abs/2506.17461),
which presents the implemented formulas.


## Projected Normal Distribution

`projnormal` implements the following functionalities for
the projected normal distribution:
* PDF and log-PDF formulas
* Maximum-likelihood parameter estimation
* Distribution sampling
* Approximations of the first and second moments
* Moment matching routines

In the example code below, we generate samples from
$\mathcal{PN}(\boldsymbol{\mu}, \Sigma)$ and compute their
PDF. The necessary formulas are implemented in the
submodule `projnormal.formulas.projected_normal`.

```python
import torch
import projnormal
import projnormal.formulas.projected_normal as pn_dist

# Sample distribution parameters

N_DIM = 3  # The package work with any dimension
mean_x = projnormal.param_sampling.make_mean(N_DIM)
covariance_x = projnormal.param_sampling.make_spdm(N_DIM)

# Generate distribution samples
samples = pn_dist.sample(
  mean_x=mean_x, covariance_x=covariance_x, n_samples=2000
)

# Compute samples PDF
pdf_values = pn_dist.pdf(
  mean_x=mean_x, covariance_x=covariance_x, y=samples
)
```

Next, we initialize a `ProjNormal` object and use it
to fit the distribution parameters to the samples.

```python
# Initialize a ProjNormal object to fit
pn_fit = projnormal.classes.ProjNormal(n_dim=N_DIM)

# Fit the parameters of the projected normal distribution
pn_fit.max_likelihood(y=samples)

# Check the fitted parameters against the original parameters
print("Fitted mean vector:", pn_fit.mean_x.detach()) 
print("True mean vector:", mean_x)

print("Fitted covariance matrix: \n", pn_fit.covariance_x.detach())
print("True covariance matrix: \n", covariance_x)
```
    

## Variants of the Projected Normal Distribution

`projnormal` also implements generalized versions of the
projected normal distribution, of the form

$$\mathbf{y} = \frac{\mathbf{x}}{\sqrt{\mathbf{x}^T \mathbf{B} \mathbf{x} + c}}$$

where $\mathbf{B}$ is a positive-definite matrix and $c$ is a
non-negative constant. The formulas for this variant of the
distribution are implemented in the
module `projnormal.formulas.projected_normal_Bc`.

In the next example we generate samples and compute their PDF
for this generalized version of the projected normal.

```python
import projnormal.formulas.projected_normal_Bc as pngen_dist

# We generate a B matrix and a constant c
B = projnormal.param_sampling.make_spdm(N_DIM) * 2.0
const = torch.as_tensor(2.0)

# Generate samples from the projected normal distribution
samples = pngen_dist.sample(
  mean_x=mean_x, covariance_x=covariance_x, B=B, const=const, n_samples=2000
)

# Compute the PDF of the samples
pdf_values = pngen_dist.pdf(
  mean_x=mean_x, covariance_x=covariance_x, B=B, const=const, y=samples
)
```

Next, we initialize a `ProjNormalEllipseConst` object, which
implements the formulas for the generalized projected normal
distribution, and use it to fit the distribution parameters.

```python
# Initialize a ProjNormalEllipseConst object to fit
pngen_fit = projnormal.classes.ProjNormalEllipseConst(
    n_dim=N_DIM,
    B=torch.eye(N_DIM) * 0.1, # Make sure samples are inside initial distribution support
)

# Fit the parameters of the projected normal distribution
pngen_fit.max_likelihood(y=samples, n_cycles=1)

# Check the fitted parameters against the original parameters
print("Fitted mean vector:", pngen_fit.mean_x.detach()) 
print("True mean vector:", mean_x)

print("Fitted covariance matrix: \n", pngen_fit.covariance_x.detach())
print("True covariance matrix: \n", covariance_x)

print("Fitted B matrix: \n", pngen_fit.B.detach())
print("True B matrix: \n", B)

print("Fitted constant: ", pngen_fit.const.detach())
print("True constant: ", const)
```

## Moment approximations

`projnormal` also provides formulas for analytically approximating
the first and second moments of the projected normal distribution
and its variants, as shown in the following example code.

```python
samples = pn_dist.sample(
  mean_x=mean_x, covariance_x=covariance_x, n_samples=5000
)

mean_y_empirical = samples.mean(dim=0)
sm_y_empirical = samples.T @ samples / (samples.shape[0] - 1)

mean_y_approx = pn_dist.mean(mean_x, covariance_x)
sm_y_approx = pn_dist.second_moment(mean_x, covariance_x)
```

These formulas can also be used to fit the distribution
parameters via moment matching:

```python
# Put the empirical moments in a dictionary
data_moments = {
    "mean": mean_y_empirical,
    "covariance": torch.cov(samples.T),
}

# Initialize a ProjNormal object to fit
pn_fit = projnormal.classes.ProjNormal(n_dim=N_DIM)

# Fit the parameters via moment matching
pn_fit.moment_match(data_moments)

# Check the fitted parameters against the original parameters
print("Fitted mean vector:", pn_fit.mean_x.detach()) 
print("True mean vector:", mean_x)

print("Fitted covariance matrix: \n", pn_fit.covariance_x.detach())
print("True covariance matrix: \n", covariance_x)
```


## Installation

### Virtual environment

We recommend installing the package in a virtual environment. For this,
you can first install `miniconda` 
([install instructions link](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)),
and then create a virtual environment with Python 3.11 using the following
shell command:

```bash
conda create -n my-projnormal python=3.11
```

You can then activate the virtual environment with the following command:

```bash
conda activate my-projnormal
```

You should activate the `my-sqfa` environment to install the package, and every
time you want to use it.


### Install package

To install the package, you can clone the GitHub
repository and install in editable mode using `pip`:

```bash
git clone https://github.com/dherrera1911/projnormal.git
cd projnormal
pip install -e "."
```

## Citation

If you use `projnormal` in your research, please cite the
preprint ["Projected Normal Distribution: Moment Approximations and Generalizations"](https://arxiv.org/abs/2506.17461):

```bibtex
@misc{herreraesposito2025projected,
      title={Projected Normal Distribution: Moment Approximations and Generalizations},
      author={Daniel Herrera-Esposito and Johannes Burge},
      year={2025},
      eprint={2506.17461},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2506.17461}, 
}
```

