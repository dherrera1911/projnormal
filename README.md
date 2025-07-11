<img src="docs/source/_static/cartoon.svg" alt="projnormal logo" width="500"/>

`projnormal` is a Python package for working with the
projected normal and related distributions. It uses a
PyTorch backend to provide efficient computations and
fitting procedures.

Given an $n$-dimensional variable
$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$,
the variable obtained by radially projecting
$\mathbf{x}$ onto the unit sphere,
$\mathbf{y} = \frac{\mathbf{x}}{||\mathbf{x}||}$,
follows a projected normal distribution, denoted
as $\mathbf{y} \sim \mathcal{PN}(\boldsymbol{\mu}, \Sigma)$.

The package was introduced in the preprint
["Projected Normal Distribution: Moment Approximations and Generalizations"](https://arxiv.org/abs/2506.17461),
which presents the implemented formulas.


## Projected Normal Distribution

`projnormal` implements the following functionalities for
the projected normal distribution (and related distributions):
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

