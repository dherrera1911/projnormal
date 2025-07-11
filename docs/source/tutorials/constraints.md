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

# Using constraints for optimization

The `projnormal` package makes use of constraints to fit the
parameters of the distributions. This tutorial explains how to make
use of constraints in the `projnormal` package.

## Why use constraints?

Constraints are useful for at least two reasons: to make
sure that the parameters are valid and identifiable, and
to aid optimization.

First, the parameters of the projected normal distribution
have to satisfy certain constraints to be valid. In particular,
the parameter $\Sigma_x$ must be positive definite,
and a constraint during learning ensures that this
requirement is met.
But the parameters of the projected normal distribution
are also underdetermined.
This is because the same distribution on the
sphere is obtained by projecting the random variable
$\mathbf{x}$, and by projecting any scaled version
$\lambda \cdot \mathbf{x}$ for any $\lambda > 0$
(with the corresponding scaling of the parameters
$\boldsymbol{\mu}_x$ and $\Sigma_x$).
Thus, an additional constraint is needed to make the
parameters identifiable (i.e. unique). Different identifiability
constraints have been used in the literature, but the
default in `projnormal` is the constrain $\|\boldsymbol{\mu}_x\| = 1$.

The second reason is that adding constraints to the parameters
can aid optimization. For example,
given scarce data, a diagonal constrain can be added to
the $\Sigma_x$, to avoid overfitting or having a rank-deficient
covariance matrix. We may also use known structure of the
problem, for example constraining $\Sigma_x$ or $\boldsymbol{\mu}_x$
to have a specific structure that reduces the number of
free parameters.

## Setting constraints in projnormal

`projnormal` implements constraints using a useful built-in
feature of PyTorch called parametrizations.

In short, once a learnable parameter (e.g. a vector or a matrix)
is defined in a PyTorch model, we can add a parametrization to it,
which will ensure that the parameter stays within the
desired constraints. For further details on parametrizations,
see below, and refer to the PyTorch
[parametrizations tutorial](https://docs.pytorch.org/tutorials/intermediate/parametrizations.html).

`projnormal` provides some built-in constraints that can be
added to the available classes. The next section shows
how to use these built-in constraints.

 
### Example: Adding a diagonal constraint to the covariance matrix

Let's first initialize a `ProjNormal` model, which comes with
the default constraints of the package.


```{code-cell} ipython3
import projnormal
import torch

N_DIM = 3

# Initialize projected normal class
pn_fit = projnormal.classes.ProjNormal(
  n_dim=N_DIM
)

print(pn_fit)
```

The model has two learnable parameters,
`mean_x` and `covariance_x`, and each comes with
a parametrization. The parameter `mean_x` has the
`Sphere()` constraint, and the parameter `covariance_x`
has the `PSD()` (positive definite) constraint.
Constraints are implemented as Python classes
that inherit from `torch.nn.Module` (this is
what `Sphere()` and `PSD()` are).

`projnormal` has some constraints available in the
module `projnormal.classes.constraints`.
For example, the class `Diagonal()` constraints a matrix
to be diagonal with positive diagonal elements.
To use this parametrization for `covariance_x`, we
first need to remove the existing parametrization, and then
add the new `Diagonal()` constraint. For this, the
`parametrize` module of PyTorch is used.

```{code-cell} ipython3
import torch.nn.utils.parametrize as parametrize

# Remove existing parametrization for covariance_x
parametrize.remove_parametrizations(
    pn_fit, "covariance_x"
)

# Add new parametrization
parametrize.register_parametrization(
    pn_fit, "covariance_x", projnormal.classes.constraints.Diagonal()
)

print(pn_fit)
```

We see that the `covariance_x` parameter now has a `Diagonal()` constraint.
To verify that the constraint works, let's generate some samples
from a projected normal distribution and fit the model to them.
The resulting parameter `covariance_x` should be diagonal.

```{code-cell} ipython3
# True parameters
mean_x = projnormal.param_sampling.make_mean(N_DIM)
cov_x = projnormal.param_sampling.make_spdm(N_DIM)

# Generate samples
samples = projnormal.formulas.projected_normal.sample(
  mean_x=mean_x,
  covariance_x=cov_x,
  n_samples=1000
)

# Fit the model to the samples
pn_fit.max_likelihood(samples, show_progress=False)

# Check the covariance matrix
print("True covariance: \n", cov_x.numpy())
print("Fitted covariance: \n", pn_fit.covariance_x.detach().numpy())
```

As expected, the fitted covariance matrix is diagonal.

:::{admonition} Parametrizations can be applied to any parameter
Although in this tutorial we illustrate parametrizations
with the `covariance_x` parameter in the `ProjNormal` class,
the same approach can be used to define constraints for
any of the parameters in any of the model classes
provided by `projnormal`.
:::


### Example: Defining a custom constraint

In coarse terms, parametrizations work by taking an
unconstrained parameter $\eta$ and transforming it into a
constrained model parameter $\theta$, via a function
$f(\eta) = \theta$. Let's illustrate this with the example of
the diagonal constraint.

In this case, the constrained parameter $\theta$ is an
$n$-by-$n$ diagonal matrix with positive diagonal elements.
One possible choice of $\eta$ is an unconstrained vector of length $n$.
The function $f(\eta)$ can be defined with the following two
steps:
1. Map each unconstrained element of $\eta$ to a positive value
   using the exponential function
2. Construct a diagonal matrix with the positive values
   from step 1 as the diagonal elements

To use such parametrization, we need to implement it
as a class that inherits from `torch.nn.Module`, where
the `forward` method implements the function $f(\eta)$.
Optionally, we can also implement the `right_inverse` method,
which maps from the constrained parameter $\theta$ back to the
unconstrained parameter $\eta$.
This is useful for assigning values to parameter $\theta$ in the
model. Also, if `right_inverse` is not used, the
parametrization assumes that the unconstrained parameter $\eta$
has the same shape as the constrained parameter $\theta$,
which is not the case for the example discussed.

Taking the above into account, we can implement the
`MyDiagonal()` constraint as follows:

```{code-cell} ipython3
import torch.nn as nn

class MyDiagonal(nn.Module):
    """Diagonal matrix with positive diagonal entries."""

    def forward(self, eta):
        diagonal_pos = torch.exp(eta)
        theta = torch.diag(diagonal_pos)
        return theta

    def right_inverse(self, theta):
        diagonal_pos = torch.diagonal(theta)
        eta = torch.log(diagonal_pos)
        return eta

# Initialize projected normal class
pn_fit2 = projnormal.classes.ProjNormal(
  n_dim=N_DIM
)

# Remove existing parametrization for covariance_x
parametrize.remove_parametrizations(
    pn_fit2, "covariance_x"
)

# Add our new parametrization
parametrize.register_parametrization(
    pn_fit2, "covariance_x", MyDiagonal()
)

print(pn_fit2)
```

Now the model `pn_fit2` has the `MyDiagonal()` constraint
on the `covariance_x` parameter.

Importantly, after registering a parametrization,
our model can be used as usual. Although now
`covariance_x` is parametrized in terms of
an unconstrained parameter `eta`, we still only
see `covariance_x` when accessing the model
normally. For more information, see the
[PyTorch parametrizations tutorial](https://docs.pytorch.org/tutorials/intermediate/parametrizations.html).

:::{admonition} The `geotorch` package provides useful constraints
For the symmetric positive definite constraint, `projnormal`
uses the package `geotorch`, which implements a variety
of constraints as PyTorch parametrizations.
See the [geotorch documentation](https://geotorch.readthedocs.io/en/latest/)
for more information.
:::


### Example: Defining advanced constraints

To aid the user in defining advanced constraints,
we provide a more elaborate example below to
illustrate the flexibility allowed by
PyTorch parametrizations.

Specifically, we will define a class to constrain
a matrix $\Sigma$ to be of the form
$\Sigma = \lambda \cdot \mathbf{I} + \sum_i^k \beta_i \mathbf{v}_i\mathbf{v}_i^T$,
where $\mathbf{I}$ is the identity matrix,
$\lambda > 0$, $\mathbf{v}_i$ are a set of basis vectors,
and $\beta_i$ are positive scalars that weight the
rank-1 matrices $\mathbf{v}_i\mathbf{v}_i^T$.
In this parametrization, the learnable parameters are
$\lambda$ and the $\beta_i$, while the $\mathbf{v}_i$ are fixed.

To implement this constraint, we will make both
$\mathbf{I}$ and the $\mathbf{v}_i$ (non-learnable)
attributes of the parametrization class.
We will add an `n_basis` argument to the class,
which specifies the number of basis vectors $\mathbf{v}_i$.
We will also make the learnable parameter
$\mathrm{log}(\lambda)$ an attribute of the class.

The code below implements this constraint:

```{code-cell} ipython3
class MyParametrization(torch.nn.Module):

    def __init__(self, n_dim, n_basis=2):
        super().__init__()
        self.n_dim = n_dim
        self.n_basis = n_basis

        # Make basis functions v_i and store as attribute
        x = torch.linspace(0, 1, n_dim)
        basis = torch.stack(
          [torch.sin((i + 1) * torch.pi * x) for i in range(n_basis)], dim=0
        )
        self.register_buffer("basis", basis)

        # Make identity matrix and store as buffer
        self.register_buffer("Id", torch.eye(n_dim))

        # Make learnable parameter log_lambda
        self.log_lambda = nn.Parameter(torch.tensor(0.0))


    def forward(self, betas):
        # Compute lambda * I
        diag = torch.exp(self.log_lambda) * self.Id

        # Compute the sum of the rank-1 matrices
        basis_mat = torch.einsum('ki,k,kj->ij', self.basis, betas, self.basis)

        # Return the full matrix
        theta = diag + basis_mat
        return theta

    def right_inverse(self, theta):

        # Extract the basis coefs from the matrix
        betas = torch.einsum('ki,ij,kj->k', self.basis, theta, self.basis) - \
          torch.exp(self.log_lambda)

        return betas


# Initialize projected normal class
pn_fit3 = projnormal.classes.ProjNormal(
  n_dim=6,
)

# Remove existing parametrization for covariance_x
parametrize.remove_parametrizations(
    pn_fit3, "covariance_x"
)

# Add our new parametrization
parametrize.register_parametrization(
    pn_fit3, "covariance_x", MyParametrization(n_dim=6, n_basis=2)
)

print(pn_fit3)
```

Now the `pn_fit3` model has the custom parametrization described,
and the parameters `log_lambda` and `betas` 
can be fit to data as usual.
