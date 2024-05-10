# Projected normal distribution

The projected normal distribution (PN) or angular Gaussian
is flexible distribution on the hypersphere, that is obtained by
projecting a multivariate normal distribution onto the
hypersphere. This is a known distribution
in the field of directional statistics.

This package implements a set of functionalities related to the PN
distribution and quadratic forms of random variables in Pytorch.
Some of the functionalities include:

- The pdf and log-pdf formulas for the PN
- Maximum-likelihood estimation of the parameters of the PN
- Sampling from the PN
- Exact formulas for the first and second moments of the PN in
  the case where the projected normal is isotropic
- Approximate formulas for the first and second moments of the PN
  in the general case
- Optimization procedures to fit the PN through moment matching

Most of the functions can also be used for a generalized
version of the PN, where the normal distribution is projected
onto an ellipse, and where a constant is added to the
denominator, so that the distribution is inside the unit
sphere (i.e. in the unit ball) rather than on the unit sphere.

Also, the package provides formulas for different moments of
quadratic forms of random variables.

This package is still under development, and the documentation
is not yet complete. Installation instructions and example
notebooks will be provided in the future.



