"""Functions for randomly sampling distribution parameters."""
import torch

__all__ = [
  "make_spdm",
  "make_mean",
  "make_ortho_vectors"
]


def __dir__():
    return __all__


def _make_orthogonal_matrix(n_dim):
    """Generate random orthogonal matrix."""
    matrix = torch.randn(n_dim, n_dim)
    low_tri = torch.tril(matrix, diagonal=-1)
    skew_sym = low_tri - low_tri.T
    orthogonal = torch.linalg.matrix_exp(skew_sym)
    return orthogonal


def make_spdm(n_dim, eigvals='uniform', eigvecs='random'):
    """Make a symmetric positive definite matrix.

    Parameters
    ----------
      n_dim : ``int``
          Dimension of matrix

      eigvals : ``str`` or ``torch.Tensor``
          Eigenvalues of the matrix. Options are: 1) Tensor of eigvals to use, of
          length ``n_dim``. 2) `'uniform'`: Eigvals are uniformly sampled
          between 0.1 and 1. 3) ``'exponential'``: Eigvals sampled from Exp(1).

      eigvecs : str
          Eigenvectors of the matrix. Options are: 1)``'random'``: Random orthogonal matrix.
          2) ``'identity'``: Identity matrix.

    Returns
    -------
      ``torch.Tensor``
          Symmetric positive definite matrix with specified eigvals. Shape is ``(n_dim, n_dim)``.
    """
    # Generate eigvals
    if isinstance(eigvals, str):
        if eigvals == 'uniform':
            eigvals = torch.rand(n_dim) * 0.95 + 0.05
            eigvals = eigvals / eigvals.mean()
        elif eigvals == 'exponential':
            u = torch.rand(n_dim)
            eigvals = - torch.log(u) + 0.01
            eigvals = eigvals / eigvals.mean()
        else:
            raise ValueError("Invalid eigenvalue option.")
    else:
        eigvals = torch.as_tensor(eigvals)

    # Generate eigvecs and make spd matrix
    if eigvecs == 'random':
        eigvecs = _make_orthogonal_matrix(n_dim)
        spdm = torch.einsum('ij,j,jk->ik', eigvecs, eigvals, eigvecs.T)
    elif eigvecs == 'identity':
        spdm = torch.diag(eigvals)
    else:
        raise ValueError("Invalid eigenvector option.")

    return spdm


def make_mean(n_dim, shape='gaussian', sparsity=0.1):
    """Generate a vector to use as the mean of a multivariate normal.

    Parameters
    ----------
      n_dim : ``int``
          Dimension of the mean vector.

      shape : ``str``
          Type of mean vector. Options are:
          ``'gaussian'`` (each element sampled from N(0,1)),
          ``'exponential'`` (each element sampled from Exp(1)),
          ``'sin'`` (sin-wave vector with random phase frequency and amplitude),
          ``'sparse'`` (sparse vector with 0s and 1s)

      sparsity: ``float``
        For ``'sparse'`` shape, the fraction of non-zero elements

    Returns
    -------
      ``torch.Tensor``
          Mean vector. Shape is ``(n_dim,)``.
    """
    if shape == 'gaussian':
        mean = torch.randn(n_dim)
    elif shape == 'exponential':
        u = torch.rand(n_dim)
        mean = - torch.log(u)
    elif shape == 'sin':
        x = torch.linspace(0, 2 * torch.pi, n_dim)
        phase = torch.rand(1) * torch.pi*2
        freq = torch.rand(1) * 2
        amplitude = torch.rand(1)*0.9 + 0.1
        mean = torch.sin(x * freq + phase) * amplitude
    elif shape == 'sparse':
        mean = torch.zeros(n_dim)
        n_nonzero = int(torch.ceil(torch.as_tensor(n_dim * sparsity)))
        indices = torch.randperm(n_dim)[:n_nonzero]
        mean[indices] = 1
    else:
        raise ValueError("Invalid shape option.")
    mean = mean / mean.norm()
    return mean


def make_ortho_vectors(n_dim, n_vec):
    """Generate a set of orthogonal vectors.

    Parameters
    ----------
      n_dim : ``int``
          Dimension of the vectors.

      n_vec : ``int``
          Number of orthogonal vectors to generate. Must be less than n_dim.

    Returns
    -------
      ``torch.Tensor``
          Orthogonal vectors of size n_dim x n_vec. Shape is ``(n_vec, n_dim)``.
    """
    if n_vec > n_dim:
        raise ValueError("Number of vectors must be less than dimension.")
    vectors = torch.randn(n_dim, n_vec)
    vectors = torch.linalg.qr(vectors)[0].t()
    return vectors
