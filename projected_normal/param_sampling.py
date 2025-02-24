import torch


__all__ = [
  "make_spdm",
  "make_mean",
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
    """ Make a symmetric positive definite matrix.

    Parameters:
    ----------------
      - n_dim: Dimension of matrix
      - eigvals: Eigenvalues of the matrix. Options are:
        - Tensor or list of eigvals to use, of length n_dim.
        - 'uniform': Uniformly distributed eigvals between 0.1 and 1.
        - 'exponential': Exponentially distributed eigvals with parameter 1
      - eigvecs: Eigenvectors of the matrix. Options are:
        - 'random': Random orthogonal matrix.
        - 'identity': Identity matrix.

    Returns:
    ----------------
      Symmetric positive definite matrix with specified eigvals.
    """
    # Generate eigvals
    if isinstance(eigvals, str):
        if eigvals == 'uniform':
            eigvals = torch.rand(n_dim) * 0.9 + 0.1
        elif eigvals == 'exponential':
            u = torch.rand(n_dim)
            eigvals = - torch.log(u)
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
    """ Generate a vector to use as the mean of a multivariate normal.

    Parameters:
    ----------------
      - n_dim: Dimension of the mean vector.
      - shape: Type of mean vector. Options are:
        - 'gaussian': Random vector with each element sampled from N(0,1)
        - 'exponential': Random vector with each element sampled from Exp(1)
        - 'sin': sin-wave shaped vector, with random phase, frequency and amplitude
            sampled uniformly from [0, 2pi], [0, 2] and [0.1, 1] respectively.
        - 'sparse': Sparse vector with 0s and 1s. The number of 1s is
          determined by the sparsity parameter.
      - sparsity: For 'sparse' shape, the fraction of non-zero elements

    Returns:
    ----------------
      Mean vector of size n_dim.
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
    return mean
