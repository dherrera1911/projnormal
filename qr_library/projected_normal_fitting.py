# Methods for fitting the parameters of the projected normal distribution
# using moment matching and the Taylor approximation.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import scipy.linalg
import geotorch
import qr_library as qr


#### Class for the projected normal distribution with learnable parameters
class ProjNorm(nn.Module):
    """
    The ProjNorm class implements a projected normal distirbution
    with several functionalities. The parameters of the distribution
    can be fitted to a given mean and covariance using the method fit.
    ----------------
    Inputs:
    ----------------
      - nDim: Dimension of the distribution.
      - muInit: Initial mean. Default is zero.
      - covInit: Initial covariance. Default is the identity.
    ----------------
    Attributes:
    ----------------
      - mu: Mean of the distribution. Shape (n).
            Constrained to the sphere.
      - cov: Covariance of the distribution. Shape (n x n).
            Constrained to the positive definite cone.
    ----------------
    Methods:
    ----------------
      - fit: Fit the parameters of the distribution to a given mean and covariance.
      - get_moments: Compute the Taylor approximation to the normalized (Y) mean
                    and covariance for the attribute mean and covariance.
      - sample: Sample from the distribution.
      - empiral_moments: Compute the mean and covariance the normalized
                    (Y) mean and covariance by sampling from the distribution.
    """
    def __init__(self, nDim, muInit=None, covInit=None, requires_grad=True):
        super().__init__()
        if muInit is None:
            muInit = torch.randn(nDim, dtype=torch.float64)
            muInit = muInit / torch.norm(muInit)
        if covInit is None:
            covInit = torch.eye(nDim, dtype=torch.float64)
        if requires_grad:
            self.mu = nn.Parameter(muInit.clone())
            geotorch.sphere(self, "mu")
            self.mu = muInit
            self.cov = nn.Parameter(covInit.clone())
            geotorch.positive_definite(self, "cov")
            self.cov = covInit
        else:
            self.mu = muInit
            self.cov = covInit


    def get_moments(self):
        """ Compute the normalized (Y) mean and covariance for the
        attribute mean and covariance.
        ----------------
        Outputs:
        ----------------
          - muOut: Normalized mean. Shape (n).
          - covOut: Normalized covariance. Shape (n x n).
        """
        muOut = qr.prnorm_mean_taylor(mu=self.mu, covariance=self.cov)
        smOut = qr.prnorm_sm_taylor(mu=self.mu, covariance=self.cov)
        covOut = qr.secondM_2_cov(secondM=smOut, mean=muOut)
        return muOut, covOut


    def fit(self, muObs, covObs, nIter=100, lr=0.1, lrGamma=0.7, decayIter=10,
            lossFun="norm"):
        """ Fit the parameters of the distribution to a given mean and covariance.
        ----------------
        Inputs:
        ----------------
          - mu: Mean to fit. Shape (n).
          - cov: Covariance to fit. Shape (n x n).
          - nIter: Number of iterations. Default is 100.
          - lr: Learning rate. Default is 0.01.
          - lrGamma: Learning rate decay. Default is 0.8.
          - decayIter: Decay the learning rate every decayIter iterations. Default is 10.
          - lossFun: Loss function to use. Options are "norm", "mse", and "wasserstein".
        ----------------
        Outputs:
        ----------------
          - lossVec: Vector of losses at each iteration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decayIter, gamma=lrGamma)
        lossVec = torch.zeros(nIter)
        for i in range(nIter):
            optimizer.zero_grad()
            muOut, covOut = self.get_moments()
            if lossFun == "norm":
                loss = loss_norm(mu1=muOut, cov1=covOut, mu2=muObs, cov2=covObs)
            elif lossFun == "mse":
                loss = loss_mse(mu1=muOut, cov1=covOut, mu2=muObs, cov2=covObs)
            elif lossFun == "wasserstein":
                loss = loss_wasserstein(mu1=muOut, cov1=covOut, mu2=muObs, cov2=covObs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Store the loss
            lossVec[i] = loss.item()
        return lossVec


    def sample(self, nSamples):
        """ Sample from the distribution.
        ----------------
        Inputs:
        ----------------
          - nSamples: Number of samples to draw.
        ----------------
        Outputs:
        ----------------
          - samples: Samples from the distribution. Shape (nSamples x n).
        """
        with torch.no_grad():
            samples = qr.sample_prnorm(mu=self.mu, covariance=self.cov,
                                       B=None, c50=0, nSamples=nSamples)
        return samples


    def empirical_moments(self, nSamples):
        """ Compute the mean and covariance the normalized (Y) mean and covariance
        by sampling from the distribution.
        ----------------
        Inputs:
        ----------------
          - nSamples: Number of samples to draw.
        ----------------
        Outputs:
        ----------------
          - muOut: Empirical mean. Shape (n).
          - covOut: Empirical covariance. Shape (n x n).
        """
        with torch.no_grad():
            statsDict = qr.empirical_moments_prnorm(mu=self.mu, covariance=self.cov,
                                                    B=None, c50=0, nSamples=nSamples)
        muOut = statsDict["mean"]
        covOut = statsDict["covariance"]
        return muOut, covOut


def loss_norm(mu1, cov1, mu2, cov2):
    """ Compute the norm of the difference between the observed and predicted moments.
    ----------------
    Inputs:
    ----------------
      - mu: Predicted mean. Shape (n).
      - cov: Predicted covariance. Shape (n x n).
      - muObs: Observed mean. Shape (n).
      - covObs: Observed covariance. Shape (n x n).
    ----------------
    Outputs:
    ----------------
      - loss: Loss between the observed and predicted moments.
    """
    loss = (mu1 - mu2).norm() + (cov1 - cov2).norm()
    return loss


def loss_mse(mu1, cov1, mu2, cov2):
    """ Compute the mean squared error between the observed and predicted moments.
    ----------------
    Inputs:
    ----------------
      - mu: Predicted mean. Shape (n).
      - cov: Predicted covariance. Shape (n x n).
      - muObs: Observed mean. Shape (n).
      - covObs: Observed covariance. Shape (n x n).
    ----------------
    Outputs:
    ----------------
      - loss: Loss between the observed and predicted moments.
    """
    loss = (mu1 - mu2).pow(2).sum() + (cov1 - cov2).pow(2).sum()
    return loss


def loss_wasserstein(mu1, cov1, mu2, cov2):
    """ Compute the Wasserstein distance between the observed and predicted moments.
    ----------------
    Inputs:
    ----------------
      - mu: Predicted mean. Shape (n).
      - cov: Predicted covariance. Shape (n x n).
      - muObs: Observed mean. Shape (n).
      - covObs: Observed covariance. Shape (n x n).
    ----------------
    Outputs:
    ----------------
      - loss: Loss between the observed and predicted moments.
    """
    loss = (mu1 - mu2).pow(2).sum() + bw_dist_sq(cov1, cov2)
    return loss


# Make a function that computes the matrix square root of a positive
# definite matrix, with a backward pass.
class MatrixSquareRoot(torch.autograd.Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)
            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input

# Create the function to compute the matrix square root.
sqrtm = MatrixSquareRoot.apply

def bw_dist_sq(mat1, mat2):
    """ Wasserstein distance between two positive definite matrices
    ----------------
    Inputs:
    ----------------
      - mat1: Positive definite matrix. Shape (n x n).
      - mat2: Positive definite matrix. Shape (n x n).
    ----------------
    Outputs:
    ----------------
      - squared_dist: Squared Wasserstein distance between the matrices.
    """
    product = torch.matmul(mat1, mat2)
    sqrt_product = sqrtm(product)
    trace_a = torch.trace(mat1)
    trace_b = torch.trace(mat2)
    trace_prod = torch.trace(sqrt_product)
    squared_dist = trace_a + trace_b - 2.0 * trace_prod
    return squared_dist


