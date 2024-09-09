##################################
##################################
#
## PROJECTED NORMAL CLASS FOR FITTING
#
##################################
##################################

#### Terminology:
# X : Random variable with multidimensional Gaussian distribution
# mu : Mean of X
# sigma : Standard deviation of X (if isotropic)
# covariance : Covariance of X
# n_dim : Dimensions of X
# Y : Random variable with projected Gaussian distribution
# gamma : Mean of Y
# psi : Covariance of Y

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.linalg as LA
import projected_normal.prnorm_func as pnf
import projected_normal.auxiliary as pna
import projected_normal.parametrizations as pnp


##################################
## PROJECTED NORMAL
##################################


#### Class for the projected normal distribution with learnable parameters
class ProjectedNormal(nn.Module):
    """
    The ProjNorm class implements a projected normal distirbution
    X/||X||, where X~N(mu, covariance). The class has methods to
    sampled, fit to data, and compute its moments.
    ----------------
    Inputs:
    ----------------
      - n_dim : Dimension of the distribution.
      - mu : Mean of X. It is converted to unit norm. Default is random.
      - cov : Initial covariance. Default is the identity.
      - requires_grad : Whether the parameters are learnable. Default is True.
      - dtype : Data type of the parameters. Default is torch.float32.
      - covariance_parametrization : Parametrization of the covariance matrix
          for learnable parameters. Options are 'LogCholesky', 'Logarithm'
          and 'Spectral'
    ----------------
    Attributes:
    ----------------
      - mu : Mean of X. Shape (n). Constrained to the unit sphere.
      - covariance : Covariance of X. Shape (n x n).
    ----------------
    Methods:
    ----------------
      - log_pdf : Compute the log probability of observed points.
      - pdf: Compute the probability of observed points.
      - moments_empirical : Compute the mean and covariance the projected
                        normal by sampling from the distribution.
      - moments_approx : Compute the Taylor approximation to the mean and covariance
                        of the projected normal.
      - sample : Sample from the distribution.
      - fit: Fit the distribution to observed data. Requires a loss function
          that determines what data is taken as input and how the loss is computed.
    """

    def __init__(
        self,
        n_dim,
        mu=None,
        covariance=None,
        requires_grad=True,
        dtype=torch.float32,
        covariance_parametrization="Spectral",
    ):
        super().__init__()

        # Initialize parameters
        if mu is None:
            mu = torch.randn(n_dim, dtype=dtype)
            mu = mu / LA.vector_norm(mu)
        else:
            mu = torch.as_tensor(mu, dtype=dtype)
            mu = mu / LA.vector_norm(mu)

        if covariance is None:
            covariance = torch.eye(n_dim, dtype=dtype)
        else:
            covariance = torch.as_tensor(covariance, dtype=dtype)

        # Convert to parametrized nn.Parameters
        if requires_grad:
            # Initialize parameters
            self.mu = nn.Parameter(mu.clone())
            self.covariance = nn.Parameter(covariance.clone())

            # Register mu parametrization
            parametrize.register_parametrization(self, "mu", pnp.Sphere())
            # Register covariance parametrization
            scale = torch.trace(covariance) / torch.sqrt(
                torch.tensor(n_dim)
            )  # Scale compared to the identity
            if covariance_parametrization == "LogCholesky":
                parametrize.register_parametrization(
                    self, "covariance", pnp.SPDLogCholesky(dtype=dtype)
                )
            if covariance_parametrization == "SoftmaxCholesky":
                parametrize.register_parametrization(
                    self, "covariance", pnp.SPDSoftmaxCholesky(dtype=dtype)
                )
            elif covariance_parametrization == "Logarithm":
                parametrize.register_parametrization(
                    self, "covariance", pnp.SPDMatrixLog(scale=scale, dtype=dtype)
                )
            elif covariance_parametrization == "Spectral":
                parametrize.register_parametrization(
                    self, "covariance", pnp.SPDSpectral(dtype=dtype)
                )

        else:
            self.mu = mu
            self.covariance = covariance

        # Set C50 and B to default constants
        self.c50 = torch.tensor(0, dtype=dtype)
        self.B_diagonal = torch.ones(n_dim, dtype=dtype)

    def moments_approx(self):
        """
        Compute the approximated mean (gamma) and covariance (psi) for the
        projected normal.
        ----------------
        Outputs:
        ----------------
          - gamma : Mean of projected normal. Shape (n_dim).
          - psi : Covariance of projected normal. Shape (n_dim x n_dim).
        """

        gamma = pnf.prnorm_mean_taylor(
            mu=self.mu,
            covariance=self.covariance,
            B_diagonal=self.B_diagonal,
            c50=self.c50,
        )

        second_moment = pnf.prnorm_sm_taylor(
            mu=self.mu,
            covariance=self.covariance,
            B_diagonal=self.B_diagonal,
            c50=self.c50,
        )

        psi = pna.second_moment_2_cov(second_moment=second_moment, mean=gamma)

        return {"gamma": gamma, "psi": psi, "second_moment": second_moment}

    def moments_empirical(self, n_samples=200000):
        """Compute the mean and covariance the normalized (Y) mean and covariance
        by sampling from the distribution.
        ----------------
        Inputs:
        ----------------
          - n_samples: Number of samples to draw.
        ----------------
        Outputs:
        ----------------
          - gamma: Empirical mean. Shape (n).
          - psi: Empirical covariance. Shape (n x n).
        """
        with torch.no_grad():
            stats_dict = pnf.empirical_moments_prnorm(
                mu=self.mu,
                covariance=self.covariance,
                B=self.B_diagonal,
                c50=self.c50,
                n_samples=n_samples,
            )
        return stats_dict

    def log_pdf(self, y):
        """
        Compute the log probability density of a given point under the projected
        normal distribution.
        ----------------
        Inputs:
        ----------------
          - y : Points to evaluate. Shape (n_points x n_dim).
        ----------------
        Outputs:
        ----------------
          - lpdf : Log probability of the point. (n_points)
        """
        # Make input same dtype as the parameters
        y = y.to(self.mu.dtype)
        # Compute the log probability
        lpdf = pnf.prnorm_log_pdf(mu=self.mu, covariance=self.covariance, y=y)
        return lpdf

    def pdf(self, y):
        """
        Compute the probability density of a given point under the projected
        normal distribution.
        ----------------
        Inputs:
        ----------------
          - y : Points to evaluate. Shape (n_points x n_dim).
        ----------------
        Outputs:
        ----------------
          - pdf : Log probability of the point. (n_points)
        """
        # Make input same dtype as the parameters
        y = y.to(self.mu.dtype)
        # Compute the log probability
        lpdf = pnf.prnorm_pdf(mu=self.mu, covariance=self.covariance, y=y)
        return lpdf

    def sample(self, n_samples):
        """Sample from the distribution.
        ----------------
        Inputs:
        ----------------
          - n_samples : Number of samples to draw.
        ----------------
        Outputs:
        ----------------
          - samples_prnorm : Samples from the distribution. Shape (n_samples x n).
        """
        with torch.no_grad():
            samples_prnorm = pnf.sample_prnorm(
                mu=self.mu,
                covariance=self.covariance,
                B=self.B_diagonal,
                c50=self.c50,
                n_samples=n_samples,
            )
        return samples_prnorm

    def initialize_optimizer_and_scheduler(self, lr=0.1, lr_gamma=0.7, decay_iter=10):
        """
        Initialize the optimizer and learning rate scheduler for training.
        ----------------
        Inputs:
        ----------------
          - lr : Learning rate for the optimizer. Default is 0.1.
          - lr_gamma : Multiplicative factor for learning rate decay. Default is 0.7.
          - decay_iter : Number of iterations after which the learning rate is decayed. Default is 10.
        ----------------
        Outputs:
        ----------------
          - optimizer : Initialized NAdam optimizer with the specified learning rate.
          - scheduler : StepLR scheduler that decays the learning rate every `decay_iter` iterations.
        """
        # Initialize the optimizer
        optimizer = torch.optim.NAdam(self.parameters(), lr=lr)
        # Initialize the scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=decay_iter, gamma=lr_gamma
        )
        return optimizer, scheduler

    def fit(self, data, optimizer, loss_function, scheduler=None, n_iter=100):
        """
        Perform a training cycle to optimize the model parameters.
        ----------------
        Inputs:
        ----------------
          - data : Observed data used to compute the loss. Must be in a format compatible with `loss_function`.
          - optimizer : Optimizer to use for parameter updates.
          - loss_function : Function that computes the loss. Takes `self` and `data` as input.
          - scheduler : Learning rate scheduler to adjust the learning rate during training. Optional.
          - n_iter : Number of iterations for the training cycle. Default is 100.
        ----------------
        Outputs:
        ----------------
          - loss_list : List of loss values recorded at each iteration.
        """
        loss_list = []
        for i in range(n_iter):
            # Zero the gradients
            optimizer.zero_grad()
            # Compute the loss
            loss = loss_function(self, data)
            # Compute the gradients
            loss.backward()
            # Optimize the parameters
            optimizer.step()
            # Step the scheduler
            if scheduler is not None:
                scheduler.step()
            # Append the loss to the list
            loss_list.append(loss.item())
        return torch.tensor(loss_list)


##################################
## PROJECTED NORMAL WITH C50
##################################


class ProjectedNormalC50(ProjectedNormal):
    """
    The ProjNormC50 class implements a projected normal distirbution
    X/sqrt(X'X + c50), where X~N(mu, covariance). The class has methods
    to sampled, fit to data, and compute its moments.
    ----------------
    Inputs:
    ----------------
      - n_dim : Dimension of the distribution.
      - mu : Mean of X. It is converted to unit norm. Default is random.
      - cov : Initial covariance. Default is the identity.
      - c50 : Constant to add to the denominator. Default is 1.
      - requires_grad : Whether the parameters are learnable. Default is True.
      - dtype : Data type of the parameters. Default is torch.float32.
      - covariance_parametrization : Parametrization of the covariance matrix
          for learnable parameters. Options are 'LogCholesky' and 'Logarithm'.
    ----------------
    Attributes:
    ----------------
      - mu : Mean of X. Shape (n). Constrained to the unit sphere.
      - covariance : Covariance of X. Shape (n x n).
    ----------------
    Methods:
    ----------------
      - moment_match : Optimize distribution parameters to match observed moments
      - ml_fit : Fit the distribution to observed data maximizing the log likelihood
      - moments_approx : Compute the Taylor approximation to the mean and covariance
                        of the projected normal.
      - log_pdf : Compute the log probability of observed points.
      - pdf: Compute the probability of observed points.
      - sample : Sample from the distribution.
      - moments_empirical : Compute the mean and covariance the projected
                        normal by sampling from the distribution.
    """

    def __init__(
        self,
        n_dim,
        mu=None,
        covariance=None,
        c50=None,
        requires_grad=True,
        dtype=torch.float32,
        covariance_parametrization="Logarithm",
    ):
        super().__init__(
            n_dim=n_dim,
            mu=mu,
            covariance=covariance,
            requires_grad=requires_grad,
            dtype=dtype,
            covariance_parametrization=covariance_parametrization,
        )
        if c50 is None:
            c50 = torch.tensor(1, dtype=dtype)
        else:
            c50 = torch.as_tensor(c50, dtype=dtype)

        if requires_grad:
            self.c50 = nn.Parameter(c50.clone())
            parametrize.register_parametrization(self, "c50", pnp.SoftMax())
        else:
            self.c50 = c50

    def log_pdf(self, y):
        """
        Compute the log probability density of a given point under the projected
        normal distribution.
        ----------------
        Inputs:
        ----------------
          - y : Points to evaluate. Shape (n_points x n_dim).
        ----------------
        Outputs:
        ----------------
          - lpdf : Log probability of the point. (n_points)
        """
        # Check that c50 is positive
        assert self.c50 > 0, "C50 must be positive for the pdf to be valid."
        # Make input same dtype as the parameters
        y = y.to(self.mu.dtype)
        # Compute the log probability
        lpdf = pnf.prnorm_c50_log_pdf(
            mu=self.mu, covariance=self.covariance, c50=self.c50, y=y
        )
        return lpdf

    def pdf(self, y):
        """
        Compute the probability density of a given point under the projected
        normal distribution.
        ----------------
        Inputs:
        ----------------
          - y : Points to evaluate. Shape (n_points x n_dim).
        ----------------
        Outputs:
        ----------------
          - pdf : Log probability of the point. (n_points)
        """
        # Check that c50 is positive
        assert self.c50 > 0, "C50 must be positive for the pdf to be valid."
        # Make input same dtype as the parameters
        y = y.to(self.mu.dtype)
        # Compute the log probability
        lpdf = pnf.prnorm_c50_pdf(
            mu=self.mu, covariance=self.covariance, c50=self.c50, y=y
        )
        return lpdf


############### Loss functions ###################


def loss_mm_norm(model, data):
    """Compute the norm of the difference between the observed and predicted moments.
    ----------------
    Inputs:
    ----------------
      - model: ProjectedNormal model.
      - data: Dictionary with the observed moments.
    ----------------
    Outputs:
    ----------------
      - loss: Loss between the observed and predicted moments.
    """
    taylor_moments = model.moments_approx()
    gamma_norm = LA.vector_norm(taylor_moments["gamma"] - data["gamma"])
    psi_norm = LA.matrix_norm(taylor_moments["psi"] - data["psi"])
    loss = gamma_norm + psi_norm
    return loss


def loss_mm_mse(model, data):
    """Compute the mean squared error between the observed and predicted moments.
    ----------------
    Inputs:
    ----------------
      - model: ProjectedNormal model.
      - data: Dictionary with the observed moments.
    ----------------
    Outputs:
    ----------------
      - loss: Loss between the observed and predicted moments.
    """
    taylor_moments = model.moments_approx()
    gamma_mse = torch.sum((taylor_moments["gamma"] - data["gamma"]) ** 2)
    psi_mse = torch.sum((taylor_moments["psi"] - data["psi"]) ** 2)
    loss = gamma_mse + psi_mse
    return loss


def loss_mm_norm_sm(model, data):
    """Compute the norm of the difference between the observed and predicted
    moments, using the second moment matrix instead of the covariance.
    ----------------
    Inputs:
    ----------------
      - model: ProjectedNormal model.
      - data: Dictionary with the observed moments.
    ----------------
    Outputs:
    ----------------
      - loss: Loss between the observed and predicted moments.
    """
    sm_observed = pna.cov_2_second_moment(covariance=data["psi"], mean=data["gamma"])
    taylor_moments = model.moments_approx()
    gamma_norm = LA.vector_norm(taylor_moments["gamma"] - data["gamma"])
    sm_norm = LA.matrix_norm(taylor_moments["second_moment"] - sm_observed)
    loss = gamma_norm + sm_norm
    return loss


def loss_mm_mse_sm(model, data):
    """Compute the mean squared error between the observed and predicted
    moments, using the second moment matrix instead of the covariance.
    ----------------
    Inputs:
    ----------------
      - model: ProjectedNormal model.
      - data: Dictionary with the observed moments.
    ----------------
    Outputs:
    ----------------
      - loss: Loss between the observed and predicted moments.
    """
    sm_observed = pna.cov_2_second_moment(covariance=data["psi"], mean=data["gamma"])
    taylor_moments = model.moments_approx()
    gamma_mse = torch.sum((taylor_moments["gamma"] - data["gamma"]) ** 2)
    sm_mse = torch.sum((taylor_moments["second_moment"] - sm_observed) ** 2)
    loss = gamma_mse + sm_mse
    return loss


def loss_log_pdf(model, data):
    """Compute the mean squared error between the observed and predicted moments.
    ----------------
    Inputs:
    ----------------
      - model: ProjectedNormal model.
      - data: Tensor with observed data points
    ----------------
    Outputs:
    ----------------
      - loss: Mean negative log likelihood of the data.
    """
    loss = torch.mean(-model.log_pdf(data))
    return loss
