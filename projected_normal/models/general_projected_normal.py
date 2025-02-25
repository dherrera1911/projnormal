"""Class for the general projected normal distribution."""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import projected_normal.distribution as prnorm
from ._constraints import Sphere, SPD
from ._optim import lbfgs_loop, nadam_loop


__all__ = [
  "ProjectedNormal",
]


def __dir__():
    return __all__


#### Class for the projected normal distribution with learnable parameters
class ProjectedNormal(nn.Module):
    """
    This class implements the general projected normal distirbution,
    which described the variable Y= X/||X||, where X~N(mean_x, covariance_x).
    The class can be used to fit distribution parameters to data.

    Attributes
    -----------
      mean_x : torch.Tensor, shape (n_dim)
          Mean of X. It is constrained to the unit sphere.

      covariance_x : torch.Tensor, shape (n_dim x n_dim)
          Covariance of X. It is constrained to be symmetric positive definite.

    Methods
    ----------
      moments():
          Compute the moments using a Taylor approximation.

      log_pdf() :
          Compute the value of the log pdf at given points.

      pdf() :
          Compute the value of the pdf at given points.

      moments_empirical() :
          Compute the moments by sampling from the distribution

      sample() :
          Sample points from the distribution.

      moment_match() :
          Fit the distribution parameters to the observed moments.

      max_likelihood() :
          Fit the distribution parameters to the observed data
          using maximum likelihood.
    """

    def __init__(
        self,
        n_dim=None,
        mean_x=None,
        covariance_x=None,
    ):
        """Initialize an instance of the ProjectedNormal class.

        Parameters
        ------------
          n_dim : int, optional
            Dimension of the underlying Gaussian distribution. If mean
            and covariance are provided, this is not required.

          mean : torch.Tensor, shape (n_dim), optional
            Mean of X. It is converted to unit norm. Default is random.

          covariance : torch.Tensor, shape (n_dim x n_dim), optional
            Initial covariance. Default is the identity.
        """
        super().__init__()
        # Initialize parameters
        if n_dim is None:
            if mean_x is None or covariance_x is None:
                raise ValueError("Either n_dim or mean and covariance must be provided.")
            else:
                n_dim = mean_x.shape[0]
        self.n_dim = n_dim
        if mean_x is None:
            mean_x = torch.randn(self.n_dim)
        elif mean_x.shape[0] != self.n_dim:
            raise ValueError("The input mean does not match n_dim")
        if covariance_x is None:
            covariance_x = torch.eye(self.n_dim)
        elif covariance_x.shape[0] != self.n_dim or covariance_x.shape[1] != self.n_dim:
            raise ValueError("The input covariance does not match n_dim")
        self.mean_x = nn.Parameter(mean_x)
        self.covariance_x = nn.Parameter(covariance_x)
        parametrize.register_parametrization(self, "mean_x", Sphere())
        parametrize.register_parametrization(self, "covariance_x", SPD())
        # Add c50 as buffer set to 0 to make child models easier
        self.register_buffer("c50", torch.tensor(0), persistent=False)


    def moments(self):
        """
        Compute the Taylor approximation to the moments
        of the variable Y = X/||X||, where X~N(mean_x, covariance_x).

        Returns
        ---------
        dict
          Dictionary containing the mean, covariance and second moment
          of the projected normal.
        """
        gamma = prnorm.c50.moments.mean(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            c50=self.c50,
        )
        second_moment = prnorm.c50.moments.second_moment(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            c50=self.c50,
        )
        cov = second_moment - torch.einsum("i,j->ij", gamma, gamma)
        return {"mean": gamma, "covariance": cov, "second_moment": second_moment}


    def moments_empirical(self, n_samples=200000):
        """
        Compute the moments of the variable Y = X/||X||, where X~N(mean_x, covariance_x),
        by sampling from the distribution.

        Returns
        ----------------
        dict
          Dictionary containing the mean, covariance and second moment
          of the projected normal.
        """
        with torch.no_grad():
            stats_dict = prnorm.c50.sampling.empirical_moments(
                mean_x=self.mean_x,
                covariance_x=self.covariance_x,
                n_samples=n_samples,
                c50=self.c50,
            )
        return stats_dict


    def log_pdf(self, y):
        """
        Compute the log pdf at points y under the projected normal distribution.

        Parameters 
        ----------------
          y : torch.Tensor, shape (n_points x n_dim)
              Points to evaluate

        Outputs:
        ----------------
          torch.Tensor, shape (n_points)
              Log PDF of the point. (n_points)
        """
        lpdf = prnorm.general.pdf.log_pdf(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            y=y,
        )
        return lpdf


    def pdf(self, y):
        """
        Compute the pdf at points y under the projected normal distribution.

        Parameters 
        ----------------
          y : torch.Tensor, shape (n_points x n_dim)
              Points to evaluate

        Outputs:
        ----------------
          torch.Tensor, shape (n_points)
              PDF of the point. (n_points)
        """
        pdf = prnorm.general.pdf.pdf(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            y=y,
        )
        return pdf


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
            samples = prnorm.c50.sampling.sample(
                mean_x=self.mean_x,
                covariance_x=self.covariance_x,
                n_samples=n_samples,
                c50=self.c50,
            )
        return samples


    def moment_match(
        self,
        data_moments,
        max_epochs=200,
        lr=0.1,
        optimizer="NAdam",
        loss_fun=None,
        show_progress=True,
        return_loss=False,
        n_cycles=3,
        cycle_gamma=0.5,
        **kwargs,
    ):
        """
        Fit the distribution parameters through moment matching.

        Parameters
        ----------------
        data_moments : dict
          Dictionary containing the observed moments. Must contain the keys
            - 'mean': torch.Tensor, shape (n_dim)
            - 'covariance': torch.Tensor, shape (n_dim x n_dim)
            - 'second_moment': torch.Tensor, shape (n_dim x n_dim)

        max_epochs : int, optional
            Number of max training epochs. By default 50.

        lr : float
            Learning rate for the optimizer. Default is 0.1.

        optimizer : str
            Optimizer to use for training. Options are 'LBFGS' and 'NAdam'.
            Default is 'NAdam'.

        loss_fun : callable
            Loss function to use for moment matching. Default is MSE loss.

        show_progress : bool
            If True, show a progress bar during training. Default is True.

        return_loss : bool
            If True, return the loss after training. Default is False.

        n_cycles : int
            For the NAdam optimier, the number of times to run the optimization loop.

        cycle_gamma : float
            For the NAdam optimizer, the factor by which lr is reduced after each run
            of the optimization loop.

        **kwargs
            Additional keyword arguments passed to the lbfgs_lopp or nadam_loop function.
            For the NAdam optimizer, the parameters `gamma` and `step_size` can be passed
            to control the learning rate schedule.

        Returns
        ----------------
        dict
          Dictionary containing the loss and training time.
        """
        # Check data_moments is a dictionary
        if not isinstance(data_moments, dict):
            raise ValueError("Data must be a dictionary.")

        # Check if the data is complete
        if not all(key in data_moments for key in ["mean", "covariance", "second_moment"]):
            raise ValueError(
              "Data must contain the keys 'mean', 'covariance' and 'second_moment'."
            )

        if optimizer == "NAdam":
            loss = []
            training_time = []
            # Run the NAdam optimizer for n_cycles
            for c in range(n_cycles):
                lr_cycle = lr * cycle_gamma ** c
                loss_cycle, training_time_cycle = nadam_loop(
                    model=self,
                    data=data_moments,
                    fit_type="mm",
                    max_epochs=max_epochs,
                    lr=lr_cycle,
                    loss_fun=loss_fun,
                    show_progress=show_progress,
                    return_loss=True,
                    **kwargs,
                )
                loss.append(loss_cycle)
                if c == 0:
                    training_time.append(training_time_cycle)
                else:
                    training_time.append(training_time_cycle + training_time[-1])

        elif optimizer == "LBFGS":
            loss, training_time = lbfgs_loop(
                model=self,
                data=data_moments,
                fit_type="mm",
                max_epochs=max_epochs,
                lr=lr,
                show_progress=show_progress,
                return_loss=True,
                **kwargs,
            )
        else:
            raise ValueError("Optimizer must be 'LBFGS' or 'NAdam'.")
        if return_loss:
            return {"loss": loss, "training_time": training_time}


    def max_likelihood(
        self,
        y,
        max_epochs=300,
        lr=0.1,
        optimizer="NAdam",
        show_progress=True,
        return_loss=False,
        n_cycles=3,
        cycle_gamma=0.5,
        **kwargs,
    ):
        """
        Fit the distribution parameters through maximum likelihood.

        Parameters
        ----------------
        y : torch.Tensor, shape (n_samples x n_dim)
            Observed data.

        max_epochs : int, optional
            Number of max training epochs. By default 50.

        lr : float
            Learning rate for the optimizer. Default is 0.1.

        optimizer : str
            Optimizer to use for training. Options are 'LBFGS' and 'NAdam'.
            Default is 'NAdam'.

        show_progress : bool
            If True, show a progress bar during training. Default is True.

        return_loss : bool
            If True, return the loss after training. Default is False.

        n_cycles : int
            For the NAdam optimier, the number of times to run the optimization loop.

        cycle_gamma : float
            For the NAdam optimizer, the factor by which lr is reduced after each run
            of the optimization loop.

        **kwargs
            Additional keyword arguments passed to the NAdam optimizer.

        Returns
        ----------------
        dict
          Dictionary containing the loss and training time.
        """
        if not isinstance(y, torch.Tensor):
            raise ValueError("y must be a torch.Tensor for log-likelihood fitting.")

        if optimizer == "NAdam":
            loss = []
            training_time = []
            # Run the NAdam optimizer for n_cycles
            for c in range(n_cycles):
                lr_cycle = lr * cycle_gamma ** c
                loss_cycle, training_time_cycle = nadam_loop(
                    model=self,
                    data=y,
                    fit_type="ml",
                    max_epochs=max_epochs,
                    lr=lr_cycle,
                    show_progress=show_progress,
                    return_loss=True,
                    **kwargs,
                )
                loss.append(loss_cycle)
                if c == 0:
                    training_time.append(training_time_cycle)
                else:
                    training_time.append(training_time_cycle + training_time[-1])

        elif optimizer == "LBFGS":
            loss, training_time = lbfgs_loop(
                model=self,
                data=y,
                fit_type="ml",
                max_epochs=max_epochs,
                lr=lr,
                show_progress=show_progress,
                return_loss=True,
                **kwargs,
            )
        else:
            raise ValueError("Optimizer must be 'LBFGS' or 'NAdam'.")

        if return_loss:
            return {"loss": loss, "training_time": training_time}


    def moment_init(self, data_moments):
        """
        Initialize the distribution parameters using the observed moments
        as the initial guess.

        Parameters
        ----------------
          data : dict
            Dictionary containing the observed moments. Must contain the keys
              - 'mean': torch.Tensor, shape (n_dim)
              - 'covariance': torch.Tensor, shape (n_dim x n_dim)
              - 'second_moment': torch.Tensor, shape (n_dim x n_dim)
        """
        data_mean_normalized = data_moments["mean"] / torch.norm(data_moments["mean"])
        self.mean_x = data_mean_normalized
        self.covariance_x = data_moments["covariance"]


    def __dir__(self):
        return ["mean_x", "covariance_x", "moments", "log_pdf", "pdf",
                "moments_empirical", "sample", "moment_match", "moment_init"]
