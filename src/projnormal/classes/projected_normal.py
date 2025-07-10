"""Class for the general projected normal distribution."""
import geotorch
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

import projnormal.formulas.projected_normal as pn_formulas
import projnormal.formulas.projected_normal_c as pnc_formulas

from ._optim import lbfgs_loop, nadam_loop
from .constraints import Sphere

__all__ = [
  "ProjNormal",
]


def __dir__():
    return __all__


#### Class for the projected normal distribution with learnable parameters
class ProjNormal(nn.Module):
    r"""
    Projected normal distribution, describing the variable
    :math:`y=x/\|x\|`, where :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)`
    follows a multivariate normal distribution.

    Parameters
    ----------
      n_dim : ``int``
          Dimension of :math:`x` (the embedding space). Optional: If ``mean_x``
          and ``covariance_x`` are provided, it is not required.

      mean_x : ``torch.Tensor``, optional
          Mean of :math:`x`. Shape ``(n_dim)``. Default is random.

      covariance_x : ``torch.Tensor``, optional
          Covariance of :math:`x`. Shape ``(n_dim, n_dim)``. Default is the identity.

    Attributes
    ----------
      mean_x : ``torch.Tensor``
          Mean of :math:`x`. Learnable parameter constrained to have unit norm. Shape ``(n_dim)``.

      covariance_x : ``torch.Tensor``
          Covariance of :math:`x`. Learnable parameter constrained to be SPD. Shape ``(n_dim, n_dim)``. 
    """

    def __init__(
        self,
        n_dim=None,
        mean_x=None,
        covariance_x=None,
    ):
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
        self.covariance_x = nn.Parameter(covariance_x.clone())

        # Add parameter constraints
        parametrize.register_parametrization(self, "mean_x", Sphere())
        geotorch.positive_definite(self, "covariance_x")
        self.covariance_x = covariance_x.clone()

        # Add const as buffer set to 0 to make child models easier
        self.register_buffer("const", torch.tensor(0), persistent=False)


    def moments(self):
        """
        Compute moments of the distribution via Taylor approximation.

        Returns
        -------
          dict
              Dictionary with keys ``mean``, ``covariance`` and ``second_moment``,
              containing the corresponding moments of the distribution.
        """
        # Use dist.const to not redefine method for the Const class
        gamma = pnc_formulas.mean(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            const=self.const,
        )
        second_moment = pnc_formulas.second_moment(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            const=self.const,
        )
        cov = second_moment - torch.einsum("i,j->ij", gamma, gamma)
        return {"mean": gamma, "covariance": cov, "second_moment": second_moment}


    def moments_empirical(self, n_samples=200000):
        """
        Compute moments of the distribution via sampling.

        Parameters
        ----------
          n_samples : ``int``
              Number of samples to draw for empirical moments. Default is ``200000``.

        Returns
        -------
          dict
              Dictionary with keys ``mean``, ``covariance`` and ``second_moment``,
              containing the corresponding moments of the distribution.
        """
        with torch.no_grad():
            stats_dict = pnc_formulas.empirical_moments(
                mean_x=self.mean_x,
                covariance_x=self.covariance_x,
                n_samples=n_samples,
                const=self.const,
            )
        return stats_dict


    def log_pdf(self, y):
        """
        Compute the log pdf of points `y`.

        Parameters
        ----------
          y : ``torch.Tensor``
              Points to evaluate the log pdf. Shape ``(n_points, n_dim)``.

        Returns
        -------
          ``torch.Tensor``
              Log-PDF of `y`. Shape ``(n_points)``.
        """
        lpdf = pn_formulas.log_pdf(
            mean_x=self.mean_x,
            covariance_x=self.covariance_x,
            y=y,
        )
        return lpdf


    def pdf(self, y):
        """
        Compute the pdf of points `y`.

        Parameters
        ----------
          y : ``torch.Tensor``
              Points to evaluate the pdf. Shape ``(n_points, n_dim)``.

        Returns
        -------
          ``torch.Tensor``
              PDF of the points `y`. Shape ``(n_points)``.
        """
        pdf = torch.exp(self.log_pdf(y))
        return pdf


    def sample(self, n_samples):
        """
        Sample from the distribution.

        Parameters
        ----------
          n_samples : ``int``
              Number of samples to draw.

        Returns
        -------
          ``torch.Tensor``
              Samples from the distribution. Shape ``(n_samples, n_dim)``.
        """
        with torch.no_grad():
            samples = pnc_formulas.sampling.sample(
                mean_x=self.mean_x,
                covariance_x=self.covariance_x,
                n_samples=n_samples,
                const=self.const,
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
        Fit the distribution parameters via moment matching.

        Parameters
        ----------
          data_moments : ``dict``
            Dictionary containing the observed moments, with keys
            ``mean`` and ``covariance``.

          max_epochs : ``int``
              Number of max training epochs. By default ``50``.

          lr : ``float``
              Learning rate for the optimizer. Default is ``0.1``.

          optimizer : ``str``
              Optimizer to use. Options are 'LBFGS' and 'NAdam'.
              Default is 'NAdam'.

          loss_fun : callable
              Loss function to use for moment matching. Default is squared error.

          show_progress : ``bool``
              If ``True``, show a progress bar during training. Default is ``True``.

          return_loss : ``bool``
              If ``True``, return the loss after training. Default is ``False``.

          n_cycles : ``int``
              For the NAdam optimier, the number of times to run the optimization loop.

          cycle_gamma : ``float``
              Factor by which ``lr`` is reduced after each optimization loop repetition.

          **kwargs
              Additional keyword arguments passed to the fitting function.
              For the NAdam optimizer, the parameters ``gamma`` and ``step_size`` can be passed
              to control the learning rate schedule.

        Returns
        -------
          ``dict``
              Dictionary containing the loss and training time.
        """
        # Check data_moments is a dictionary
        if not isinstance(data_moments, dict):
            raise ValueError("Data must be a dictionary.")

        # Check if the data is complete
        if not all(key in data_moments for key in ["mean", "covariance"]):
            raise ValueError(
              "Data must contain the keys 'mean' and 'covariance'."
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
            loss = torch.cat(loss)
            training_time = torch.cat(training_time)

        elif optimizer == "LBFGS":
            for c in range(n_cycles):
                if c > 0:
                    noise = torch.randn(self.n_dim, device=self.mean_x.device)
                    noise = noise * torch.norm(self.mean_x.detach()) / 10
                    self.mean_x = self.mean_x + noise
                    try:
                        self.covariance_x = self.covariance_x.detach() * 1.5
                    except Exception:
                        pass

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
        n_cycles=1,
        cycle_gamma=0.5,
        **kwargs,
    ):
        """
        Fit the distribution parameters via maximum likelihood,
        by iteratively minimizing the negative log-likelihood of the data `y`.

        Parameters
        ----------
          y : ``torch.Tensor``
              Observed data. Shape ``(n_samples, n_dim)``.

          max_epochs : ``int``
              Number of max training epochs. By default ``50``.

          lr : ``float``
              Learning rate for the optimizer. Default is ``0.1``.

          optimizer : ``str``
              Optimizer to use. Options are 'LBFGS' and 'NAdam'.
              Default is 'NAdam'.

          loss_fun : callable
              Loss function to use for moment matching. Default is squared error.

          show_progress : ``bool``
              If ``True``, show a progress bar during training. Default is ``True``.

          return_loss : ``bool``
              If ``True``, return the loss after training. Default is ``False``.

          n_cycles : ``int``
              For the NAdam optimier, the number of times to run the optimization loop.

          cycle_gamma : ``float``
              Factor by which ``lr`` is reduced after each optimization loop repetition.

          **kwargs
              Additional keyword arguments passed to the fitting function.
              For the NAdam optimizer, the parameters ``gamma`` and ``step_size`` can be passed
              to control the learning rate schedule.

        Returns
        -------
          ``dict``
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
            loss = torch.cat(loss)
            training_time = torch.cat(training_time)

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
        as the initial guess (after making the mean unit norm).

        Parameters
        ----------
          data_moments : ``dict``
            Dictionary with keys ``mean`` and ``covariance``, containing the observed
            mean and covariance of the data respectively.
        """
        data_mean_normalized = data_moments["mean"] / torch.norm(data_moments["mean"])
        self.mean_x = data_mean_normalized
        self.covariance_x = data_moments["covariance"] \
            + 1e-5 * torch.eye(self.n_dim,
                               device=self.mean_x.device,
                               dtype=self.mean_x.dtype
                              )


    def __dir__(self):
        """List of methods available in the ProjNormal class."""
        return ["mean_x", "covariance_x", "moments", "log_pdf", "pdf", "max_likelihood",
                "moments_empirical", "sample", "moment_match", "moment_init"]
