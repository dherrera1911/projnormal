"""Routine to fit projected normal parameters using Gradient Descent."""

import time

import torch
from torch import optim
from tqdm import tqdm

__all__ = ["lbfgs_loop", "nadam_loop"]


def __dir__():
    return __all__


STOP_EPOCHS = 3 # Consecutive epochs to stop training if loss change is below atol


def mse_loss(momentsA, momentsB):
    """Compute the Euclidean distance between the observed and model moments."""
    distance_means_sq = torch.sum(
        (momentsA["mean"]*20 - momentsB["mean"]*20)**2
    )
    distance_sm_sq = torch.sum(
        (momentsA["covariance"]*200 - momentsB["covariance"]*200)**2
    )
    return distance_means_sq + distance_sm_sq


def norm_loss(momentsA, momentsB):
    """Compute the Euclidean distance between the observed and model moments."""
    distance_means_sq = torch.sqrt(
      torch.sum((momentsA["mean"] - momentsB["mean"])**2) + 1e-8
    )
    distance_sm_sq = torch.sqrt(
      torch.sum((momentsA["covariance"] - momentsB["covariance"])**2) + 1e-8
    )
    return distance_means_sq + distance_sm_sq


def _mm_data_check(data):
    """Check that data is of type expected for moment matching."""
    # Check data is a dictionary
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary for moment_matching.")
    # Check if the data is complete
    if not all(key in data for key in ["mean", "covariance"]):
        raise ValueError(
          "Data must contain the keys 'mean' and 'covariance'."
        )


def _ll_data_check(data, n_dim):
    """Check that data is of type expected for log-likelihood fitting."""
    # Check that data is a pytorch tensor
    if not isinstance(data, torch.Tensor):
        raise ValueError("Data must be a torch.Tensor for log-likelihood fitting.")
    # Check that data is of the right shape
    if data.ndim != 2:
        raise ValueError("Data must have shape (n_samples, n_dim) for log-likelihood fitting.")
    if data.shape[1] != n_dim:
        raise ValueError(
            f"Data must have shape (n_samples, {n_dim})."
        )


def lbfgs_loop(
    model,
    data,
    fit_type="mm",
    max_epochs=200,
    lr=0.1,
    atol=1e-10,
    loss_fun=None,
    show_progress=True,
    return_loss=False,
    **kwargs,
):
    """
    Fit the model parameters to the observed data moments using gradient descent.

    Parameters
    ----------
    model : Object of class ProjNormal or subclass
        The model used for fitting.

    data :
        If fit_type is 'mm', data should be a dictionary containing the data statistics
        in the following format:
              - 'mean': torch.Tensor of shape (n_dim).
              - 'covariance': torch.Tensor of shape (n_dim, n_dim).
              - 'second_moment': torch.Tensor of shape (n_dim, n_dim).
        If fit_type is 'ml', data should be a torch.Tensor of shape (n_samples, n_dim).

    fit_type : str, optional
        Type of fitting to perform. Can be either 'mm' standing
        for moment-matching, or 'ml' standing for maximum-likelihood.
        By default 'mm'.

    max_epochs : int, optional
        Number of max training epochs. By default 200.

    lr : float, optional
        Learning rate, by default 0.1.

    atol : float, optional
        Tolerance for stopping training, by default 1e-10.

    loss_fun : callable, optional
        Loss function to use for moment matchin. By default None,
        which uses the squared Euclidean distance between the observed and model moments.

    show_progress : bool
        If True, show a progress bar during training. Default is True.

    return_loss : bool
        If True, return the loss after training. Default is False.

    kwargs : dict
        Additional arguments to pass to LBFGS optimizer.

    Returns
    -------
    dict
        Dictionary containing the loss and training time at each epoch.
    """
    if loss_fun is None:
        loss_fun = mse_loss

    # Define the closure function depending on the type of fit
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        **kwargs,
    )
    if fit_type == "mm":
        _mm_data_check(data)
        def closure():
            optimizer.zero_grad()
            model_moments = model.moments()
            loss = loss_fun(model_moments, data)
            loss.backward()
            return loss
    elif fit_type == "ml":
        _ll_data_check(data, model.n_dim)
        def closure():
            optimizer.zero_grad()
            log_likelihood = model.log_pdf(data)
            loss = -torch.sum(log_likelihood)
            loss.backward()
            return loss

    # Initialize variables
    loss_list = []
    training_time = []
    total_start_time = time.time()

    prev_loss = 0.0
    consecutive_stopping_criteria_met = 0

    for e in tqdm(
        range(max_epochs), desc="Epochs", unit="epoch", disable=not show_progress
    ):
        epoch_loss = optimizer.step(closure)
        epoch_time = time.time() - total_start_time
        loss_change = abs(prev_loss - epoch_loss.item())

        # Check if loss change is below atol
        if loss_change < atol:
            consecutive_stopping_criteria_met += 1
        else:
            consecutive_stopping_criteria_met = 0

        prev_loss = epoch_loss.item()
        training_time.append(epoch_time)
        loss_list.append(epoch_loss.item())

        # Stop if loss change is below atol for 3 consecutive epochs
        if consecutive_stopping_criteria_met >= STOP_EPOCHS:
            tqdm.write(
                f"Loss change below {atol} for {STOP_EPOCHS} consecutive epochs. "
                + f"Stopping training at epoch {e + 1}/{max_epochs}."
            )
            break
    if consecutive_stopping_criteria_met == 0:
        print(
            f"Reached max_epochs ({max_epochs}) without meeting stopping criteria."
            + "Consider increasing max_epochs, changing initialization or "
            + "using dtype=torch.float64."
        )

    if return_loss:
        return torch.tensor(loss_list), torch.tensor(training_time)
    else:
        return None


def nadam_loop(
    model,
    data,
    fit_type="mm",
    max_epochs=200,
    lr=0.1,
    step_size=10,
    gamma=0.85,
    loss_fun=None,
    show_progress=True,
    return_loss=False,
):
    """
    Fit the model parameters to the observed data moments using the NAdam optimizer
    and a StepLR scheduler.

    Parameters
    ----------
    model : ProjNormal (or subclass)
        The model used for fitting.

    data :
        If fit_type is 'mm', data should be a dictionary containing the data statistics
        in the following format:
              - 'mean': torch.Tensor of shape (n_dim).
              - 'covariance': torch.Tensor of shape (n_dim, n_dim).
              - 'second_moment': torch.Tensor of shape (n_dim, n_dim).
        If fit_type is 'ml', data should be a torch.Tensor of shape (n_samples, n_dim).

    fit_type : str, optional
        Type of fitting to perform. Can be either 'mm' standing
        for moment-matching, or 'ml' standing for maximum-likelihood.
        By default 'mm'.

    max_epochs : int, optional
        Maximum number of training epochs (default = 200).

    lr : float, optional
        Initial learning rate (default = 0.1).

    loss_fun : callable, optional
        Loss function to use for moment matchin. By default None, which
        uses the Euclidean distance between the observed and model moments.

    step_size : int, optional
        Period of learning rate decay in epochs (default = 10).

    gamma : float, optional
        Multiplicative factor of learning rate decay (default = 0.1).

    show_progress : bool
        If True, display a progress bar (default = True).

    return_loss : bool
        If True, return the loss values at each epoch (default = False).

    Returns
    -------
    None or (torch.Tensor, torch.Tensor)
        If `return_loss` is True, returns (loss_list, training_time).
        Otherwise returns None.
    """
    if loss_fun is None:
        loss_fun = mse_loss

    if fit_type == "mm":
        _mm_data_check(data)
    elif fit_type == "ml":
        _ll_data_check(data, model.n_dim)

    # Initialize NAdam optimizer
    optimizer = optim.NAdam(model.parameters(), lr=lr)
    # StepLR scheduler: decay LR by gamma every `step_size` epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_list = []
    training_time = []
    total_start_time = time.time()

    for _e in tqdm(
        range(max_epochs), desc="Epochs", unit="epoch", disable=not show_progress
    ):
        # Forward pass
        if fit_type == "mm":
            model_moments = model.moments()
            loss = loss_fun(model_moments, data)
        elif fit_type == "ml":
            log_likelihood = model.log_pdf(data)
            loss = -torch.sum(log_likelihood)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Scheduler step at each epoch
        scheduler.step()

        # Record loss and time
        epoch_time = time.time() - total_start_time
        training_time.append(epoch_time)
        loss_list.append(loss.item())

    if return_loss:
        return torch.tensor(loss_list), torch.tensor(training_time)
    else:
        return None
