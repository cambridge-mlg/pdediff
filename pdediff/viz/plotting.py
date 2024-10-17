"""Helper function for plotting"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from pdediff.viz.kolmogorov import draw
from pdediff.mcs import KolmogorovFlow

def plot_1d_trajectories(trajectories):
    b, t, x = trajectories.squeeze(2).shape

    fig, axes = plt.subplots(b, 1, figsize=(5, 2 * trajectories.shape[0]))

    if b > 1:
        for i in range(b):
            axes[i].imshow(trajectories.squeeze(2)[i, :].T)
    else:
        axes.imshow(trajectories.squeeze(2)[0, :].T)

    return fig


def plot_mean_and_std(metrics, labels, title: str = "", xlabel: str = "", ylabel: str = "", figsize: tuple = (8, 8)):
    """
    Method to plot the mean and std of a metric.
    It assumes that metric has shape  [B, T] #todo: check this is valid always

    Args:
        metric: metric computed on different samples
        label: name of the method for which we are computing the metric (used in the legend)
        title: plot title
        xlabel: label for the x-axis
        ylabel: label for the y-axis
        figsize: size of the plot
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    for i, metric in enumerate(metrics):
        assert (
            len(metric.shape) == 2
        ), "We are assuming that metric has shape batch x traj_length, if not either change or adapt the plotting method"

        # now we plot the average
        if torch.is_tensor(metric):
            metric_average = torch.mean(metric, 0)
            metric_std = torch.std(metric, 0)
        else:
            metric_average = np.mean(metric, 0)
            metric_std = np.std(metric, 0)

        axes.plot(np.arange(len(metric[0, :])), metric_average, c=f"C{i}", alpha=1, label=labels[i], zorder=10)
        axes.errorbar(np.arange(len(metric[0, :])), metric_average, yerr=metric_std, c=f"C{i}", alpha=0.3)

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend()

    return fig


def plot_comparison_metric(
    metric: list[ArrayLike],
    labels: list[str],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (8, 8),
):
    """
    Method to plot the same metric of different methods.

    Args:
        metric: list containing different metric from different methods.
        labels: list containing different label associated with the metric inside the correlation list
        title: plot title
        xlabel: label for the x-axis
        ylabel: label for the y-axis
        figsize: size of the plot
    """

    fig, axes = plt.subplots(1, 1, figsize=figsize)

    for i in range(len(metric)):
        axes.plot(np.arange(len(metric[i])), metric[i], c=f"C{i}", label=labels[i], alpha=0.5)

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend()

    return fig


def plot_metric_different_samples_and_average(
    metric: ArrayLike, label: str, title: str = "", xlabel: str = "", ylabel: str = "", figsize: tuple = (8, 8)
):
    """
    Method to plot the pearson correlation of a single method.
    It assumes that metric has sahpe  [B, T] #todo: check this is valid always

    Args:
        metric: metric computed on different samples
        label: name of the method for which we are computing the metric (used in the legend)
        title: plot title
        xlabel: label for the x-axis
        ylabel: label for the y-axis
        figsize: size of the plot
    """

    assert (
        len(metric.shape) == 2
    ), "We are assuming that metric has shape batch x traj_length, if not either change or adapt the plotting method"

    fig, axes = plt.subplots(1, 1, figsize=figsize)

    for i in range(metric.shape[0]):
        axes.plot(np.arange(len(metric[0, :])), metric[i, :], c="lightgrey", alpha=0.3)

    # now we plot the average
    if torch.is_tensor(metric):
        metric_average = torch.mean(metric, 0)
    else:
        metric_average = np.mean(metric, 0)

    axes.plot(np.arange(len(metric[0, :])), metric_average, c="firebrick", alpha=0.8, label=label, zorder=10)

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend()

    return fig


def plot_mean_and_std(metrics, labels, title: str = "", xlabel: str = "", ylabel: str = "", figsize: tuple = (8, 8)):
    """
    Method to plot the mean and std of a metric.
    It assumes that metric has shape  [B, T] #todo: check this is valid always

    Args:
        metric: metric computed on different samples
        label: name of the method for which we are computing the metric (used in the legend)
        title: plot title
        xlabel: label for the x-axis
        ylabel: label for the y-axis
        figsize: size of the plot
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    for i, metric in enumerate(metrics):
        assert (
            len(metric.shape) == 2
        ), "We are assuming that metric has shape batch x traj_length, if not either change or adapt the plotting method"

        # now we plot the average
        if torch.is_tensor(metric):
            metric_average = torch.mean(metric, 0)
            metric_std = torch.std(metric, 0)
        else:
            metric_average = np.mean(metric, 0)
            metric_std = np.std(metric, 0)

        axes.plot(np.arange(len(metric[0, :])), metric_average, c=f"C{i}", alpha=1, label=labels[i], zorder=10)
        axes.errorbar(np.arange(len(metric[0, :])), metric_average, yerr=metric_std, c=f"C{i}", alpha=0.3)

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend()

    return fig

def plot_mean_and_std_err(
    metrics,
    labels,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (8, 8),
    y_log_scale: bool = False,
):
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    for i, metric in enumerate(metrics):
        assert (
            len(metric.shape) == 2
        ), "We are assuming that metric has shape batch x traj_length, if not either change or adapt the plotting method"

        # now we plot the average
        if torch.is_tensor(metric):
            metric_average = torch.mean(metric, 0)
            metric_std_err = torch.std(metric, 0) / np.sqrt(metric.shape[0])
        else:
            metric_average = np.mean(metric, 0)
            metric_std_err = np.std(metric, 0) / np.sqrt(metric.shape[0])

        axes.plot(np.arange(len(metric[0, :])), metric_average, c=f"C{i}", alpha=0.6, label=labels[i], zorder=10)

        print(metric_average)
        print("---")
        axes.fill_between(
            np.arange(len(metric[0, :])),
            metric_average - metric_std_err,
            metric_average + metric_std_err,
            alpha=0.6,
            color=f"C{i}",
        )

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if y_log_scale:
        axes.set_yscale("log")
    axes.legend()

    return fig


def plot_kolmogorov_vorticity_trajectories(trajectories):
    # simple method to plot Kolmogorov trajectories
    # these are 64x64 states
    chain = KolmogorovFlow(size=256, dt=0.2)

    # First compute the vorticity of the trajectories
    vorticity = chain.vorticity(trajectories) # this results in batch, traj_length, state1, state2
    img = draw(vorticity, zoom=4)
    return img

