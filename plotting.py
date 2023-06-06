"""
Plotting functions for the experiments.
"""
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

PlotFunction = Callable[[Axes, str, str], None]


def moving_average(vector: np.ndarray, window_size: int = 100) -> np.ndarray:
    """Computes the moving average of a vector.

    Args:
        vector (np.ndarray): Vector to compute the moving average of.
        window_size (int, optional): Size of the window. Defaults to 100.
    Returns:
        np.ndarray: Moving average of the vector.
    """
    return np.convolve(vector, np.ones(window_size), "valid") / window_size


def plot_statistic(
    root_dir: str,
    plot_dir: str,
    plotting_function: PlotFunction,
):
    """Plotting wrapper to walk file tree and manage axes.

    Args:
        root_dir (str): Root directory of the experiments.
        plot_dir (str): Directory to save the plots in.
        plotting_function (Callable): Function to plot the statistic.
    """
    for environment in os.listdir(os.path.join(root_dir)):
        env_dir = os.path.join(root_dir, environment)
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        for update in os.listdir(env_dir):
            update_dir = os.path.join(env_dir, update)
            plotting_function(axes, update_dir, update)  # pyright: ignore
        fig.savefig(  # pyright: ignore
            os.path.join(plot_dir, f"{environment} {plotting_function.__name__}.png")
        )


def plot_returns(axis: Axes, update_dir: str, update: str):
    """PlotFunction, for agent returns.

    Args:
        axis (Axes): Axis to plot on.
        update_dir (str): Directory of the update.
        update (str): Name of the update.
    """
    returns = [
        np.load(os.path.join(update_dir, f_name))
        for f_name in os.listdir(update_dir)
        if "returns" in f_name
    ]
    returns = np.stack(
        [
            moving_average(returns_[: min(len(r) for r in returns)], 20)
            for returns_ in returns
        ],
        axis=0,
    )
    mean_returns = np.mean(returns, axis=0)
    std_returns = np.std(returns, axis=0)
    axis.plot(mean_returns, label=f"{update}")
    axis.fill_between(
        np.arange(len(mean_returns)),
        mean_returns - std_returns,
        mean_returns + std_returns,
        alpha=0.2,
    )
    axis.set_xlabel("Episodes")
    axis.set_ylabel("Episodic Return")
    axis.legend()


def plot_entropy(axis: Axes, update_dir: str, update: str):
    """
    PlotFunction, for agent entropy.

    Args:
        axis (Axes): Axis to plot on.
        update_dir (str): Directory of the update.
        update (str): Name of the update.
    """
    entropies = [
        np.load(os.path.join(update_dir, f_name))
        for f_name in os.listdir(update_dir)
        if "entropies" in f_name
    ]
    entropies = np.stack(
        [
            moving_average(
                (entropies_[:, -1, ...]).flatten(),
                20,
            )
            for entropies_ in entropies
        ],
        axis=0,
    )
    mean_entropies = np.mean(entropies, axis=0)
    std_entropies = np.std(entropies, axis=0)
    axis.plot(mean_entropies, label=f"{update}")
    axis.fill_between(
        np.arange(len(mean_entropies)),
        mean_entropies - std_entropies,
        mean_entropies + std_entropies,
        alpha=0.2,
    )
    axis.set_xlabel("Episodes")
    axis.set_ylabel("Entropy")
    axis.legend()


def plot_advantage(axis: Axes, update_dir: str, update: str):
    """
    PlotFunction, for agent advantage.

    Args:
        axis (Axes): Axis to plot on.
        update_dir (str): Directory of the update.
        update (str): Name of the update.
    """
    advantages = [
        np.load(os.path.join(update_dir, f_name))
        for f_name in os.listdir(update_dir)
        if "advantage" in f_name
    ]
    advantages = np.stack(
        [
            moving_average((advantages_[:, 1, ...] ** 2).flatten(), 100)
            for advantages_ in advantages
        ],
        axis=0,
    )
    mean_advantages = np.mean(advantages, axis=0)
    std_advantages = np.std(advantages, axis=0)
    axis.plot(mean_advantages, label=f"{update}")
    axis.fill_between(
        np.arange(len(mean_advantages)),
        mean_advantages - std_advantages,
        mean_advantages + std_advantages,
        alpha=0.2,
    )

    axis.set_xlabel("Episodes")
    axis.set_ylabel("Advantage")
    axis.legend()


if __name__ == "__main__":
    root_path = os.path.join(os.curdir, "experiments")
    plot_path = os.path.join(os.curdir, "plots")

    plot_statistic(
        root_path,
        plot_path,
        plot_returns,
    )
    plot_statistic(
        root_path,
        plot_path,
        plot_entropy,
    )
    plot_statistic(
        root_path,
        plot_path,
        plot_advantage,
    )
