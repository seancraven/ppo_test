"""
Need to make some files that take averages over seeds plots both the mean and std for each algo
This should then be done for each task.
"""
import os

import matplotlib.pyplot as plt
import numpy as np


def mov_avg(returns: np.ndarray, window_size: int = 100) -> np.ndarray:
    returns_mov_avg = np.convolve(returns, np.ones(window_size), "valid") / window_size
    return returns_mov_avg


if __name__ == "__main__":
    # root = os.path.join(os.getcwd(), "experiments")
    # for env in os.listdir(root):
    #     fig, ax = plt.subplots(2, 1)
    #     for update in os.listdir(os.path.join(root, env)):
    #         try:
    #             mov_returns = np.stack(
    #                 [
    #                     mov_avg(np.load(os.path.join(root, env, update, fname)))
    #                     for fname in os.listdir(os.path.join(root, env, update))
    #                     if "returns" in fname
    #                 ]
    #             )
    #         except Exception as e:
    #             print(e)
    #             print("Searching in :", os.path.join(root, env, update))
    #             print("files: ", os.path.join(root, env, update))
    #
    #         mean = np.mean(mov_returns, axis=0)
    #         std = np.std(mov_returns, axis=0)
    #         ax[0].plot(mean, label=update)
    #         ax[0].fill_between(
    #             np.arange(len(mean)),
    #             mean - std,
    #             mean + std,
    #             alpha=0.2,
    #         )
    #         ax[1].plot(mov_returns.T)
    #
    #     fig.legend()
    #     fig.savefig(os.path.join("plots", f"{env}.png"))
    file = os.path.join(
        os.getcwd(), "experiments", "LunarLander-v2", "ppo", "42_returns.npy"
    )
    returns = mov_avg(np.load(file))
    plt.plot(returns)
    plt.show()
