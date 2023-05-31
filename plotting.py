import matplotlib.pyplot as plt
import numpy as np


def plot_returns(ax, returns):
    returns_mov_avg = np.convolve(returns, np.ones(100), "valid") / 100
    ax.plot(returns_mov_avg)


if __name__ == "__main__":
    returns = np.load("returns.npy")
    ppo_returns = np.load("returns_ppo.npy")
    fig, ax = plt.subplots()
    plot_returns(ax, returns)
    plot_returns(ax, ppo_returns)
    fig.savefig("returns.png")
