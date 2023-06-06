"""
Agent class for actor critic methods.

TODO:
    - Change training so it is a function that takes an agent and an environment.
    - Training baseline test to make sure any refactoring doesn't break anything.
"""
# pylint: disable=no-member
# pylint: disable=not-callable
from __future__ import annotations

from typing import Any
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.distributions import Distribution
from torch.distributions import Normal
from torch.nn import functional as F


class OneHotEncoder(nn.Module):
    """
    Layer to do one_hot encoding of discrete vector inputs.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, integers: torch.Tensor) -> torch.Tensor:
        return F.one_hot(integers.long(), num_classes=self.num_classes).float()


class ContinousHead(nn.Module):
    """
    Layer for continuous action spaces.

    No covariance, just a diagonal matrix.

    Includes a linear layer to map the input to the output dimention.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mean = nn.Linear(self.in_dim, self.out_dim)
        self.sigma = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> Distribution:
        mean = self.mean(x)
        sigma = self.sigma(x).exp()
        return Normal(mean, sigma)


class DiscreteHead(nn.Module):
    """
    Layer for discrete action spaces.

    Softmax for multivariate head.

    Includes a linear layer to map the input to the output dimention.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> Distribution:
        logits = self.layer(x)
        return Categorical(logits=logits)


class RLEnvironmentError(Exception):
    """Raised when the environment is not supported."""


class Actor(nn.Module):
    """Stochastic nonlinear policy."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        internal_dim: int = 64,
    ):
        super().__init__()
        match observation_space.shape:
            case tuple(observation_space.shape):
                self.input_dim = int(np.prod(observation_space.shape))
            case None:
                raise RLEnvironmentError("Environment must have shape supported.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.internal_dim = internal_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.internal_dim),
            nn.ReLU(),
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.ReLU(),
            self._out_space_to_head(action_space),
        )

    def forward(self, states: torch.Tensor) -> Distribution:
        states = torch.Tensor(states).reshape(-1, self.input_dim).to(self.device)
        return self.net(states)

    def _out_space_to_head(self, space: gym.Space) -> nn.Module:
        """Returns a input layer, that maps the observation_space
        to the internal_dim.

        If the action space is discrete then the head of the model is
        a softmax.

        If the action space is continuous, then the head of the model is a gaussian.


        Args:
            space: The observation space of the environment.
            internal_dim: The dimension of the internal layer.
        Returns:
            head: The input layer.


            NotImplementedError: If the space is not discrete or continuous.
        """
        match space:
            case gym.spaces.Box():
                return ContinousHead(self.internal_dim, int(np.prod(space.shape)))
            case gym.spaces.Discrete():
                return DiscreteHead(self.internal_dim, int(space.n))
            case _:
                print(space)
                raise NotImplementedError("Only Box and Discrete spaces are supported.")


class Agent(nn.Module):
    """
    Basic agent, for implementing ppo on vector environments, with
    discrete action spaces.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 1e-3,
        critic_lr: float = 5e-3,
    ):
        super().__init__()

        match observation_space.shape:
            case tuple(observation_space.shape):
                self.input_dim = int(np.prod(observation_space.shape))

            case None:
                raise RLEnvironmentError("Environment must have shape supported.")

        self.internal_dim = 64
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.actor = Actor(observation_space, action_space, self.internal_dim).to(
            self.device
        )

        self.critic: nn.Module = nn.Sequential(
            nn.Linear(self.input_dim, self.internal_dim),
            nn.Tanh(),
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.Tanh(),
            nn.Linear(self.internal_dim, 1),
        ).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def forward(self, states: Any) -> Tuple[Distribution, torch.Tensor]:
        """Forward pass returns the action logits and state values."""

        states = torch.tensor(states).reshape(-1, self.input_dim).to(self.device)
        state_values = self.critic(states)
        action_dist = self.actor(states)
        return action_dist, state_values

    def get_action(
        self, states: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects an action from the agent's policy.

        Args:
            states_actions: The states and actions of the agents.

        Returns:
            actions: Tensor of actions: (batch_size, action_dim)
            log_probs: Tensor of log probabilities of the actions: (batch_size, 1)
            action_logits: Tensor of action logits: (batch_size, action_dim)
            state_values: Tensor of state values: (batch_size, 1)
            entropy: Tensor of entropy values: (batch_size, 1)
        """
        action_dist, state_values = self.forward(states)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return actions, log_probs, state_values, entropy
