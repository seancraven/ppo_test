"""
Agent class for actor critic methods.
"""
# pylint: disable=no-member
# pylint: disable=not-callable
from __future__ import annotations

from typing import Any
from typing import Tuple

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch import nn
from torch import optim


class Agent(nn.Module):
    """
    Basic agent, for implementing ppo on vector env.
    """

    def __init__(self, envs: VectorEnv):
        self.internal_dim = 64
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor: nn.Module = nn.Sequential(
            nn.Linear(np.array(envs.observation_space.shape).prod(), self.internal_dim),
            nn.Tanh(),
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.Tanh(),
            nn.Linear(self.internal_dim, np.array(envs.action_space.shape).prod()),
        ).to(self.device)

        self.critic: nn.Module = nn.Sequential(
            nn.Linear(np.array(envs.observation_space.shape).prod(), self.internal_dim),
            nn.Tanh(),
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.Tanh(),
            nn.Linear(self.internal_dim, 1),
        ).to(self.device)
        self.num_envs = envs.num_envs

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

    def forward(self, states_actions: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns the action logits and state values."""
        states_actions = torch.tensor(states_actions).to(self.device)
        state_values = self.critic(states_actions)
        action_logits = self.actor(states_actions)
        return action_logits, state_values

    def get_action(
        self, states_actions: np.ndarray
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
        action_logits, state_values = self.forward(states_actions)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return actions, log_probs, state_values, entropy

    def calculate_gae(  # pylint: disable=too-many-arguments
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lambda_: float,
        ent_coef: float,
    ) -> torch.Tensor:
        """Calculates the generalized advantage estimate. Using recursive TD(lambda).
        Args:
            rewards: Tensor of rewards: (batch_size, 1)
            action_log_probs: Tensor of log probabilities of the actions: (batch_size).
            values: Tensor of state values: (batch_size, 1).
            entropy: Tensor of entropy values: (batch_size, 1).
            masks: Tensor of masks: (batch_size, 1), 1 if the episode is not done,
            0 otherwise.
            gamma: The discount factor for the mdp.
            lambda_: The lambda parameter for TD(lambda), controls the amount of
            bias/variance.
            ent_coef: The entropy coefficient, for exploration encouragement.

        Returns:
            advantages: Tensor of advantages: (batch_size, 1).
        """
        max_timestep = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        for timestep in reversed(range(max_timestep - 1)):
            delta = (
                rewards[timestep]
                + gamma * values[timestep + 1] * masks[timestep]
                - values[timestep]
                + ent_coef * entropy[timestep]
            )
            advantages[timestep] = (
                delta + gamma * lambda_ * masks[timestep] * advantages[timestep + 1]
            )
        return advantages

    def update(
        self,
        advantages: torch.Tensor,
        action_log_probs: torch.Tensor,
    ):
        """Updates the agent's policy using gae actor critic.
        Args:
            advantages: Tensor of advantages: (batch_size, 1).
            action_log_probs: Tensor of log probabilities of the actions: (batch_size).

        """

        self.critic_opt.zero_grad()
        square_td_errors = advantages.pow(2).mean()
        square_td_errors.backward()
        self.critic_opt.step()

        # reset gradients.
        self.actor_opt.zero_grad()
        # Semi-gradient update for actor network
        actor_loss = -(advantages.detach() * action_log_probs).mean()
        actor_loss.backward()
        self.actor_opt.step()
