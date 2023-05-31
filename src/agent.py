"""
Agent class for actor critic methods.
"""
# pylint: disable=no-member
# pylint: disable=not-callable
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from torch import nn, optim
from torch.distributions import Categorical
from tqdm import tqdm


class Agent(nn.Module):
    """
    Basic agent, for implementing ppo on vector environments, with
    discrete action spaces.
    """

    def __init__(self, envs: VectorEnv):
        super().__init__()
        self.internal_dim = 64
        self.num_envs = envs.num_envs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # R^observation_space -> R^action_space
        self.actor: nn.Module = nn.Sequential(
            nn.Linear(np.prod(envs.single_observation_space.shape), self.internal_dim),
            nn.Tanh(),
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.Tanh(),
            nn.Linear(self.internal_dim, envs.single_action_space.n),
        ).to(self.device)

        # R^observation_space -> R
        self.critic: nn.Module = nn.Sequential(
            nn.Linear(np.prod(envs.single_observation_space.shape), self.internal_dim),
            nn.Tanh(),
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.Tanh(),
            nn.Linear(self.internal_dim, 1),
        ).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

    def forward(self, states: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns the action logits and state values."""
        states = torch.tensor(states).to(self.device)
        state_values = self.critic(states)
        action_logits = self.actor(states)
        return action_logits, state_values

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
        action_logits, state_values = self.forward(states)
        action_dist = Categorical(logits=action_logits)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        assert actions.shape[0] == self.num_envs
        return actions, log_probs, state_values, entropy

    @staticmethod
    def get_action_probs(action_logits: torch.Tensor, actions: torch.Tensor):
        action_dist = Categorical(logits=action_logits)
        return action_dist.log_prob(actions)

    def calculate_gae(  # pylint: disable=too-many-arguments
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lambda_: float,
    ) -> torch.Tensor:
        """Calculates the generalized advantage estimate. Using recursive TD(lambda).
        Args:
            rewards: Tensor of rewards: (batch_size, timestep)
            action_log_probs: Tensor of log probabilities of the actions: (batch_size).
            values: Tensor of state values: (batch_size, timestep).
            entropy: Tensor of entropy values: (batch_size, timestep).
            masks: Tensor of masks: (batch_size, timestep), 1 if the episode is not done,
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
            )
            advantages[timestep] = (
                delta + gamma * lambda_ * masks[timestep] * advantages[timestep + 1]
            )
        return advantages

    def a2c_update(
        self,
        advantages: torch.Tensor,
        action_log_probs: torch.Tensor,
        entropy: torch.Tensor = torch.Tensor([0]),
        ent_coef: float = 0.01,
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
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy
        )
        actor_loss.backward()
        self.actor_opt.step()

    def ppo_update(
        self,
        advantages: torch.Tensor,
        action_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        epsilon: float,
    ):
        """Updates the agent's policy and value estimate using ppo.

        Args:
            advantages: Tensor of advantages: (batch_size, 1).
            action_log_probs: Tensor of log probabilities of the actions: (batch_size).
            old_log_probs: Tensor of log probabilities of the actions on the
            previous episode's policy: (batch_size).
            epsilon: The clipping parameter for the ppo loss.

        """
        self.critic_opt.zero_grad()
        square_td_errors = advantages.pow(2).mean()
        square_td_errors.backward()
        self.critic_opt.step()
        self.actor_opt.zero_grad()

        # Semi-gradient update for actor network

        ratio = torch.exp(action_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

        # Get min actross each env.
        actor_loss = -(
            torch.stack(
                (advantages.detach() * clipped_ratio, advantages.detach() * ratio),
                dim=-1,
            )
            .min(
                dim=-1,
            )
            .values.mean()
        )
        actor_loss.backward()
        self.actor_opt.step()

    @staticmethod
    def ppo_train(
        env_name: str,
        hyp: PPOHyperParameters,
        **env_kwargs,
    ) -> nn.Module:
        """Trains an agent using PPO. Saves returns and entropy.

        Args:
            env_name: The environment for training.
            hyp: The hyperparameters for training

        Returns:
            The trained agent.
        """

        envs = gym.vector.make(
            env_name,
            num_envs=hyp.num_envs,
            *env_kwargs,
        )
        env_wrapper = RecordEpisodeStatistics(
            envs, deque_size=hyp.num_envs * hyp.num_timesteps
        )

        agent = Agent(envs)

        advantages = []
        entropies = []
        states, _ = env_wrapper.reset(seed=hyp.seed)

        for _ in tqdm(range(hyp.num_episodes)):
            rewards = torch.zeros((hyp.num_timesteps, hyp.num_envs)).to(agent.device)
            value_estimates = torch.zeros_like(rewards).to(agent.device)
            action_log_probs = torch.zeros_like(rewards).to(agent.device)
            ents = torch.zeros_like(rewards).to(agent.device)
            mask = torch.zeros_like(rewards).to(agent.device)
            old_action_log_probs = torch.zeros_like(rewards).to(agent.device)

            old_actor = deepcopy(agent.actor)

            for timestep in range(hyp.num_timesteps):
                actions, log_probs, values, entropy = agent.get_action(states)
                old_log_probs = Categorical(
                    logits=old_actor(torch.Tensor(states).to(agent.device))
                ).log_prob(actions)
                states, reward, done, _, _ = env_wrapper.step(actions.cpu().numpy())

                ents[timestep] = entropy.to(agent.device)
                value_estimates[timestep] = values.squeeze().to(agent.device)
                action_log_probs[timestep] = log_probs.to(agent.device)
                old_action_log_probs[timestep] = old_log_probs.to(agent.device)

                rewards[timestep] = torch.squeeze(torch.tensor(reward)).to(agent.device)
                mask[timestep] = torch.tensor(1 - done).to(agent.device)

            advantage = agent.calculate_gae(
                rewards,
                value_estimates,
                mask,
                gamma=hyp.gamma,
                lambda_=hyp.td_lambda_lambda,
            )

            agent.ppo_update(
                advantage, action_log_probs, old_action_log_probs, hyp.epsilon
            )

            advantages.append(advantage.detach().cpu().numpy())
            entropies.append(ents.detach().cpu().numpy())

            ## Plotting
        np.stack(advantages)
        np.stack(entropies)
        np.save("returns_ppo.npy", np.array(env_wrapper.return_queue))
        np.save("advantages_ppo.npy", advantages)
        np.save("entropies_ppo.npy", entropies)

        return agent.cpu()


@staticmethod
def a2c_train(
    env_name: str,
    hyp: A2CHyperParameters,
    **env_kwargs,
) -> nn.Module:
    """Trains an agent using Actor Critic. Saves returns and entropy.

    Args:
        env_name: The environment for training.
        hyp: The hyperparameters for training

    Returns:
        The trained agent.
    """

    envs = gym.vector.make(
        env_name,
        num_envs=hyp.num_envs,
        *env_kwargs,
    )
    agent = Agent(envs)
    env_wrapper = RecordEpisodeStatistics(
        envs, deque_size=hyp.num_envs * hyp.num_timesteps
    )

    advantages = []
    entropies = []
    states, _ = env_wrapper.reset(seed=hyp.seed)

    for _ in tqdm(range(hyp.num_episodes)):
        rewards = torch.zeros((hyp.num_timesteps, hyp.num_envs)).to(agent.device)
        value_estimates = torch.zeros_like(rewards).to(agent.device)
        action_log_probs = torch.zeros_like(rewards).to(agent.device)
        ents = torch.zeros_like(rewards).to(agent.device)
        mask = torch.zeros_like(rewards).to(agent.device)

        for timestep in range(hyp.num_timesteps):
            actions, log_probs, values, entropy = agent.get_action(states)

            states, reward, done, _, _ = env_wrapper.step(actions.cpu().numpy())

            ents[timestep] = entropy.to(agent.device)
            value_estimates[timestep] = values.squeeze().to(agent.device)
            action_log_probs[timestep] = log_probs.to(agent.device)

            rewards[timestep] = torch.squeeze(torch.tensor(reward)).to(agent.device)
            mask[timestep] = torch.tensor(1 - done).to(agent.device)

        advantage = agent.calculate_gae(
            rewards,
            value_estimates,
            mask,
            gamma=hyp.gamma,
            lambda_=hyp.td_lambda_lambda,
        )

        agent.update(advantage, action_log_probs, ents.mean(), hyp.entropy_coef)

        advantages.append(advantage.detach().cpu().numpy())
        entropies.append(ents.detach().cpu().numpy())

    np.stack(advantages)
    np.stack(entropies)
    np.save("returns.npy", np.array(env_wrapper.return_queue))
    np.save("advantages.npy", advantages)
    np.save("entropies.npy", entropies)

    return agent.cpu()


@dataclass
class HyperParameters:
    """Hyperparametrs for the agent, with default values"""

    num_episodes: int = 1000
    num_envs: int = 16
    num_timesteps: int = 1000
    seed: int = 0
    td_lambda_lambda: float = 0.95
    gamma: float = 0.99


@dataclass
class PPOHyperParameters(HyperParameters):
    """Hyperparametrs for PPO training, with default values"""

    epsilon: float = 0.1


@dataclass
class A2CHyperParameters(HyperParameters):
    """Hyperparametrs for A2C training, with default values"""

    entropy_coef: float = 0.01
