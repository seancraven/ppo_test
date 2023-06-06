"""
Each algorithm defines how the agent learns from it's experience.
using the policy gradient techniques.

Module containing the policy gradient algorithms. 

Each Algorithm inherits a basic set of hyperparameters, and an inteface from
AgentTrainingParameters. This Protocol defines how the each episode is updated
, and contains specific hyperparameters for each algorithm.
"""
# pylint: disable = no-member
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from torch import nn
from tqdm import tqdm

from src.agent import Agent  # pyright: ignore


def train_agent(
    agent: Agent,
    envs: gym.vector.VectorEnv,
    training: AgentTrainingParameters,
    dir_name: str = "",
) -> nn.Module:
    """Trains an agent using update function. Saves returns and entropy.

    Args:
        env_name: The environment for training.
        hyp: The hyperparameters for training

    Returns:
        The trained agent.
    """

    env_wrapper = RecordEpisodeStatistics(
        envs, deque_size=training.num_envs * training.num_episodes
    )

    advantages = []
    entropies = []

    states, _ = env_wrapper.reset(seed=training.seed)

    for _ in tqdm(range(training.num_episodes)):
        states, advantage, final_entropy = training.episode(
            states, agent, training, env_wrapper
        )
        advantages.append(advantage.detach().cpu().numpy())
        entropies.append(final_entropy.detach().cpu().numpy())

    np.stack(advantages)
    np.stack(entropies)
    np.save(
        f"{dir_name}/{training.seed}_returns.npy",
        np.array(env_wrapper.return_queue),
    )
    np.save(f"{dir_name}/{training.seed}_advantages.npy", advantages)
    np.save(f"{dir_name}/{training.seed}_entropies.npy", entropies)

    return agent.cpu()


@dataclass
class AgentTrainingParameters(Protocol):
    """Container for general training hyperparameters, and training behaviour."""

    update_name: str
    num_episodes: int = 1000
    num_envs: int = 10
    num_timesteps: int = 128
    seed: int = 0
    td_lambda_lambda: float = 0.95
    gamma: float = 0.99
    lrs: Tuple[float, float] = (1e-3, 5e-4)

    @staticmethod
    def episode(
        states: np.ndarray,
        agent: Agent,
        hyp: AgentTrainingParameters,
        env_wrapper: RecordEpisodeStatistics,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Defines how experience from an episode updates the agent."""
        ...

    @staticmethod
    def update(
        agent: Agent,
        advantages: torch.Tensor,
        action_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        hyperparameters: AgentTrainingParameters,
    ):
        """Defines the how the parameter are updated for the actor critic."""
        ...


def calculate_gae(
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
        masks: Tensor of masks: (batch_size, timestep), 1 if the episode is not
        done, 0 otherwise.
        gamma: The discount factor for the mdp.
        lambda_: The lambda parameter for TD(lambda), controls the amount of
        bias/variance.
        ent_coef: The entropy coefficient, for exploration encouragement.

    Returns:
        advantages: Tensor of advantages: (batch_size, timestep).
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


@dataclass
class PPOTraining(AgentTrainingParameters):
    """Hyperparametrs for PPO training, with default values"""

    epsilon: float = 0.1
    update_name: str = "ppo"

    @staticmethod
    def episode(
        states: np.ndarray,
        agent: Agent,
        hyp: PPOTraining,
        env_wrapper: RecordEpisodeStatistics,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        rewards = torch.zeros((hyp.num_timesteps, hyp.num_envs)).to(agent.device)
        value_estimates = torch.zeros_like(rewards).to(agent.device)
        action_log_probs = torch.zeros_like(rewards).to(agent.device)
        old_action_log_probs = torch.zeros_like(rewards).to(agent.device)
        mask = torch.zeros_like(rewards).to(agent.device)

        old_actor = deepcopy(agent.actor).to(agent.device)

        entropy = torch.Tensor([0]).to(agent.device)

        for timestep in range(hyp.num_timesteps):
            actions, log_probs, values, entropy = agent.get_action(states)
            old_log_probs = old_actor(states).log_prob(actions)
            states, reward, done, _, _ = env_wrapper.step(
                actions.cpu().numpy().reshape(env_wrapper.env.action_space.shape)
            )

            rewards[timestep] = torch.Tensor(reward).squeeze().to(agent.device)
            value_estimates[timestep] = values.squeeze().to(agent.device)
            action_log_probs[timestep] = log_probs.squeeze().to(agent.device)
            old_action_log_probs[timestep] = old_log_probs.squeeze().to(agent.device)
            mask[timestep] = torch.Tensor(1 - done).squeeze().to(agent.device)

        advantage = calculate_gae(
            rewards,
            value_estimates,
            mask,
            gamma=hyp.gamma,
            lambda_=hyp.td_lambda_lambda,
        )

        PPOTraining.update(
            agent, advantage, action_log_probs, old_action_log_probs, hyp
        )

        return states, advantage, entropy

    @staticmethod
    def update(
        agent: Agent,
        advantages: torch.Tensor,
        action_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        hyperparameters: PPOTraining,
    ):
        """Updates the agent's policy and value estimate using ppo.

        Args:
            advantages: Tensor of advantages: (batch_size, 1).
            action_log_probs: Tensor of log probabilities of the actions: (batch_size).
            old_log_probs: Tensor of log probabilities of the actions on the
            previous episode's policy: (batch_size).
            epsilon: The clipping parameter for the ppo loss.

        """
        agent.critic_opt.zero_grad()
        square_td_errors = advantages.pow(2).mean()
        square_td_errors.backward()
        agent.critic_opt.step()
        agent.actor_opt.zero_grad()

        ratio = torch.exp(action_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(
            ratio, 1 - hyperparameters.epsilon, 1 + hyperparameters.epsilon
        )
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
        agent.actor_opt.step()


@dataclass
class A2CTraining(AgentTrainingParameters):
    """Hyperparametrs for A2C training, with default values"""

    entropy_coef: float = 0.1
    update_name: str = "a2c"

    @staticmethod
    def episode(
        states: np.ndarray,
        agent: Agent,
        hyp: A2CTraining,
        env_wrapper: RecordEpisodeStatistics,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        rewards = torch.zeros((hyp.num_timesteps, hyp.num_envs)).to(agent.device)
        value_estimates = torch.zeros_like(rewards).to(agent.device)
        action_log_probs = torch.zeros_like(rewards).to(agent.device)
        mask = torch.zeros_like(rewards).to(agent.device)
        entropy = torch.Tensor([0]).to(agent.device)

        for timestep in range(hyp.num_timesteps):
            actions, log_probs, values, entropy = agent.get_action(states)

            states, reward, done, _, _ = env_wrapper.step(
                actions.cpu().numpy().reshape(env_wrapper.env.action_space.shape)
            )

            rewards[timestep] = torch.Tensor(reward).squeeze().to(agent.device)
            value_estimates[timestep] = values.squeeze().to(agent.device)
            action_log_probs[timestep] = log_probs.squeeze().to(agent.device)
            mask[timestep] = torch.Tensor(1 - done).squeeze().to(agent.device)

        advantage = calculate_gae(
            rewards,
            value_estimates,
            mask,
            gamma=hyp.gamma,
            lambda_=hyp.td_lambda_lambda,
        )
        A2CTraining.update(agent, advantage, action_log_probs, hyp, entropy.mean())
        return states, advantage, entropy

    @staticmethod
    def update(
        agent: Agent,
        advantages: torch.Tensor,
        action_log_probs: torch.Tensor,
        hyperparameters: A2CTraining,
        entropy: torch.Tensor = torch.Tensor([0]),
    ):
        """Updates the agent's policy using gae actor critic.
        Args:
            advantages: Tensor of advantages: (batch_size, timestep).
            action_log_probs: Tensor of log probabilities of the actions:
            (batch_size, timestep).

        """

        agent.critic_opt.zero_grad()
        square_td_errors = advantages.pow(2).mean()
        square_td_errors.backward()
        agent.critic_opt.step()

        agent.actor_opt.zero_grad()
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean()
            - hyperparameters.entropy_coef * entropy.mean()
        )
        actor_loss.backward()
        agent.actor_opt.step()
