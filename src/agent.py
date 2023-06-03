"""
Agent class for actor critic methods.


True abstraction over heads of the neural networks are distrubutions, as 
we define some stochastic policy. 
"""
# pylint: disable=no-member
# pylint: disable=not-callable
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.distributions import Distribution
from torch.distributions import Normal
from torch.nn import functional as F
from tqdm import tqdm


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

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
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

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=5e-3)

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

    def a2c_update(
        self,
        advantages: torch.Tensor,
        action_log_probs: torch.Tensor,
        entropy: torch.Tensor = torch.Tensor([0]),
        ent_coef: float = 0.01,
    ):
        """Updates the agent's policy using gae actor critic.
        Args:
            advantages: Tensor of advantages: (batch_size, timestep).
            action_log_probs: Tensor of log probabilities of the actions:
            (batch_size, timestep).

        """

        self.critic_opt.zero_grad()
        square_td_errors = advantages.pow(2).mean()
        square_td_errors.backward()
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
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

        ratio = torch.exp(action_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
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
    def train_agent(
        env_name: str,
        training: AgentTrainingParameters,
        dir_name: str = "",
        **env_kwargs,
    ) -> nn.Module:
        """Trains an agent using update function. Saves returns and entropy.

        Args:
            env_name: The environment for training.
            hyp: The hyperparameters for training

        Returns:
            The trained agent.
        """

        envs = gym.vector.make(
            env_name,
            num_envs=training.num_envs,
            asynchronous=False,
            **env_kwargs,
        )
        env_wrapper = RecordEpisodeStatistics(
            envs, deque_size=training.num_envs * training.num_episodes
        )

        agent = Agent(
            envs.single_observation_space,
            envs.single_action_space,
        )

        advantages = []
        entropies = []

        states, _ = env_wrapper.reset(seed=training.seed)

        for _ in tqdm(range(training.num_episodes)):
            states, advantage, ents = training.episode(
                states, agent, training, env_wrapper
            )
            advantages.append(advantage.detach().cpu().numpy())
            entropies.append(ents.detach().cpu().numpy())

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
    """Hyperparametrs for the agent, with default values"""

    update_name: str
    num_episodes: int = 1000
    num_envs: int = 10
    num_timesteps: int = 128
    seed: int = 0
    td_lambda_lambda: float = 0.95
    gamma: float = 0.99
    actor_lr: float = 0.001
    critic_lr: float = 0.005

    @staticmethod
    def episode(
        intal_state: np.ndarray,
        agent: Agent,
        hyp: AgentTrainingParameters,
        env_wrapper: RecordEpisodeStatistics,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        ...


@dataclass
class PPOTraining(AgentTrainingParameters):
    """Hyperparametrs for PPO training, with default values"""

    epsilon: float = 0.1
    update_name: str = "ppo"

    @staticmethod
    def episode(
        intal_state: np.ndarray,
        agent: Agent,
        hyp: PPOTraining,
        env_wrapper: RecordEpisodeStatistics,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        states = intal_state
        rewards = torch.zeros((hyp.num_timesteps, hyp.num_envs)).to(agent.device)
        value_estimates = torch.zeros_like(rewards).to(agent.device)
        action_log_probs = torch.zeros_like(rewards).to(agent.device)
        old_action_log_probs = torch.zeros_like(rewards).to(agent.device)
        ents = torch.zeros_like(rewards).to(agent.device)
        mask = torch.zeros_like(rewards).to(agent.device)

        old_actor = deepcopy(agent.actor).to(agent.device)

        for timestep in range(hyp.num_timesteps):
            actions, log_probs, values, entropy = agent.get_action(states)
            old_log_probs = old_actor(states).log_prob(actions)
            states, reward, done, _, _ = env_wrapper.step(
                actions.cpu().numpy().reshape(env_wrapper.env.action_space.shape)
            )

            rewards[timestep] = torch.tensor(reward).squeeze().to(agent.device)
            value_estimates[timestep] = values.squeeze().to(agent.device)
            action_log_probs[timestep] = log_probs.squeeze().to(agent.device)
            old_action_log_probs[timestep] = old_log_probs.squeeze().to(agent.device)
            ents[timestep] = entropy.squeeze().to(agent.device)
            mask[timestep] = torch.tensor(1 - done).squeeze().to(agent.device)

        advantage = agent.calculate_gae(
            rewards,
            value_estimates,
            mask,
            gamma=hyp.gamma,
            lambda_=hyp.td_lambda_lambda,
        )

        agent.ppo_update(advantage, action_log_probs, old_action_log_probs, hyp.epsilon)

        return states, advantage, ents


@dataclass
class A2CTraining(AgentTrainingParameters):
    """Hyperparametrs for A2C training, with default values"""

    entropy_coef: float = 0.1
    update_name: str = "a2c"

    @staticmethod
    def episode(
        intal_state: np.ndarray,
        agent: Agent,
        hyp: A2CTraining,
        env_wrapper: RecordEpisodeStatistics,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        states = intal_state
        rewards = torch.zeros((hyp.num_timesteps, hyp.num_envs)).to(agent.device)
        value_estimates = torch.zeros_like(rewards).to(agent.device)
        action_log_probs = torch.zeros_like(rewards).to(agent.device)
        ents = torch.zeros_like(rewards).to(agent.device)
        mask = torch.zeros_like(rewards).to(agent.device)

        for timestep in range(hyp.num_timesteps):
            actions, log_probs, values, entropy = agent.get_action(states)

            states, reward, done, _, _ = env_wrapper.step(
                actions.cpu().numpy().reshape(env_wrapper.env.action_space.shape)
            )

            rewards[timestep] = torch.tensor(reward).squeeze().to(agent.device)
            value_estimates[timestep] = values.squeeze().to(agent.device)
            action_log_probs[timestep] = log_probs.squeeze().to(agent.device)
            ents[timestep] = entropy.squeeze().to(agent.device)
            mask[timestep] = torch.tensor(1 - done).squeeze().to(agent.device)

        advantage = agent.calculate_gae(
            rewards,
            value_estimates,
            mask,
            gamma=hyp.gamma,
            lambda_=hyp.td_lambda_lambda,
        )
        agent.a2c_update(advantage, action_log_probs, entropy.mean(), hyp.entropy_coef)
        return states, advantage, ents
