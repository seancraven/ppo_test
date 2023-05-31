# pylint: disable=E1101
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from src.agent import Agent


def main():
    num_envs = 10
    num_episodes = 100
    num_timesteps = 128

    envs = gym.vector.make(
        "LunarLander-v2",
        num_envs=num_envs,
    )
    agent = Agent(envs)
    env_wrapper = gym.wrappers.RecordEpisodeStatistics(
        envs, deque_size=num_envs * num_timesteps
    )

    advantages = []
    entropies = []
    states, _ = env_wrapper.reset(seed=42)

    for _ in tqdm(range(num_episodes)):
        rewards = torch.zeros((num_timesteps, num_envs)).to(agent.device)
        value_estimates = torch.zeros_like(rewards).to(agent.device)
        action_log_probs = torch.zeros_like(rewards).to(agent.device)
        ents = torch.zeros_like(rewards).to(agent.device)
        mask = torch.zeros_like(rewards).to(agent.device)

        for timestep in range(num_timesteps):
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
            gamma=0.99,
            lambda_=0.95,
        )

        agent.update(advantage, action_log_probs, ents.mean(), 0.01)

        advantages.append(advantage.detach().cpu().numpy())
        entropies.append(ents.detach().cpu().numpy())

        ## Plotting
    np.stack(advantages)
    np.stack(entropies)
    returns = np.array(env_wrapper.return_queue)
    np.save("returns.npy", returns)
    np.save("advantages.npy", advantages)
    np.save("entropies.npy", entropies)


def ppo():
    num_envs = 10
    num_episodes = 100
    num_timesteps = 128

    envs = gym.vector.make(
        "LunarLander-v2",
        num_envs=num_envs,
    )
    agent = Agent(envs)
    env_wrapper = gym.wrappers.RecordEpisodeStatistics(
        envs, deque_size=num_envs * num_timesteps
    )

    advantages = []
    entropies = []
    states, _ = env_wrapper.reset(seed=42)

    for _ in tqdm(range(num_episodes)):
        rewards = torch.zeros((num_timesteps, num_envs)).to(agent.device)
        value_estimates = torch.zeros_like(rewards).to(agent.device)
        action_log_probs = torch.zeros_like(rewards).to(agent.device)
        ents = torch.zeros_like(rewards).to(agent.device)
        mask = torch.zeros_like(rewards).to(agent.device)
        old_action_log_probs = torch.zeros_like(rewards).to(agent.device)

        old_actor = deepcopy(agent.actor)

        for timestep in range(num_timesteps):
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
            gamma=0.99,
            lambda_=0.95,
        )

        agent.ppo_update(advantage, action_log_probs, old_action_log_probs, 0.1)

        advantages.append(advantage.detach().cpu().numpy())
        entropies.append(ents.detach().cpu().numpy())

        ## Plotting
    np.stack(advantages)
    np.stack(entropies)
    returns = np.array(env_wrapper.return_queue)
    np.save("returns_ppo.npy", returns)
    np.save("advantages_ppo.npy", advantages)
    np.save("entropies_ppo.npy", entropies)


if __name__ == "__main__":
    ppo()
