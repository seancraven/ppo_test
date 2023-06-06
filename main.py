# pylint: disable=no-member
"""
Script for experiments.
"""
import os
import random
from typing import Dict
from typing import List

import gymnasium as gym

import src.policy_gradient_algorithms
from src.agent import Agent
from src.policy_gradient_algorithms import A2CTraining
from src.policy_gradient_algorithms import AgentTraining
from src.policy_gradient_algorithms import PPOTraining


def train_agents(
    updates: List[AgentTraining],
    envs: Dict[str, gym.vector.VectorEnv],
    seeds: List[int],
):
    """Loop over all environments and training algorithms and train agents."""
    for update in updates:
        for name, env in envs.items():
            for seed in seeds:
                directory = os.path.join(
                    os.curdir, "experiments", name, update.update_name
                )
                update.seed = seed
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                agent = Agent(env.single_observation_space, env.single_action_space)

                src.policy_gradient_algorithms.train_agent(
                    agent,
                    env,
                    update,
                    dir_name=directory,
                )


if __name__ == "__main__":
    update_list = [A2CTraining(num_envs=10), PPOTraining(num_envs=10)]
    environments = {
        "CartPole-v1": gym.vector.make("CartPole-v1", num_envs=10),
        "LunarLander-v2": gym.vector.make("LunarLander-v2", num_envs=10),
        "Acrobot-v1": gym.vector.make("Acrobot-v1", num_envs=10),
    }
    random_seeds = [random.randint(0, 100) for _ in range(8)]
    train_agents(update_list, environments, random_seeds)
