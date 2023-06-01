"""
Script for experiments.
"""
import os

from src.agent import A2CTraining
from src.agent import Agent
from src.agent import PPOTraining

if __name__ == "__main__":
    a2c = A2CTraining()
    ppo = PPOTraining()
    envs = [
        "LunarLander-v2",
    ]
    seeds = (42,)
    updates = [a2c, ppo]
    for update in updates:
        for env in envs:
            for seed in seeds:
                directory = os.path.join(
                    os.curdir, "experiments", env, update.update_name
                )
                update.seed = seed
                if not os.path.isdir(directory):
                    os.makedirs(directory)

                Agent.train_agent(env, ppo, dir_name=directory)
                Agent.train_agent(env, a2c, dir_name=directory)
