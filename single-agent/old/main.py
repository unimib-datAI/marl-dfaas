import numpy as np

import gymnasium as gym

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

import environment

def TrafficManagementEnvCreator(env_config):
    env = gym.wrappers.TimeLimit(environment.TrafficManagementEnv(env_config),
                                 max_episode_steps=100)

    return env

register_env("TrafficManagementEnv", TrafficManagementEnvCreator)

def main():
    env_config = {
        "debug": False
    }

    algo = (
        PPOConfig()
        .framework(framework="torch")
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=1)
        .environment(env="TrafficManagementEnv", env_config=env_config)
        .build()
    )

    episodes = 10
    for i in range(episodes):
        print(f'---------------------')
        print(f'Episode {i} of {episodes}')
        print(f'---------------------')

        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save().checkpoint.path
            print(f"Checkpoint saved in directory {checkpoint_dir}")


def main_old():
    # Register the environment to the Gymnasium registry.
    gym.register(id="TrafficManagementEnv-v0", entry_point="environment:TrafficManagementEnv")

    env = gym.make('TrafficManagementEnv-v0')
    env.reset()

    for step in range(1000):
        print('====================')
        print(f'Timestep {step}')
        print('====================')

        action = np.random.dirichlet([1, 1, 1])
        print(f'Action chosen: {action}')

        state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f'Terminated = {terminated} Truncated = {truncated}')
            env.reset()

        print()

    env.close()

if __name__ == '__main__':
    main()
