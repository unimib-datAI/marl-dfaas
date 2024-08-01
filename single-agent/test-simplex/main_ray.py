import numpy as np

import gymnasium as gym

from ray.rllib.utils.spaces.simplex import Simplex

class SimplexTestEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = Simplex(shape=(1,3), concentration=np.array([1, 1, 1]))
        self.observation_space = gym.spaces.Box(shape=(1,), low=-1, high=1)

    def reset(self):
        return np.zeros(1)

    def step(self, action):
        return np.zeros(1), 0, False, False, info


def main():
    env = SimplexTestEnv({})

    result = env.step(np.zeros(1))

if __name__ == "__main__":
    main()
