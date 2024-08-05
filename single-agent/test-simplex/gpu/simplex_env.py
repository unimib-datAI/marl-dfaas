import numpy as np

from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env

from RL4CC.environment.base_environment import BaseEnvironment

class SimplexTestEnv(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)

        # Overwrite self.action_space with a custom action space.
        self.action_space = Simplex(shape=(3,))

    def step(self, action):
        assert type(action) in [tuple, np.ndarray], f"action is not an array/tuple, is a {type(action)} with content {action}"

        assert len(action) == 3, f"action must of length 3, now is length {len(action)}"

        obs, reward, done, truncated, info = super().step(action)
        info["given_action"] = (action[0], action[1], action[2])

        return obs, reward, done, truncated, info


register_env("SimplexTestEnv", lambda config: SimplexTestEnv(config))