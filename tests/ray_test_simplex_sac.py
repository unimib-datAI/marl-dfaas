# This script tests Ray RLlib with the SAC algorithm using an environment with
# Simplex as the action space. It has the same problem with
# "ray_test_sac_ma.py".
from pathlib import Path
import logging

import gymnasium as gym

from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env

# Disable Ray's warnings.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logger for this module.
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(Path(__file__).name)


class SimplexTest(gym.Env):
    def __init__(self, config=None):
        self.action_space = Simplex(shape=(3,))
        self.observation_space = gym.spaces.Box(shape=(1,), low=-1, high=1)
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        self.current_step = 0

        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        obs = self.observation_space.sample()
        reward = self.np_random.random()
        terminated = self.current_step == self.max_steps
        return obs, reward, terminated, False, {}


register_env("SimplexTest", lambda env_config: SimplexTest(config=env_config))


def main(checkpoint_path=None):
    # Algorithm config.
    sac_config = (
        SACConfig()
        # By default RLlib uses the new API stack, but I use the old one.
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="SimplexTest")
        .training(
            replay_buffer_config={"type": "MultiAgentPrioritizedReplayBuffer"}
        )  # Buffer using default parameters.
        .env_runners(num_env_runners=0)
        .framework("torch")
        .evaluation(evaluation_interval=None)
        .resources(num_gpus=1)
    )

    # Build the experiment.
    sac_algo = sac_config.build()
    logger.info("Algorithm initialized")
    logger.info(f"Experiment directory created at {sac_algo.logdir!r}")

    # Run the training phase.
    for iteration in range(20):
        logger.info(f"Iteration {iteration}")
        sac_algo.train()
    logger.info("Training terminated")

    sac_algo.stop()
    logger.info("Training stopped")


if __name__ == "__main__":
    main()
