# This script tests the Ray's ability to save and load model checkpoints using
# the Simplex space as the action or observation space.
from pathlib import Path
import logging

import gymnasium as gym

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env

# Disable Ray's warnings.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logger for this module.
logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s", level=logging.DEBUG)
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


def create_and_save():
    # Algorithm config.
    ppo_config = (PPOConfig()
                  .environment(env="SimplexTest")
                  .framework("torch")
                  .rollouts(num_rollout_workers=0)  # Only a local worker.
                  .evaluation(evaluation_interval=None)
                  .resources(num_gpus=1)
                  )

    # Build the experiment.
    ppo_algo = ppo_config.build()
    logger.info("Algorithm initialized")

    # Run the training phase for 2 iterations.
    for iteration in range(2):
        logger.info(f"Iteration {iteration}")
        ppo_algo.train()
    logger.info("Training terminated")

    saved = ppo_algo.save()
    path_saved = saved.checkpoint.path
    logger.info(f"Checkpoint saved to {path_saved!r}")

    ppo_algo.stop()
    logger.info("Training stopped")

    return path_saved


def load_and_train(checkpoint_path):
    ppo_algo = Algorithm.from_checkpoint(checkpoint_path)
    logger.info(f"Algorithm restored from {checkpoint_path!r}")

    checkpoint_iter = ppo_algo.iteration
    for iteration in range(checkpoint_iter+1, checkpoint_iter+3):
        logger.info(f"Iteration {iteration}")
        ppo_algo.train()
    logger.info("Training terminated")

    ppo_algo.stop()
    logger.info("Training stopped")

    saved = ppo_algo.save()
    path_saved = saved.checkpoint.path
    logger.info(f"Checkpoint saved to {path_saved!r}")


if __name__ == "__main__":
    checkpoint_path = create_and_save()

    load_and_train(checkpoint_path)
