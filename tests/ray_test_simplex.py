# This script simply runs a training experiment of a dummy environment with PPO
# using Ray RLLib.
import gymnasium as gym

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env


class SimplexTest(gym.Env):
    """SimplexTest is a sample environment that basically does nothing."""

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


if __name__ == "__main__":
    # Algorithm config.
    ppo_config = (
        PPOConfig()
        # By default RLlib uses the new API stack, but I use the old one.
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .environment(env="SimplexTest")
        .framework("torch")
        .env_runners(num_env_runners=0)  # Get experiences in the main process.
        .evaluation(evaluation_interval=None)  # No automatic evaluation.
        .resources(num_gpus=1)
    )

    # Build the experiment.
    ppo_algo = ppo_config.build()
    print(f"Algorithm initialized ({ppo_algo.logdir = }")

    iterations = 2
    print(f"Start of training ({iterations = })")
    for iteration in range(iterations):
        print(f"Iteration {iteration}")
        ppo_algo.train()
    print("Training terminated")

    ppo_algo.stop()
    print("Training end")
