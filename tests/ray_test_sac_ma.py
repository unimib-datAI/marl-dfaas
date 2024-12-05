# This script tests Ray RLlib with the SAC algorithm using a demo multi-agent
# environment.
#
# When reaching the 11° iteration (or one more or less), the training breaks
# because there is an error with the tensor, I suspect because there is an
# explosion of the gradient. I think the solution is to do a hyperparameter
# tuning, but I have not tried it.
from pathlib import Path
import logging

import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.policy.policy import PolicySpec
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


class MultiEnvTest(MultiAgentEnv):
    def __init__(self, config=None):
        # Required by Ray.
        self._agent_ids = {"agent_0", "agent_1"}

        # Each agent has a different action space.
        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict(
            {"agent_0": Simplex(shape=(3,)), "agent_1": Simplex(shape=(2,))}
        )

        # Each agent has a different observation space.
        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict(
            {
                "agent_0": gym.spaces.Dict(
                    {
                        "random_0": gym.spaces.Box(low=50, high=150, dtype=np.int32),
                        "random_1": gym.spaces.Box(low=50, high=150, dtype=np.int32),
                    }
                ),
                "agent_1": gym.spaces.Box(low=0, high=1, dtype=np.int32),
            }
        )

        self.max_steps = 100

        super().__init__()

    def reset(self, seed=None, options=None):
        self.current_step = 0

        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        obs = self.observation_space.sample()

        rewards = {
            "agent_0": self.np_random.random(),
            "agent_1": self.np_random.random(),
        }

        finished = self.current_step == self.max_steps
        terminated = {agent: finished for agent in self._agent_ids}
        terminated["__all__"] = all(terminated.values())

        truncated = {agent: False for agent in self._agent_ids}
        truncated["__all__"] = False

        return obs, rewards, terminated, truncated, {}


register_env("MultiEnvTest", lambda env_config: MultiEnvTest(config=env_config))


def main(checkpoint_path=None):
    dummy_env = MultiEnvTest()

    # Since we are running a multi-agent environment, we need to specify each
    # policy to be trained. In this case, each agent has an associated fixed
    # policy (also because each agent has a different action and observation
    # space).
    policies = {
        "policy_agent_0": PolicySpec(
            policy_class=None,
            observation_space=dummy_env.observation_space["agent_0"],
            action_space=dummy_env.action_space["agent_0"],
            config=None,
        ),
        "policy_agent_1": PolicySpec(
            policy_class=None,
            observation_space=dummy_env.observation_space["agent_1"],
            action_space=dummy_env.action_space["agent_1"],
            config=None,
        ),
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        """This function is called at each step to assign the agent to a policy.
        In this case, each agent has a fixed corresponding policy."""
        return f"policy_{agent_id}"

    # Algorithm config.
    sac_config = (
        SACConfig()
        # By default RLlib uses the new API stack, but I use the old one.
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .environment(env="MultiEnvTest")
        .framework("torch")
        .training(
            replay_buffer_config={"type": "MultiAgentPrioritizedReplayBuffer"}
        )  # Buffer using default parameters.
        .env_runners(num_env_runners=0)
        .evaluation(evaluation_interval=None)
        .resources(num_gpus=1)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    )

    # Build the experiment.
    sac_algo = sac_config.build()
    logger.info("Algorithm initialized")
    logger.info(f"Experiment directory created at {sac_algo.logdir!r}")

    # Run the training phase.
    for iteration in range(100):
        logger.info(f"Iteration {iteration}")
        sac_algo.train()
    logger.info("Training terminated")

    sac_algo.stop()
    logger.info("Training stopped")


if __name__ == "__main__":
    main()
