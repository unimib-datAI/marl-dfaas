"""This module defined a custom algorithm for the DFaaS (or compatible)
environment in which agents always chose to process incoming input requests
locally.

It defines a custom policy (APLPolicy), a custom algorithm configuration
(APLConfig), and the algorithm (APL)."""

import numpy as np

from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms import Algorithm, AlgorithmConfig, registry
from ray.rllib.utils.metrics import (
    SAMPLE_TIMER,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
)
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.spaces.simplex import Simplex


class APLPolicy(Policy):
    """Policy of "Always Process Locally" algorithm."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.env_name = config["env"]
        if self.env_name == "SingleDFaaS":
            # This is a special env, it does not take any action.
            return

        if not isinstance(action_space, Simplex):
            raise ValueError(
                f"Only Simplex action space is supported by APLPolicy, found {action_space.__class__!r}"
            )

        if action_space.shape != (3,):
            raise ValueError(f"Only shape=(3,) supported, found {action_shape.shape}")

    def compute_actions(self, obs_batch, state_batches, **kwargs):
        """Computes actions for the current policy.

        All other arguments are ingored and captured by **kwargs.

        Returns only the batch of output actions."""
        observations = len(obs_batch)

        if self.env_name == "SingleDFaaS":
            # Dummy action, it is not considered.
            actions = np.repeat([0], observations, axis=0)
        else:
            # Always process locally the incoming requests.
            actions = np.repeat([[1.0, 0.0, 0.0]], observations, axis=0)

        return actions, [], {}

    def get_weights(self):
        """Returns an empty dictionary since there is no neural network
        involved.

        Method required to save a checkpoint."""
        return {}

    def set_weights(self, weights):
        """Does nothing for this policy. Required to load from a checkpoint."""
        pass


# Register the APLPolicy in the global registry to ensure that checkpointed
# experiments can be loaded in the future.
registry.POLICIES["APLPolicy"] = f"{__name__}.APLPolicy"


class APLConfig(AlgorithmConfig):
    """Always Process Locally algorithm configuration."""

    def __init__(self, algo_class=None):
        super().__init__(algo_class=APL)

        # Not used, but is to avoid a warning when disabling new API stack.
        self.exploration_config = {"type": "StochasticSampling"}

        # Same as PPO and SAC: at each iteration, collect a fixed number of
        # experiences based on the train batch size.
        self.rollout_fragment_length = "auto"


class APL(Algorithm):
    """Always process locally algorithm.

    This is a simple heuristic that always chooses to process incoming input
    requests locally."""

    @classmethod
    def get_default_config(cls):
        """Returns the default configuration for APL algorithm."""
        return APLConfig(cls)

    @classmethod
    def get_default_policy_class(cls, config):
        """Returns the default policy class for APL algorithm."""
        return APLPolicy

    def training_step(self) -> None:
        """Default single iteration logic of an algorithm."""
        # Most of this code is inspired by the PPO training step.

        # Collect batches from sample workers until we have a full batch.
        with self._timers[SAMPLE_TIMER]:
            train_batch = synchronous_parallel_sample(
                worker_set=self.env_runner_group,
                max_env_steps=self.config.total_train_batch_size,
                sample_timeout_s=self.config.sample_timeout_s,
            )

            # Return early if all our workers failed.
            if not train_batch:
                return {}

            train_batch = train_batch.as_multi_agent()
            self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # There is nothing to train, since APL is a simple deterministic
        # heuristic algorithm.

        self.env_runner.set_global_vars(
            {"timestep": self._counters[NUM_AGENT_STEPS_SAMPLED]}
        )

        return {}
