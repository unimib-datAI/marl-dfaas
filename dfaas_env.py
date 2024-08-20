import math

import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env

register_env("DFaaS", lambda env_config: DFaaS(config=env_config))


class DFaaS(MultiAgentEnv):
    def __init__(self, config={}):
        super().__init__()

        # Number of nodes in DFaaS network.
        self.nodes = 2

        self.action_space = {"node_0": Simplex(shape=(2,)),
                             "node_1": Simplex(shape=(2,))
                             }

        # For now, the observation space is just the input requests for a single
        # step.
        self.observation_space = {
                "node_0": gym.spaces.Box(low=50, high=150, dtype=np.int32),
                "node_1": gym.spaces.Box(low=50, high=150, dtype=np.int32),
                }

        # Maximum number of requests a node can handle in a single step.
        self.max_requests_step = 100

        # Maximum steps for each node.
        self.node_max_steps = config.get("node_max_steps", 100)

        # The master seed for the RNG. This is used in each episode (each
        # "reset()") to create a new RNG that will be used for generating input
        # requests.
        #
        # Using the master seed make sure to generate a reproducible sequence of
        # seeds.
        self.master_seed = config.get("seed", 0)
        self.master_rng = np.random.default_rng(seed=self.master_seed)

    def reset(self, *, seed=None, options=None):
        # Current step for each node (from 0 to self.node_max_steps). The
        # episode ends when all nodes end.
        self.current_step = {"node_0": 0, "node_1": 0}

        # The environment is turn-based: each turn, only one node can take an
        # action. This flag indicates which node should take the action.
        self.turn = 0

        # Seed used for this episode.
        iinfo = np.iinfo(np.uint32)
        self.seed = self.master_rng.integers(0, high=iinfo.max, size=1)[0]

        # Create the RNG used to generate input requests.
        self.rng = np.random.default_rng(seed=self.seed)
        self.np_random = self.rng  # Required by the Gymnasium API

        # Number of requests for the node whose turn it is. In this case it is
        # for node_0 because it always starts when the environment is reset.
        self.input_requests = self._get_input_requests()

        obs = {"node_0": self.input_requests}
        info = {}

        return obs, info

    def step(self, action_dict=None):
        assert action_dict is not None, "action_dict is required"
        action_dist = action_dict.get(f"node_{self.turn}")
        assert action_dist is not None, f"It is turn {self.turn} but action_dict doesn't contain the action for node_{self.turn}"

        # Convert the action distribution (a distribution of probabilities) into
        # the number of requests to locally process and reject.
        reqs_local, reqs_reject = self._convert_distribution(action_dist)

        # The node cannot locally process more requests than it can handle, so
        # the excess local requests are automatically converted to rejected
        # requests.
        #
        # TODO: There is a problem with this: the agent does not know that the
        # action has exceeded the number of locally handled requests because the
        # reward ignores this.
        if reqs_local > self.max_requests_step:
            local_excess = reqs_local - self.max_requests_step
            reqs_reject += local_excess
            reqs_local -= local_excess

        # Calculate reward for the current node.
        reward = self._calculate_reward(reqs_local, reqs_reject)

        current_node_id = f"node_{self.turn}"

        # Update the observation space for the next turn.
        self._update_observation_space()

        next_node_id = f"node_{self.turn}"

        # Observation for the next agent, the RL agent (RL algorithm) will
        # choose the action for this agent.
        obs = {next_node_id: self.input_requests}

        # Reward for the last agent.
        rewards = {current_node_id: reward}

        # Terminated and truncated: There is a special value '__all__' that is
        # only enabled when all alerts are terminated.
        terminated = {}
        for node_id in self.current_step:
            terminated[node_id] = self.current_step[node_id] == self.node_max_steps
        terminated["__all__"] = all(terminated.values())
        truncateds = {node_id: False for node_id in self.current_step}

        # No information available. TODO: add useful information.
        info = {}

        return obs, rewards, terminated, truncateds, info

    def _update_observation_space(self):
        self.turn = (self.turn + 1) % self.nodes

        self.input_requests = self._get_input_requests()

        self.current_step[f"node_{self.turn}"] += 1

    def _calculate_reward(self, reqs_local, reqs_reject):
        '''Returns the reward for the given action (the number of locally
        processed requests and rejected requests). The reward is a number in the
        range 0 to 1.'''
        reqs_total = reqs_local + reqs_reject

        # If there are more requests than the node can handle locally, the
        # optimal strategy should be to process all possible requests locally
        # and reject the extra ones.
        if reqs_total > self.max_requests_step:
            # The reward penalises the agent if the action doesn't maximise the
            # request process locally.
            if reqs_local < self.max_requests_step:
                # The new value is the number of rejected requests that will be
                # considered a penalty for the reward. Note that some rejections
                # are inevitable and will not be penalized, only those that can
                # be processed locally but the agent didn't.
                reqs_reject = self.max_requests_step - reqs_local
                reqs_total = self.max_requests_step
            else:
                reqs_reject = 0

        # The reward is a range from 0 to 1. It decreases as the number of
        # unnecessary rejected requests increases.
        return 1 - reqs_reject / reqs_total

    def _convert_distribution(self, action_dist):
        # Extract the single actions probabilities from the array.
        prob_local, prob_reject = action_dist

        # Get the corresponding number of requests for each action. Note: the
        # number of requests is a discrete number, so there is a fraction of the
        # action probabilities that is left out of the calculation.
        actions = [int(prob_local * self.input_requests),
                   int(prob_reject * self.input_requests)]

        processed_requests = sum(actions)

        # There is a fraction of unprocessed input requests. We need to fix this
        # problem by assigning the remaining requests to the higher fraction for
        # the three action probabilities, because that action is the one that
        # loses the most.
        if processed_requests < self.input_requests:
            # Extract the fraction for each action probability.
            fractions = [prob_local * self.input_requests - actions[0],
                         prob_reject * self.input_requests - actions[1]]

            # Get the highest fraction index and and assign remaining requests
            # to that action.
            max_fraction_index = np.argmax(fractions)
            actions[max_fraction_index] += self.input_requests - processed_requests

        assert sum(actions) == self.input_requests
        return actions

    def _get_input_requests(self):
        average_requests = 100
        period = 50
        amplitude_requests = 50

        current_step = self.current_step[f"node_{self.turn}"]

        noise_ratio = .1
        base_input = average_requests + amplitude_requests * math.sin(2 * math.pi * current_step / period)
        noisy_input = base_input + noise_ratio * self.rng.normal(0, amplitude_requests)
        input_requests = int(noisy_input)

        # Force values outside limits to respect observation space.
        input_requests = np.clip(input_requests, 50, 150)

        return input_requests
