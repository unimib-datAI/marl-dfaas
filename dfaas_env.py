import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env


class DFaaS(MultiAgentEnv):
    def __init__(self, config={}):
        # Number and ID of nodes (agent in Ray term) in DFaaS network.
        self.nodes = 2
        self._agent_ids = {"node_0", "node_1"}

        # It is the possible max and min value for the reward returned by the
        # step() call.
        self.reward_range = (.0, 1.)

        # Maximum number of requests a node can handle in a single step.
        self.max_requests_step = 100

        # Provide full (preferred format) observation- and action-spaces as
        # Dicts mapping agent IDs to the individual agents' spaces.

        # Distribution of how many requests are processed locally and rejected.
        action_space = Simplex(shape=(2,))
        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict({
            # Each agent has the same action space.
            agent: action_space for agent in self._agent_ids
            })

        obs_space = gym.spaces.Dict({
            # Number of input requests to process for a single step.
            "input_requests": gym.spaces.Box(low=50, high=150, dtype=np.int32),

            # Queue capacity (currently a constant).
            "queue_capacity": gym.spaces.Box(low=0, high=self.max_requests_step, dtype=np.int32)
            })
        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict({
            # Each agent has the same observation space.
            agent: obs_space for agent in self._agent_ids
            })

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

        super().__init__()

    def reset(self, *, seed=None, options=None):
        # Current step for each node (from 0 to self.node_max_steps). The
        # episode ends when all nodes end.
        self.current_step = {"node_0": 0, "node_1": 0}

        # The environment is turn-based: each turn, only one node can take an
        # action. This variable indicates which node should take the action.
        self.turn = 0

        # Seed used for this episode.
        iinfo = np.iinfo(np.uint32)
        self.seed = self.master_rng.integers(0, high=iinfo.max, size=1)[0]

        # Create the RNG used to generate input requests.
        self.rng = np.random.default_rng(seed=self.seed)
        self.np_random = self.rng  # Required by the Gymnasium API

        # Generate all input requests for the environment.
        self.input_requests = self._get_input_requests()

        # These values refer to the last action/reward performed by an agent in
        # a single step(). They are set to None because no action/reward was
        # logged in reset(). Used by the _additional_info method.
        self.last_action = None
        self.last_reward = None

        # Start with node_0.
        input_requests = self.input_requests["node_0"][0]

        obs = {"node_0": {
            # Because the observation is a Box, the returned value must be an
            # np.ndarray even if there is a single value, otherwise Ray will
            # throw an error.
            "input_requests": np.array([input_requests], dtype=np.int32),
            "queue_capacity": np.array([self.max_requests_step], dtype=np.int32)}
               }

        info = self._additional_info()

        return obs, info

    def step(self, action_dict=None):
        assert action_dict is not None, "action_dict is required"
        action_dist = action_dict.get(f"node_{self.turn}")
        assert action_dist is not None, f"Expected turn {self.turn} but {action_dict = }"

        current_node_id = f"node_{self.turn}"

        # Convert the action distribution (a distribution of probabilities) into
        # the number of requests to locally process and reject.
        input_requests = self.input_requests[current_node_id][self.current_step[current_node_id]]
        reqs_local, reqs_reject = self._convert_distribution(input_requests, action_dist)
        self.last_action = {"local": reqs_local, "reject": reqs_reject}

        # Calculate reward for the current node.
        reward = self._calculate_reward(reqs_local, reqs_reject)
        self.last_reward = reward

        # Update the observation space for the next turn.
        self._update_observation_space()

        # Terminated and truncated: There is a special value '__all__' that is
        # only enabled when all alerts are terminated.
        terminated = {}
        for node_id in self.current_step:
            terminated[node_id] = self.current_step[node_id] == self.node_max_steps
        terminated["__all__"] = all(terminated.values())
        truncated = {node_id: False for node_id in self.current_step}
        truncated["__all__"] = all(truncated.values())

        next_node_id = f"node_{self.turn}"

        # Return the observation only if the next agent has not terminated,
        # because if it has, there is no next action.
        obs = {}
        if not terminated[next_node_id]:
            input_requests = self.input_requests[next_node_id][self.current_step[next_node_id]]

            obs = {next_node_id: {
                    # See 'reset()' method on why np.array is required.
                    "input_requests": np.array([input_requests], dtype=np.int32),
                    "queue_capacity": np.array([self.max_requests_step], dtype=np.int32)
                    }
                   }

        # Reward for the last agent.
        rewards = {current_node_id: reward}

        # Create the additional information dictionary.
        info = self._additional_info()

        return obs, rewards, terminated, truncated, info

    def _update_observation_space(self):
        """Updates the observation space and moves to the next turn."""
        # Advance the current step for the current agent.
        self.current_step[f"node_{self.turn}"] += 1

        # Change the next expected agent action.
        self.turn = (self.turn + 1) % self.nodes

    def _calculate_reward(self, reqs_local, reqs_reject):
        """Returns the reward for the given action (the number of locally
        processed requests and rejected requests). The reward is a number in the
        range 0 to 1."""
        reqs_total = reqs_local + reqs_reject

        # The agent (policy) tried to be sneaky, but it is not possible to
        # locally process more requests than the internal limit for each step.
        # This behavior must be discouraged by penalizing the reward, but not as
        # much as by rejecting too many requests (the .5 factor).
        if reqs_local > self.max_requests_step:
            reqs_local_exceed = reqs_local - self.max_requests_step
            return 1 - (reqs_local_exceed / 100) * .5

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

    @staticmethod
    def _convert_distribution(input_requests, action_dist):
        """Converts the given action distribution (e.g. [.7, .3]) into the
        absolute number of requests to process locally or reject. Returns the
        result as a tuple."""

        # Extract the single actions probabilities from the array.
        prob_local, prob_reject = action_dist

        # Get the corresponding number of requests for each action. Note: the
        # number of requests is a discrete number, so there is a fraction of the
        # action probabilities that is left out of the calculation.
        actions = [int(prob_local * input_requests),
                   int(prob_reject * input_requests)]

        processed_requests = sum(actions)

        # There is a fraction of unprocessed input requests. We need to fix this
        # problem by assigning the remaining requests to the higher fraction for
        # the three action probabilities, because that action is the one that
        # loses the most.
        if processed_requests < input_requests:
            # Extract the fraction for each action probability.
            fractions = [prob_local * input_requests - actions[0],
                         prob_reject * input_requests - actions[1]]

            # Get the highest fraction index and and assign remaining requests
            # to that action.
            max_fraction_index = np.argmax(fractions)
            actions[max_fraction_index] += input_requests - processed_requests

        assert sum(actions) == input_requests
        return actions

    def _get_input_requests(self):
        """Calculate the input requests for all agents for all steps.

        Returns a dictionary whose keys are the agent IDs and whose value is an
        np.ndarray containing the input requests for each step."""
        average_requests = 100
        period = 50
        amplitude_requests = 50
        noise_ratio = .1

        input_requests = {}
        steps = np.arange(self.max_requests_step)
        for agent in self._agent_ids:
            # TODO: do not directly check the value of the agent ID.
            fn = np.sin if agent == "node_0" else np.cos

            base_input = average_requests + amplitude_requests * fn(2 * np.pi * steps / period)
            noisy_input = base_input + noise_ratio * self.rng.normal(0, amplitude_requests, size=self.max_requests_step)
            input_requests[agent] = np.asarray(noisy_input, dtype=np.int32)
            np.clip(input_requests[agent], 50, 150, out=input_requests[agent])

        return input_requests

    def _additional_info(self):
        """Builds and returns the info dictionary for the current step."""
        # Since DFaaS is a multi-agent environment, the keys of the returned
        # info dictionary must be the agent IDs of the returned observation or
        # the special "__common__" key, otherwise Ray will complain.
        #
        # I do not like this constraint, so I just use the common key.
        info = {"__common__": {}}

        node = f"node_{self.turn}"
        prev_node = f"node_{(self.turn - 1) % self.nodes}"

        if self.current_step[node] < self.node_max_steps:
            input_requests = self.input_requests[node][self.current_step[node]]

            info["__common__"]["turn"] = node
            info["__common__"][node] = {
                    "input_requests": input_requests,
                    "current_step": self.current_step[node]
                    }

        # Note that the last action refers to the previous agent, not the
        # current one!
        if self.last_action is not None:
            assert self.last_reward is not None

            info["__common__"]["prev_turn"] = prev_node
            info["__common__"][prev_node] = {"action": self.last_action,
                                             "reward": self.last_reward}

        return info


# Register the environment with Ray so that it can be used automatically when
# creating experiments.
register_env("DFaaS", lambda env_config: DFaaS(config=env_config))
