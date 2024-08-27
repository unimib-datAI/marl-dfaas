import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env


class DFaaS(MultiAgentEnv):
    def __init__(self, config={}):
        # Number and IDs of the agents in the  DFaaS network.
        self.agents = 2
        self.agent_ids = ["node_0", "node_1"]

        # This attribute is required by Ray and must be a set. I use the list
        # version instead in this environment.
        self._agent_ids = set(self.agent_ids)

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
            agent: action_space for agent in self.agent_ids
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
            agent: obs_space for agent in self.agent_ids
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
        # Current step of the agents.
        self.current_step = 0

        # The environment is turn-based: on each step() call, only one agent can
        # perform the action. So in each step of the environment, we have to
        # cycle through the agents before moving on to the next step.
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
        current_agent = self.agent_ids[self.turn]

        assert action_dict is not None, "action_dict is required"
        action_dist = action_dict.get(current_agent)
        assert action_dist is not None, f"Expected agent {current_agent!r} but {action_dict = }"

        # Convert the action distribution (a distribution of probabilities) into
        # the number of requests to locally process and reject.
        input_requests = self.input_requests[current_agent][self.current_step]
        reqs_local, reqs_reject = self._convert_distribution(input_requests, action_dist)
        self.last_action = {"local": reqs_local, "reject": reqs_reject}

        # Calculate reward for the current node.
        reward = self._calculate_reward(reqs_local, reqs_reject)
        self.last_reward = reward

        # Go to the next turn, but if all agents in the step have been cycled,
        # go to the next environment step.
        self.turn += 1
        if self.turn == self.agents:
            self.current_step += 1
            self.turn = 0

        # Each key in the terminated dictionary indicates whether an individual
        # agent has terminated. There is a special key "__all__" which is true
        # only if all agents have terminated.
        terminated = {}
        if self.current_step == self.node_max_steps - 1:
            # We are in the last step, so we need to check individual agents.
            for agent_idx in range(self.agents):
                terminated[self.agent_ids[agent_idx]] = self.turn > agent_idx
        elif self.current_step == self.node_max_steps:
            # We are past the last step: nothing more to do.
            terminated = {agent: True for agent in self.agent_ids}
        else:
            # Not the last step: not terminated.
            terminated = {agent: False for agent in self.agent_ids}
        terminated["__all__"] = all(terminated.values())

        # Truncated is always set to False because it is not used.
        truncated = {agent: False for agent in self.agent_ids}
        truncated["__all__"] = False

        next_agent = self.agent_ids[self.turn]

        # Return the observation only if the next agent has not terminated,
        # because if it has, there is no next action.
        obs = {}
        if not terminated[next_agent]:
            input_requests = self.input_requests[next_agent][self.current_step]

            obs = {next_agent: {
                    # See 'reset()' method on why np.array is required.
                    "input_requests": np.array([input_requests], dtype=np.int32),
                    "queue_capacity": np.array([self.max_requests_step], dtype=np.int32)
                    }
                   }

        # Reward for the last agent.
        rewards = {current_agent: reward}

        # Create the additional information dictionary.
        info = self._additional_info()

        return obs, rewards, terminated, truncated, info

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
        for agent in self.agent_ids:
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

        agent = self.agent_ids[self.turn]

        # This method may be called after one of the agents has been terminated.
        # Therefore, we need to check the termination of the current agent.
        if self.current_step < self.node_max_steps:
            input_requests = self.input_requests[agent][self.current_step]

            info["__common__"]["turn"] = agent
            info["__common__"][agent] = {
                    "input_requests": input_requests,
                    "current_step": self.current_step
                    }

        # Note that the last action refers to the previous agent, not the
        # current one!
        if self.last_action is not None:
            assert self.last_reward is not None

            prev_agent = self.agent_ids[(self.turn - 1) % self.agents]
            info["__common__"]["prev_turn"] = prev_agent
            info["__common__"][prev_agent] = {"action": self.last_action,
                                              "reward": self.last_reward}

        return info


# Register the environment with Ray so that it can be used automatically when
# creating experiments.
register_env("DFaaS", lambda env_config: DFaaS(config=env_config))
