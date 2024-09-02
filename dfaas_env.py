import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env


class DFaaS(MultiAgentEnv):
    def __init__(self, config={}):
        # Number and IDs of the agents in the DFaaS network.
        self.agents = 2
        self.agent_ids = ["node_0", "node_1"]

        # This attribute is required by Ray and must be a set. I use the list
        # version instead in this environment.
        self._agent_ids = set(self.agent_ids)

        # It is the possible max and min value for the reward returned by the
        # step() call.
        self.reward_range = (.0, 1.)

        # The size of each agent's local queue. The queue can be filled with
        # requests to be processed locally.
        self.queue_capacity_max = {
                "node_0": config.get("queue_capacity_max_node_0", 100),
                "node_1": config.get("queue_capacity_max_node_1", 100),
                }

        # Provide full (preferred format) observation- and action-spaces as
        # Dicts mapping agent IDs to the individual agents' spaces.

        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict({
            # Distribution of how many requests are processed locally, forwarded
            # and rejected.
            "node_0": Simplex(shape=(3,)),

            # Distribution of how many requests are processed locally and
            # rejected.
            "node_1": Simplex(shape=(2,))
            })

        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict({
            "node_0": gym.spaces.Dict({
                # Number of input requests to process for a single step.
                "input_requests": gym.spaces.Box(low=50, high=150, dtype=np.int32),

                # Queue capacity (currently a constant).
                "queue_capacity": gym.spaces.Box(low=0, high=self.queue_capacity_max["node_0"], dtype=np.int32),

                # Forwarding capacity (depends on node_1 queue).
                "forward_capacity": gym.spaces.Box(low=0, high=self.queue_capacity_max["node_1"], dtype=np.int32)
                 }),

            "node_1": gym.spaces.Dict({
                # Number of input requests to process for a single step.
                "input_requests": gym.spaces.Box(low=50, high=150, dtype=np.int32),

                # Queue capacity (currently a constant).
                "queue_capacity": gym.spaces.Box(low=0, high=self.queue_capacity_max["node_1"], dtype=np.int32)
                })
            })

        # Number of steps in the environment.
        self.max_steps = config.get("max_steps", 100)

        # The master seed for the RNG. This is used in each episode (each
        # "reset()") to create a new RNG that will be used for generating input
        # requests.
        #
        # Using the master seed make sure to generate a reproducible sequence of
        # seeds.
        self.master_seed = config.get("seed", 0)
        self.master_rng = np.random.default_rng(seed=self.master_seed)

        super().__init__()

    def get_config(self):
        """Returns a dictionary with the current configuration of the
        environment."""
        config = {}
        config["queue_capacity_max_node_0"] = self.queue_capacity_max["node_0"]
        config["queue_capacity_max_node_1"] = self.queue_capacity_max["node_1"]
        config["max_steps"] = self.max_steps
        config["seed"] = self.master_seed
        return config

    def reset(self, *, seed=None, options=None):
        # Current step.
        self.current_step = 0

        # Seed used for this episode.
        if isinstance(options, dict) and "override_seed" in options:
            # By default, the seed is generated. But can be overrided (usually
            # on manual calls, not from Ray).
            self.seed = options["override_seed"]
        else:
            iinfo = np.iinfo(np.uint32)
            self.seed = self.master_rng.integers(0, high=iinfo.max, size=1)[0]

        # Create the RNG used to generate input requests.
        self.rng = np.random.default_rng(seed=self.seed)
        self.np_random = self.rng  # Required by the Gymnasium API

        # Generate all input requests for the environment.
        self.input_requests = self._get_input_requests()

        # Queue state for each agent (number of requests to process locally).
        # The queues start empty (max capacity) and can be full.
        self.queue = {agent: 0 for agent in self.agent_ids}

        # This variable holds the action, excess, and reward for the current
        # step. It is None because there is no action in the first step. Mainly
        # used by the _additional_info() method.
        self.last_action_data = None

        obs = self._build_observation()
        self.last_obs = obs
        info = self._additional_info()

        return obs, info

    def step(self, action_dict):
        # Action for node_0.
        input_requests_0 = self.input_requests["node_0"][self.current_step]

        # Convert the action distribution (a distribution of probabilities) into
        # the number of requests to locally process, to forward and to reject.
        action_0 = self._convert_distribution_0(input_requests_0, action_dict["node_0"])

        # Action for node_1.
        input_requests_1 = self.input_requests["node_1"][self.current_step]

        # Convert the action distribution (a distribution of probabilities) into
        # the number of requests to locally process and reject.
        action_1 = self._convert_distribution(input_requests_1, action_dict["node_1"])

        # We have the actions, now update the environment state.
        excess = self._manage_workload(action_0, action_1)

        # Calculate the reward for both agents.
        rewards = {}
        rewards["node_0"] = self._calculate_reward_0(action_0, excess["node_0"], self.last_obs["node_0"]["forward_capacity"])
        rewards["node_1"] = self._calculate_reward(action_1, excess["node_1"])

        # Make sure the reward is of type float.
        for agent in self.agent_ids:
            rewards[agent] = float(rewards[agent])

        self.last_action_data = (action_0, action_1, excess, rewards)

        # Go to the next step.
        self.current_step += 1

        # Free the queues (for now...).
        self.queue = {agent: 0 for agent in self.agent_ids}

        # Each key in the terminated dictionary indicates whether an individual
        # agent has terminated. There is a special key "__all__" which is true
        # only if all agents have terminated.
        terminated = {agent: False for agent in self.agent_ids}
        if self.current_step == self.max_steps:
            # We are past the last step: nothing more to do.
            terminated = {agent: True for agent in self.agent_ids}
        terminated["__all__"] = all(terminated.values())

        # Truncated is always set to False because it is not used.
        truncated = {agent: False for agent in self.agent_ids}
        truncated["__all__"] = False

        if self.current_step < self.max_steps:
            obs = self._build_observation()
        else:
            # Return a dummy observation because this is the last step.
            obs = self.observation_space_sample()
        self.last_obs = obs

        # Create the additional information dictionary.
        info = self._additional_info()

        return obs, rewards, terminated, truncated, info

    def _build_observation(self):
        """Builds and returns the observation for the current step."""
        assert self.current_step < self.max_steps

        # Initialize the observation dictionary.
        obs = {agent: {} for agent in self.agent_ids}

        # Set common observation values for the agents.
        for agent in self.agent_ids:
            # The queue capacity is always a fixed value for now.
            obs[agent]["queue_capacity"] = np.array([self.queue_capacity_max[agent]], dtype=np.int32)

            input_requests = self.input_requests[agent][self.current_step]
            obs[agent]["input_requests"] = np.array([input_requests], dtype=np.int32)

        # Only node_0 has forwarding capacity. The value depends on the input
        # requests and the queue capacity of node_1 in the next step.
        #
        # The value is non-negative. If it's zero, it means that node_1 in the
        # next step cannot accept any forwarded requests from node_0. If it's
        # positive, node_1 can accept a maximum number of forwarded requests,
        # but does not guarantee processing.
        input_requests = self.input_requests["node_1"][self.current_step]
        forward_capacity = self.queue_capacity_max["node_1"] - input_requests
        if forward_capacity < 0:
            forward_capacity = 0
        obs["node_0"]["forward_capacity"] = np.array([forward_capacity], dtype=np.int32)

        return obs

    def _calculate_reward(self, action, excess):
        """Returns the reward for the agent "node_1" for the current step.

        The reward is based on:

            - The action, a 2-length tuple containing the number of requests to
              process locally  and to reject.

            - The excess, a 1-length tuple containing the local requests that
              exceed the queue capacity.

        This reward function assumes that the queue starts empty at each
        step."""
        assert len(action) == 2, "Expected (local, reject)"
        assert len(excess) == 1, "Expected (local_excess)"

        reqs_total = sum(action)
        reqs_local, reqs_reject = action
        local_excess = excess[0]

        # The agent (policy) tried to be sneaky, but it is not possible to
        # locally process more requests than the internal limit for each step.
        # This behavior must be discouraged by penalizing the reward, but not as
        # much as by rejecting too many requests (the .5 factor).
        if local_excess > 0:
            return 1 - (local_excess / self.queue_capacity_max["node_1"]) * .5

        # If there are more input requests than available slots in the agent
        # queue, the optimal strategy should be to fill the queue and then
        # reject the other requests.
        if reqs_total > self.queue_capacity_max["node_1"]:
            # The reward penalises the agent if the action doesn't maximise the
            # request process locally.
            if reqs_local < self.queue_capacity_max["node_1"]:
                # The new value is the number of rejected requests that will be
                # considered a penalty for the reward. Note that some rejections
                # are inevitable and will not be penalized, only those that can
                # be processed locally but the agent didn't.
                reqs_reject = self.queue_capacity_max["node_1"] - reqs_local
                reqs_total = self.queue_capacity_max["node_1"]
            else:
                reqs_reject = 0

        # The reward is a range from 0 to 1. It decreases as the number of
        # unnecessary rejected requests increases.
        return 1 - reqs_reject / reqs_total

    def _calculate_reward_0(self, action, excess, forward_capacity):
        """Returns the reward for the agent "node_0" for the current step.

        The reward is based on:

            - The action, a 3-length tuple containing the number of requests to
              process locally, to forward and to reject.

            - The excess, a 3-length tuple containing the local requests that
              exceed the queue capacity, the forwarded requests that exceed the
              forwarding capacity, and the forwarded requests that were rejected
              by the other agent.

        This reward function assumes that the queue starts empty at each step.
        """
        assert len(action) == 3, "Expected (local, forward, reject)"
        assert len(excess) == 3, "Expected (local_excess, forward_excess, forward_reject)"

        reqs_total = sum(action)
        reqs_local, reqs_forward, reqs_reject = action
        local_excess, forward_excess, forward_reject = excess
        assert local_excess <= reqs_local
        assert forward_excess <= reqs_forward
        assert forward_reject <= reqs_forward

        assert forward_capacity >= 0

        reward = 1

        # The agent (policy) tried to be sneaky, but it is not possible to
        # locally process more requests than the internal limit for each step.
        # This behavior must be discouraged by penalizing the reward, but not as
        # much as by rejecting too many requests (the .5 factor).
        if local_excess > 0:
            reward -= (local_excess / self.queue_capacity_max["node_0"]) * .6

        # The same also for forwarding.
        if forward_excess > 0:
            if forward_capacity > 0:
                reward -= (forward_excess / forward_capacity) * .3
            else:
                reward -= .3

        if forward_capacity > 0:
            reward -= (forward_reject / forward_capacity) * .4

        # If there are more requests than the node can handle locally, the
        # optimal strategy should be to process all possible requests locally
        # and forward or reject the extra ones.
        if reqs_total > self.queue_capacity_max["node_0"]:
            # The reward penalises the agent if the action doesn't maximise the
            # request process locally.
            if reqs_local < self.queue_capacity_max["node_0"]:
                # The new value is the number of rejected requests that will be
                # considered a penalty for the reward. Note that some rejections
                # are inevitable and will not be penalized, only those that can
                # be processed locally but the agent didn't.
                reqs_reject = self.queue_capacity_max["node_0"] - reqs_local - reqs_forward
                reqs_reject = np.clip(reqs_reject, a_min=0, a_max=None)
                reqs_total = self.queue_capacity_max["node_0"] + reqs_forward
            elif reqs_forward < forward_capacity:
                reqs_reject = reqs_reject - (forward_capacity - reqs_forward)
                reqs_reject = np.clip(reqs_reject, a_min=0, a_max=None)
                reqs_total = self.queue_capacity_max["node_0"] + reqs_forward
            else:
                reqs_reject = 0

        # The reward is a range from 0 to 1. It decreases as the number of
        # unnecessary rejected requests increases.
        reward -= (reqs_reject / reqs_total) * 2

        reward = np.clip(reward, .0, 1.)
        return reward

    @staticmethod
    def _convert_distribution_0(input_requests, action_dist):
        """Converts the given action distribution (e.g. [.7, .2, .1]) into the
        absolute number of requests to process locally, to forward and to
        reject. Returns the result as a tuple.

        This function is only for node_0 agent."""
        assert len(action_dist) == 3, "Expected (local, forward, reject)"

        # Extract the three actions from the action distribution
        prob_local, prob_forwarded, prob_rejected = action_dist

        # Get the corresponding number of requests for each action. Note: the
        # number of requests is a discrete number, so there is a fraction of the
        # action probabilities that is left out of the calculation.
        actions = [
                int(prob_local * input_requests),  # local requests
                int(prob_forwarded * input_requests),  # forwarded requests
                int(prob_rejected * input_requests)]  # rejected requests

        processed_requests = sum(actions)

        # There is a fraction of unprocessed input requests. We need to fix this
        # problem by assigning the remaining requests to the higher fraction for
        # the three action probabilities, because that action is the one that
        # loses the most.
        if processed_requests < input_requests:
            # Extract the fraction for each action probability.
            fractions = [prob_local * input_requests - actions[0],
                         prob_forwarded * input_requests - actions[1],
                         prob_rejected * input_requests - actions[2]]

            # Get the highest fraction index and and assign remaining requests
            # to that action.
            max_fraction_index = np.argmax(fractions)
            actions[max_fraction_index] += input_requests - processed_requests

        assert sum(actions) == input_requests
        return tuple(actions)

    @staticmethod
    def _convert_distribution(input_requests, action_dist):
        """Converts the given action distribution (e.g. [.7, .3]) into the
        absolute number of requests to process locally or reject. Returns the
        result as a tuple.

        This function is only for node_1 agent."""
        assert len(action_dist) == 2, "Expected (local, reject)"

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
        return tuple(actions)

    def _manage_workload(self, action_0, action_1):
        """Fills the agent queues with the requests provided by the actions and
        returns a dictionary containing the surplus of local requests for node_0
        and node_1, the forwarded surplus, and the forwarded requests that were
        rejected for node_0."""
        local_0, forward_0, _ = action_0
        local_1, _ = action_1

        # Helper function.
        def fill_queue(agent, requests):
            """Fill the queue of the specified agent with the specified number
            of requests. Returns the number of excess requests (requests that
            cannot be added to the queue)."""
            excess = 0

            free_slots = self.queue_capacity_max[agent] - self.queue[agent]
            if free_slots >= requests:
                # There are enough slots in the queue to handle the number of
                # requests specified by the action.
                self.queue[agent] += requests
            else:
                # The requests specified by the action do not have enough slots
                # in the queue to be processed locally. So we have a number of
                # requests that are overflowing.
                excess = requests - free_slots

                # However, use the available slots.
                self.queue[agent] += requests - excess

            return excess

        # Local processing for node_0.
        local_excess_0 = fill_queue("node_0", local_0)

        # Forwarding for node_0.
        forward_excess = 0
        forward_capacity = self.last_obs["node_0"]["forward_capacity"]
        if forward_0 > forward_capacity:
            # Too many requests were attempted to be forwarded.
            forward_excess = forward_0 - forward_capacity

            # forward_excess is a np.ndarray, must be unwrapped.
            forward_excess = forward_excess.item()

            # Limit the forwarded requests to the maximum.
            forward_0 = forward_capacity

        # Local processing for node_1. Input requests for node_1 have the
        # priority over the forwarded requests from node_0.
        local_excess_1 = fill_queue("node_1", local_1)

        # Handle now forwarded requests.
        forward_reject = fill_queue("node_1", forward_0)

        retval = {"node_0": (local_excess_0, forward_excess, forward_reject),
                  "node_1": (local_excess_1,)}
        return retval

    def _get_input_requests(self):
        """Calculate the input requests for all agents for all steps.

        Returns a dictionary whose keys are the agent IDs and whose value is an
        np.ndarray containing the input requests for each step."""
        average_requests = 100
        period = 50
        amplitude_requests = 50
        noise_ratio = .1

        input_requests = {}
        steps = np.arange(self.max_steps)
        for agent in self.agent_ids:
            # TODO: do not directly check the value of the agent ID.
            fn = np.sin if agent == "node_0" else np.cos

            base_input = average_requests + amplitude_requests * fn(2 * np.pi * steps / period)
            noisy_input = base_input + noise_ratio * self.rng.normal(0, amplitude_requests, size=self.max_steps)
            input_requests[agent] = np.asarray(noisy_input, dtype=np.int32)
            np.clip(input_requests[agent], 50, 150, out=input_requests[agent])

        return input_requests

    def _additional_info(self):
        """Builds and returns the info dictionary for the current step."""
        # Each agent has its own additional information dictionary.
        info = {agent: {} for agent in self.agent_ids}

        # This special key contains information that is not specific to an
        # individual agent.
        info["__common__"] = {"current_step": self.current_step}

        for agent in self.agent_ids:
            info[agent]["observation"] = self.last_obs[agent]

        # Also save the actions, excess and rewards from the last step.
        if self.last_action_data is not None:
            info["node_0"]["action"] = {
                    "local": self.last_action_data[0][0],
                    "forward": self.last_action_data[0][1],
                    "reject": self.last_action_data[0][2]}
            info["node_1"]["action"] = {
                    "local": self.last_action_data[1][0],
                    "reject": self.last_action_data[1][1]}

            info["node_0"]["excess"] = {
                    "local_excess": self.last_action_data[2]["node_0"][0],
                    "forward_excess": self.last_action_data[2]["node_0"][1],
                    "forward_reject": self.last_action_data[2]["node_0"][2],
                    }
            info["node_1"]["excess"] = {
                    "local_excess": self.last_action_data[2]["node_1"][0]
                    }

            info["node_0"]["reward"] = self.last_action_data[3]["node_0"]
            info["node_1"]["reward"] = self.last_action_data[3]["node_1"]

        return info


# Register the environment with Ray so that it can be used automatically when
# creating experiments.
register_env("DFaaS", lambda env_config: DFaaS(config=env_config))
