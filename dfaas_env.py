# This file contains the DFaaS multi-agent environment and associated callbacks.
#
# Actually contains three (or more) classes:
#
#   1. The DFaaS asymmetric environment (only one node can forward),
#   2. The DFaaS (symmetrical) environment,
#   3. The callbacks used for both environments.
#
# This file may contain additional classes for specialized environments and
# callbacks during experimentation.
import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class DFaaS_ASYM(MultiAgentEnv):
    # String representing the type of this environment.
    type = "SYM"

    def __init__(self, config={}):
        # Number and IDs of the agents in the DFaaS network.
        self.agents = 2
        self.agent_ids = ["node_0", "node_1"]

        # String representing the type of this environment.
        self.suffix = "ASYM"

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

                # Ratio of forwarded rejected requests in the previosus step.
                "forward_reject": gym.spaces.Box(low=0, high=1)
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

        # This variable is used by _additional_info() to build the info
        # dictionary. It is None in reset() because there is no action and
        # reward in this step.
        self.last_info = None

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
        action_1 = self._convert_distribution_1(input_requests_1, action_dict["node_1"])

        # We have the actions, now update the environment state.
        info_work = self._manage_workload(action_0, action_1)

        # Calculate the reward for both agents.
        rewards = {}
        rewards["node_0"] = self._calculate_reward_0_v2(action_0,
                                                        (info_work["node_0"]["local_excess"], info_work["node_0"]["forward_reject"]),
                                                        (self.queue["node_0"], self.queue_capacity_max["node_0"]))
        rewards["node_1"] = self._calculate_reward_1_v2(action_1,
                                                        (info_work["node_1"]["local_excess"],),
                                                        (info_work["node_1"]["queue_status_pre_forward"], self.queue_capacity_max["node_1"]))

        # Make sure the reward is of type float.
        for agent in self.agent_ids:
            rewards[agent] = float(rewards[agent])

        self.last_info = {
                "action": {"node_0": action_0, "node_1": action_1},
                "rewards": rewards,
                "workload": info_work
                }

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

        if self.last_info is None:
            last_forward_reject = 0
        else:
            forward_reqs = self.last_info["action"]["node_0"][1]
            forward_reject = self.last_info["workload"]["node_0"]["forward_reject"]

            if forward_reject != 0:
                last_forward_reject = forward_reject / forward_reqs
            else:
                last_forward_reject = 0
        obs["node_0"]["forward_reject"] = np.array([last_forward_reject], dtype=np.float32)

        return obs

    def _calculate_reward_1(self, action, excess):
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

    @staticmethod
    def _calculate_reward_1_v2(action, excess, queue):
        """Returns the reward for agent "node_0" for the current step.

        The reward is based on:

            - The action, a 3-length tuple containing the number of requests to
              process locally and to reject.

            - The excess, a 1-length tuple containing the number of local
              requests that exceed the queue capacity.

            - The queue, a 2-length tuple containing the status of the queue (0
              is empty) and the maximum capacity of the queue.

        The reward returned is in the range [0, 1]."""
        assert len(action) == 2, "Expected (local, reject)"
        assert len(excess) == 1, "Expected (local_excess)"
        assert len(queue) == 2, "Expected (queue_status, queue_max_capacity)"

        reqs_total = sum(action)
        reqs_local, reqs_reject = action
        local_excess = excess[0]
        queue_status, queue_max = queue

        if reqs_total == 0:
            return 1.

        if local_excess > 0:
            return 1 - (local_excess / reqs_local)

        free_slots = queue_max - queue_status
        assert free_slots >= 0

        # Calculate the number of excess reject requests.
        if free_slots > reqs_reject:
            reject_excess = reqs_reject
        else:  # reqs_reject >= free_slots
            valid_reject = reqs_reject - free_slots
            assert valid_reject >= 0
            reject_excess = reqs_reject - valid_reject
        assert reject_excess >= 0

        return 1 - (reject_excess / reqs_total)

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
    def _calculate_reward_0_v2(action, excess, queue):
        """Returns the reward for agent "node_0" for the current step.

        The reward is based on:

            - The action, a 3-length tuple containing the number of requests to
              process locally, to forward, and to reject.

            - The excess, a 2-length tuple containing the local requests that
              exceed the queue capacity and the forwarded requests that were
              rejected by the other agent.

            - The queue, a 2-length tuple containing the status of the queue (0
              is empty) and the maximum capacity of the queue.

        The reward returned is in the range [0, 1]."""
        assert len(action) == 3, "Expected (local, forward, reject)"
        assert len(excess) == 2, "Expected (local_excess, forward_reject)"
        assert len(queue) == 2, "Expected (queue_status, queue_max_capacity)"

        reqs_total = sum(action)
        reqs_local, reqs_forward, reqs_reject = action
        local_excess, forward_reject = excess
        queue_status, queue_max = queue

        if reqs_total == 0:
            return 1.

        free_slots = queue_max - queue_status
        assert free_slots >= 0

        # Calculate the number of excess forward requests.
        #
        # The forwarded and rejected requests must be excluded from the count
        # because they are considered separately.
        reqs_forward -= forward_reject
        if free_slots > reqs_forward:
            forward_excess = reqs_forward
        else:  # reqs_forward >= free_slots
            valid_forward = reqs_forward - free_slots
            assert valid_forward >= 0
            forward_excess = reqs_forward - valid_forward

        # Calculate the number of excess reject requests.
        free_slots = free_slots - forward_excess
        assert free_slots >= 0
        if free_slots > reqs_reject:
            valid_reject = 0
            reject_excess = reqs_reject
        else:  # reqs_reject >= free_slots
            valid_reject = reqs_reject - free_slots
            assert valid_reject >= 0
            reject_excess = reqs_reject - valid_reject

        # Calculate the number of rejected requests that could have been
        # forwarded.
        if forward_reject == 0 and valid_reject > 0:
            # Assume that all rejected requests could have been forwarded
            # because no forwarded requests were rejected.
            reject_excess += valid_reject
            valid_reject = 0

        assert local_excess >= 0 \
            and forward_reject >= 0 \
            and forward_excess >= 0 \
            and reject_excess >= 0
        wrong_reqs = local_excess + forward_reject + forward_excess + reject_excess
        assert wrong_reqs <= reqs_total, f"({local_excess = } + {forward_reject = } + {forward_excess = } + {reject_excess = }) <= {reqs_total}"

        return 1 - (wrong_reqs / reqs_total)

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
    def _convert_distribution_1(input_requests, action_dist):
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
        returns a dictionary containing information for the two agents:

            - For "node_0": the number of excess local requests and rejected
              forwarded requests,
            - For "node_1": the number of excess local requests, the queue
              status before and after processing forwarded requests from
              "node_0".
        """
        local_0, forward_0, _ = action_0
        local_1, _ = action_1

        info = {agent: {} for agent in self.agent_ids}

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
        info["node_0"]["local_excess"] = fill_queue("node_0", local_0)

        # Local processing for node_1. Input requests for node_1 have the
        # priority over the forwarded requests from node_0.
        info["node_1"]["local_excess"] = fill_queue("node_1", local_1)
        info["node_1"]["queue_status_pre_forward"] = self.queue["node_1"]

        # Handle now forwarded requests.
        info["node_0"]["forward_reject"] = fill_queue("node_1", forward_0)
        info["node_1"]["queue_status_post_forward"] = self.queue["node_1"]

        return info

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
        if self.last_info is not None:
            info["node_0"]["action"] = {
                    "local": self.last_info["action"]["node_0"][0],
                    "forward": self.last_info["action"]["node_0"][1],
                    "reject": self.last_info["action"]["node_0"][2]}
            info["node_1"]["action"] = {
                    "local": self.last_info["action"]["node_1"][0],
                    "reject": self.last_info["action"]["node_1"][1]}

            info["node_0"]["excess"] = {
                    "local_excess": self.last_info["workload"]["node_0"]["local_excess"],
                    "forward_reject": self.last_info["workload"]["node_0"]["forward_reject"],
                    }
            info["node_1"]["excess"] = {
                    "local_excess": self.last_info["workload"]["node_1"]["local_excess"],
                    }

            info["node_1"]["queue_status"] = {
                    "pre_forward": self.last_info["workload"]["node_1"]["queue_status_pre_forward"],
                    "post_forward": self.last_info["workload"]["node_1"]["queue_status_post_forward"]
                    }

            info["node_0"]["reward"] = self.last_info["rewards"]["node_0"]
            info["node_1"]["reward"] = self.last_info["rewards"]["node_1"]

        return info


class DFaaS_ASYM_MULTIPLE_RATIO(MultiAgentEnv):
    # String representing the type of this environment.
    type = "SYM"

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

        # Number of latest steps to return in the observation space (the ratio
        # of forwarded but rejected requests).
        self.forward_reject_steps = config.get("forward_reject_steps", 3)
        assert self.forward_reject_steps >= 1, f"forward_reject_steps must be positive, found {self.forward_reject_steps}"

        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict({
            "node_0": gym.spaces.Dict({
                # Number of input requests to process for a single step.
                "input_requests": gym.spaces.Box(low=50, high=150, dtype=np.int32),

                # Queue capacity (currently a constant).
                "queue_capacity": gym.spaces.Box(low=0, high=self.queue_capacity_max["node_0"], dtype=np.int32),

                # Ratio of forwarded but rejected requests in the previous steps.
                "forward_reject": gym.spaces.Box(low=0, high=1, shape=(self.forward_reject_steps,))
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

        # This variable is used by _additional_info() to build the info
        # dictionary. It is None in reset() because there is no action and
        # reward in this step.
        self.last_info = None

        # This variable is used to store the most recent ratios of forwarded but
        # rejected requests used to build the observation dictionary for the
        # current step.
        self.last_forward_reject_ratios = np.zeros(self.forward_reject_steps, dtype=np.float32)

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
        action_1 = self._convert_distribution_1(input_requests_1, action_dict["node_1"])

        # We have the actions, now update the environment state.
        info_work = self._manage_workload(action_0, action_1)

        # Calculate the reward for both agents.
        rewards = {}
        rewards["node_0"] = self._calculate_reward_0_v2(action_0,
                                                        (info_work["node_0"]["local_excess"], info_work["node_0"]["forward_reject"]),
                                                        (self.queue["node_0"], self.queue_capacity_max["node_0"]))
        rewards["node_1"] = self._calculate_reward_1_v2(action_1,
                                                        (info_work["node_1"]["local_excess"],),
                                                        (info_work["node_1"]["queue_status_pre_forward"], self.queue_capacity_max["node_1"]))

        # Make sure the reward is of type float.
        for agent in self.agent_ids:
            rewards[agent] = float(rewards[agent])

        self.last_info = {
                "action": {"node_0": action_0, "node_1": action_1},
                "rewards": rewards,
                "workload": info_work
                }

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

        # Shift the last saved ratios to the left. The first slot must be zeroed
        # because it is the current ratio for this step.
        self.last_forward_reject_ratios = np.roll(self.last_forward_reject_ratios, 1)
        self.last_forward_reject_ratios[0] = 0

        if self.last_info is not None:
            forward_reqs = self.last_info["action"]["node_0"][1]
            forward_reject = self.last_info["workload"]["node_0"]["forward_reject"]

            if forward_reject > 0:
                self.last_forward_reject_ratios[0] = forward_reject / forward_reqs

        obs["node_0"]["forward_reject"] = self.last_forward_reject_ratios

        return obs

    def _calculate_reward_1(self, action, excess):
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

    @staticmethod
    def _calculate_reward_1_v2(action, excess, queue):
        """Returns the reward for agent "node_0" for the current step.

        The reward is based on:

            - The action, a 3-length tuple containing the number of requests to
              process locally and to reject.

            - The excess, a 1-length tuple containing the number of local
              requests that exceed the queue capacity.

            - The queue, a 2-length tuple containing the status of the queue (0
              is empty) and the maximum capacity of the queue.

        The reward returned is in the range [0, 1]."""
        assert len(action) == 2, "Expected (local, reject)"
        assert len(excess) == 1, "Expected (local_excess)"
        assert len(queue) == 2, "Expected (queue_status, queue_max_capacity)"

        reqs_total = sum(action)
        reqs_local, reqs_reject = action
        local_excess = excess[0]
        queue_status, queue_max = queue

        if reqs_total == 0:
            return 1.

        if local_excess > 0:
            return 1 - (local_excess / reqs_local)

        free_slots = queue_max - queue_status
        assert free_slots >= 0

        # Calculate the number of excess reject requests.
        if free_slots > reqs_reject:
            reject_excess = reqs_reject
        else:  # reqs_reject >= free_slots
            valid_reject = reqs_reject - free_slots
            assert valid_reject >= 0
            reject_excess = reqs_reject - valid_reject
        assert reject_excess >= 0

        return 1 - (reject_excess / reqs_total)

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
    def _calculate_reward_0_v2(action, excess, queue):
        """Returns the reward for agent "node_0" for the current step.

        The reward is based on:

            - The action, a 3-length tuple containing the number of requests to
              process locally, to forward, and to reject.

            - The excess, a 2-length tuple containing the local requests that
              exceed the queue capacity and the forwarded requests that were
              rejected by the other agent.

            - The queue, a 2-length tuple containing the status of the queue (0
              is empty) and the maximum capacity of the queue.

        The reward returned is in the range [0, 1]."""
        assert len(action) == 3, "Expected (local, forward, reject)"
        assert len(excess) == 2, "Expected (local_excess, forward_reject)"
        assert len(queue) == 2, "Expected (queue_status, queue_max_capacity)"

        reqs_total = sum(action)
        reqs_local, reqs_forward, reqs_reject = action
        local_excess, forward_reject = excess
        queue_status, queue_max = queue

        if reqs_total == 0:
            return 1.

        free_slots = queue_max - queue_status
        assert free_slots >= 0

        # Calculate the number of excess forward requests.
        #
        # The forwarded and rejected requests must be excluded from the count
        # because they are considered separately.
        reqs_forward -= forward_reject
        if free_slots > reqs_forward:
            forward_excess = reqs_forward
        else:  # reqs_forward >= free_slots
            valid_forward = reqs_forward - free_slots
            assert valid_forward >= 0
            forward_excess = reqs_forward - valid_forward

        # Calculate the number of excess reject requests.
        free_slots = free_slots - forward_excess
        assert free_slots >= 0
        if free_slots > reqs_reject:
            valid_reject = 0
            reject_excess = reqs_reject
        else:  # reqs_reject >= free_slots
            valid_reject = reqs_reject - free_slots
            assert valid_reject >= 0
            reject_excess = reqs_reject - valid_reject

        # Calculate the number of rejected requests that could have been
        # forwarded.
        if forward_reject == 0 and valid_reject > 0:
            # Assume that all rejected requests could have been forwarded
            # because no forwarded requests were rejected.
            reject_excess += valid_reject
            valid_reject = 0

        assert local_excess >= 0 \
            and forward_reject >= 0 \
            and forward_excess >= 0 \
            and reject_excess >= 0
        wrong_reqs = local_excess + forward_reject + forward_excess + reject_excess
        assert wrong_reqs <= reqs_total, f"({local_excess = } + {forward_reject = } + {forward_excess = } + {reject_excess = }) <= {reqs_total}"

        return 1 - (wrong_reqs / reqs_total)

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
    def _convert_distribution_1(input_requests, action_dist):
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
        returns a dictionary containing information for the two agents:

            - For "node_0": the number of excess local requests and rejected
              forwarded requests,
            - For "node_1": the number of excess local requests, the queue
              status before and after processing forwarded requests from
              "node_0".
        """
        local_0, forward_0, _ = action_0
        local_1, _ = action_1

        info = {agent: {} for agent in self.agent_ids}

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
        info["node_0"]["local_excess"] = fill_queue("node_0", local_0)

        # Local processing for node_1. Input requests for node_1 have the
        # priority over the forwarded requests from node_0.
        info["node_1"]["local_excess"] = fill_queue("node_1", local_1)
        info["node_1"]["queue_status_pre_forward"] = self.queue["node_1"]

        # Handle now forwarded requests.
        info["node_0"]["forward_reject"] = fill_queue("node_1", forward_0)
        info["node_1"]["queue_status_post_forward"] = self.queue["node_1"]

        return info

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
        if self.last_info is not None:
            info["node_0"]["action"] = {
                    "local": self.last_info["action"]["node_0"][0],
                    "forward": self.last_info["action"]["node_0"][1],
                    "reject": self.last_info["action"]["node_0"][2]}
            info["node_1"]["action"] = {
                    "local": self.last_info["action"]["node_1"][0],
                    "reject": self.last_info["action"]["node_1"][1]}

            info["node_0"]["excess"] = {
                    "local_excess": self.last_info["workload"]["node_0"]["local_excess"],
                    "forward_reject": self.last_info["workload"]["node_0"]["forward_reject"],
                    }
            info["node_1"]["excess"] = {
                    "local_excess": self.last_info["workload"]["node_1"]["local_excess"],
                    }

            info["node_1"]["queue_status"] = {
                    "pre_forward": self.last_info["workload"]["node_1"]["queue_status_pre_forward"],
                    "post_forward": self.last_info["workload"]["node_1"]["queue_status_post_forward"]
                    }

            info["node_0"]["reward"] = self.last_info["rewards"]["node_0"]
            info["node_1"]["reward"] = self.last_info["rewards"]["node_1"]

        return info


class DFaaS(MultiAgentEnv):
    # String representing the type of this environment.
    type = "SYM"

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
            agent: Simplex(shape=(3,)) for agent in self.agent_ids
            })

        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict({
            agent: gym.spaces.Dict({
                # Number of input requests to process for a single step.
                "input_requests": gym.spaces.Box(low=50, high=150, dtype=np.int32),

                # Queue capacity (currently a constant).
                "queue_capacity": gym.spaces.Box(low=0, high=self.queue_capacity_max["node_0"], dtype=np.int32),

                # Number of forwarded requests that have been rejected by the
                # other agent in the previous step.
                "forward_reject": gym.spaces.Box(low=0, high=150, dtype=np.int32)
                 }) for agent in self.agent_ids
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

        # This variable is used by _additional_info() to build the info
        # dictionary. It is None in reset() because there is no action and
        # reward in this step.
        self.last_info = None

        obs = self._build_observation()
        self.last_obs = obs
        info = self._additional_info()

        return obs, info

    def step(self, action_dict):
        action = {}  # Absolute number of requests for each action.
        for agent in self.agent_ids:
            # Get input requests.
            input_requests = self.input_requests[agent][self.current_step]

            # Convert the action distribution (a distribution of probabilities)
            # into the number of requests to locally process, to forward and to
            # reject.
            action[agent] = self._convert_distribution(input_requests, action_dict[agent])

        # We have the actions, now update the environment state.
        info_work = self._manage_workload(action)

        # Calculate the reward for both agents.
        rewards = {}
        for agent in self.agent_ids:
            rewards[agent] = self._calculate_reward_v2(action[agent],
                                                       (info_work[agent]["local_excess"], info_work[agent]["forward_reject"]),
                                                       (info_work[agent]["queue_status_pre_forward"], self.queue_capacity_max[agent]))
            # Make sure the reward is of type float.
            rewards[agent] = float(rewards[agent])

        self.last_info = {
                "action": action,
                "rewards": rewards,
                "workload": info_work
                }

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

            if self.last_info is None:
                last_forward_reject = 0
            else:
                last_forward_reject = self.last_info["workload"][agent]["forward_reject"]
            obs[agent]["forward_reject"] = np.array([last_forward_reject], dtype=np.int32)

        return obs

    def _calculate_reward(self, action, excess, forward_capacity):
        """Returns the reward for an agent for the current step.

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
    def _calculate_reward_v2(action, excess, queue):
        """Returns the reward for an agent for the current step.

        The reward is based on:

            - The action, a 3-length tuple containing the number of requests to
              process locally, to forward, and to reject.

            - The excess, a 2-length tuple containing the local requests that
              exceed the queue capacity and the forwarded requests that were
              rejected by the other agent.

            - The queue, a 2-length tuple containing the status of the queue (0
              is empty) and the maximum capacity of the queue.

        The reward returned is in the range [0, 1]."""
        assert len(action) == 3, "Expected (local, forward, reject)"
        assert len(excess) == 2, "Expected (local_excess, forward_reject)"
        assert len(queue) == 2, "Expected (queue_status, queue_max_capacity)"

        reqs_total = sum(action)
        reqs_local, reqs_forward, reqs_reject = action
        local_excess, forward_reject = excess
        queue_status, queue_max = queue

        if reqs_total == 0:
            return 1.

        free_slots = queue_max - queue_status
        assert free_slots >= 0

        # Calculate the number of excess forward requests.
        #
        # The forwarded and rejected requests must be excluded from the count
        # because they are considered separately.
        reqs_forward -= forward_reject
        if free_slots > reqs_forward:
            forward_excess = reqs_forward
        else:  # reqs_forward >= free_slots
            valid_forward = reqs_forward - free_slots
            assert valid_forward >= 0
            forward_excess = reqs_forward - valid_forward

        # Calculate the number of excess reject requests.
        free_slots = free_slots - forward_excess
        assert free_slots >= 0
        if free_slots > reqs_reject:
            valid_reject = 0
            reject_excess = reqs_reject
        else:  # reqs_reject >= free_slots
            valid_reject = reqs_reject - free_slots
            assert valid_reject >= 0
            reject_excess = reqs_reject - valid_reject

        # Calculate the number of rejected requests that could have been
        # forwarded.
        if forward_reject == 0 and valid_reject > 0:
            # Assume that all rejected requests could have been forwarded
            # because no forwarded requests were rejected.
            reject_excess += valid_reject
            valid_reject = 0

        assert local_excess >= 0 \
            and forward_reject >= 0 \
            and forward_excess >= 0 \
            and reject_excess >= 0
        wrong_reqs = local_excess + forward_reject + forward_excess + reject_excess
        assert wrong_reqs <= reqs_total, f"({local_excess = } + {forward_reject = } + {forward_excess = } + {reject_excess = }) <= {reqs_total}"

        return 1 - (wrong_reqs / reqs_total)

    @staticmethod
    def _convert_distribution(input_requests, action_dist):
        """Converts the given action distribution (e.g. [.7, .2, .1]) into the
        absolute number of requests to process locally, to forward and to
        reject. Returns the result as a tuple."""
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

    def _manage_workload(self, action):
        """Fills the agent queues with the requests provided by the actions and
        returns a dictionary containing the same information for all agents:

            - The number of excess local requests,
            - The number of rejected forwarded requests,
            - The queue status before and after processing the forwarded
              requests from the other agent.
        """
        assert len(action) == self.agents, f"Expected {self.agents} entries, found {len(action)}"
        local_0, forward_0, reject_0 = action["node_0"]
        local_1, forward_1, reject_1 = action["node_1"]

        info = {agent: {} for agent in self.agent_ids}

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

        # Fill both local queues with local requests first.
        info["node_0"]["local_excess"] = fill_queue("node_0", local_0)
        info["node_1"]["local_excess"] = fill_queue("node_1", local_1)
        info["node_0"]["queue_status_pre_forward"] = self.queue["node_0"]
        info["node_1"]["queue_status_pre_forward"] = self.queue["node_1"]

        # Try to fill the local queues with the forwarded requests provided by
        # the opposing agent.
        info["node_0"]["forward_reject"] = fill_queue("node_1", forward_0)
        info["node_1"]["forward_reject"] = fill_queue("node_0", forward_1)
        info["node_0"]["queue_status_post_forward"] = self.queue["node_0"]
        info["node_1"]["queue_status_post_forward"] = self.queue["node_1"]

        return info

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
        if self.last_info is not None:
            for agent in self.agent_ids:
                info[agent]["action"] = {
                        "local": self.last_info["action"][agent][0],
                        "forward": self.last_info["action"][agent][1],
                        "reject": self.last_info["action"][agent][2]
                        }
                info[agent]["excess"] = {
                        "local_excess": self.last_info["workload"][agent]["local_excess"],
                        "forward_reject": self.last_info["workload"][agent]["forward_reject"]
                        }
                info[agent]["queue_status"] = {
                    "pre_forward": self.last_info["workload"][agent]["queue_status_pre_forward"],
                    "post_forward": self.last_info["workload"][agent]["queue_status_post_forward"]
                    }
                info[agent]["reward"] = self.last_info["rewards"][agent]

        return info


# Register the environments with Ray so that they can be used automatically when
# creating experiments.
def register(env_class):
    register_env(env_class.__name__, lambda env_config: env_class(config=env_config))


register(DFaaS_ASYM)
register(DFaaS_ASYM_MULTIPLE_RATIO)
register(DFaaS)


class DFaaSCallbacks(DefaultCallbacks):
    """User defined callbacks for the DFaaS environment (both SYM and ASYM).

    See the Ray's API documentation for DefaultCallbacks, the custom class
    overrides (and uses) only a subset of callbacks and keyword arguments."""

    def on_episode_start(self, *, episode, base_env, **kwargs):
        """Callback run right after an episode has started.

        Only the episode and base_env keyword arguments are used, other
        arguments are ignored."""
        # Make sure this episode has just been started (only initial obs logged
        # so far).
        assert episode.length <= 0, f"'on_episode_start()' callback should be called right after env reset! {episode.length = }"

        env = base_env.envs[0]

        # Save environment seed directly in hist_data.
        episode.hist_data["seed"] = [env.seed]

        # Common keys with one entry for each agent in each step.
        keys = ["observation_queue_capacity", "observation_input_requests",
                "action_local", "action_forward", "action_reject",
                "excess_local", "excess_forward_reject",
                "queue_status_pre_forward", "queue_status_post_forward",
                "reward"]

        # Initialize the dictionaries and lists.
        for key in keys:
            episode.user_data[key] = {agent: [] for agent in env.agent_ids}

        # The way to get the info data is complicated because of the Ray API.
        # However, we need to save the first observation because it contains the
        # initial data.
        info = env._additional_info()

        # Track common info for all agents.
        for agent in env.agent_ids:
            # Note that each element is a np.ndarray of size 1. It must be
            # unwrapped!
            episode.user_data["observation_queue_capacity"][agent].append(info[agent]["observation"]["queue_capacity"].item())
            episode.user_data["observation_input_requests"][agent].append(info[agent]["observation"]["input_requests"].item())

            # Do not track forward rejects as they are already stored in the
            # info dictionary at each step. This value is only important for the
            # current step, because it shows how many forwarded requests were
            # rejected by the other agent in the last step, not in the future!

    def on_episode_step(self, *, episode, base_env, **kwargs):
        """Called on each episode step (after the action has been logged).

        Only the episode and base_env keyword arguments are used, other
        arguments are ignored"""
        # Make sure this episode is ongoing.
        assert episode.length > 0, f"'on_episode_step()' callback should not be called right after env reset! {episode.length = }"

        env = base_env.envs[0]

        info = env._additional_info()

        # Track common info for all agents.
        for agent in env.agent_ids:
            episode.user_data["action_local"][agent].append(info[agent]["action"]["local"])
            episode.user_data["action_reject"][agent].append(info[agent]["action"]["reject"])
            episode.user_data["excess_local"][agent].append(info[agent]["excess"]["local_excess"])
            episode.user_data["reward"][agent].append(info[agent]["reward"])

        # Track forwarded requests only for node_0.
        episode.user_data["action_forward"]["node_0"].append(info["node_0"]["action"]["forward"])
        episode.user_data["excess_forward_reject"]["node_0"].append(info["node_0"]["excess"]["forward_reject"])

        # Track queue status only for node_1.
        episode.user_data["queue_status_pre_forward"]["node_1"].append(info["node_1"]["queue_status"]["pre_forward"])
        episode.user_data["queue_status_post_forward"]["node_1"].append(info["node_1"]["queue_status"]["post_forward"])

        # If it is the last step, skip the observation because it will not be
        # paired with the next action.
        if env.current_step < env.max_steps:
            for agent in env.agent_ids:
                episode.user_data["observation_queue_capacity"][agent].append(info[agent]["observation"]["queue_capacity"].item())
                episode.user_data["observation_input_requests"][agent].append(info[agent]["observation"]["input_requests"].item())

    def on_episode_end(self, *, episode, **kwargs):
        """Called when an episode is done (after terminated/truncated have been
        logged).

        Only the episode keyword arguments is used, other arguments are
        ignored."""
        # Note that this has to be a list of length 1 because there can be
        # multiple episodes in a single iteration, so at the end Ray will append
        # the list to a general list for the iteration.
        for key in episode.user_data.keys():
            episode.hist_data[key] = [episode.user_data[key]]

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Called at the end of Algorithm.train()."""
        # Final checker to verify the callbacks are executed.
        result["callbacks_ok"] = True

        # The problem here is that Ray cumulates the values of the keys under
        # hist_stats across iterations, but I do not want this behavior.
        # Solution: keep only the values generated by episodes in this
        # iteration.
        episodes = result["episodes_this_iter"]
        for key in result["hist_stats"]:
            result["hist_stats"][key] = result["hist_stats"][key][-episodes:]

        # Because they are repeated by Ray within the result dictionary.
        del result["sampler_results"]
