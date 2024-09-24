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
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Initialize logger for this module.
logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s", level=logging.DEBUG)
_logger = logging.getLogger(Path(__file__).name)


class DFaaS_ASYM(MultiAgentEnv):
    # String representing the type of this environment.
    type = "ASYM"

    # Keys contained in the additional_info dictionary.
    info_keys = {"observation_input_requests": np.int32,
                 "observation_queue_capacity": np.int32,
                 "action_local": np.int32,
                 "action_forward": np.int32,
                 "action_reject": np.int32,
                 "excess_local": np.int32,
                 "excess_forward_reject": np.int32,
                 "queue_status_pre_forward": np.int32,
                 "queue_status_post_forward": np.int32,
                 "reward": np.float32}

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
                "input_requests": gym.spaces.Box(low=0, high=150, dtype=np.int32),

                # Queue capacity (currently a constant).
                "queue_capacity": gym.spaces.Box(low=0, high=self.queue_capacity_max["node_0"], dtype=np.int32),

                # Forwarded requests in the previous step.
                "last_forward_requests": gym.spaces.Box(low=0, high=150, dtype=np.int32),

                # Forwarded but rejected requests in the previous step. Note
                # that last_forward_rejects <= last_forward_requests.
                "last_forward_rejects": gym.spaces.Box(low=0, high=150, dtype=np.int32)
                 }),

            "node_1": gym.spaces.Dict({
                # Number of input requests to process for a single step.
                "input_requests": gym.spaces.Box(low=50, high=150, dtype=np.int32),

                # Queue capacity (currently a constant).
                "queue_capacity": gym.spaces.Box(low=0, high=self.queue_capacity_max["node_1"], dtype=np.int32)
                })
            })

        # Number of steps in the environment. The default is one step for every
        # 5 minutes of a 24-hour day.
        self.max_steps = config.get("max_steps", 288)

        # Type of input requests.
        self.input_requests_type = config.get("input_requests_type", "synthetic")
        match self.input_requests_type:
            case "synthetic":
                pass
            case "real":
                assert self.max_steps == 288, f"With {self.input_requests_type = } only 288 max_steps are supported"
            case _:
                assert False, f"Unsupported {self.input_requests_type = }"

        # Is the env created for evaluation only? If so, the input requests may
        # differ from the training ones.
        self.evaluation = config.get("evaluation", False)

        super().__init__()

    def get_config(self):
        """Returns a dictionary with the current configuration of the
        environment."""
        config = {}
        config["queue_capacity_max_node_0"] = self.queue_capacity_max["node_0"]
        config["queue_capacity_max_node_1"] = self.queue_capacity_max["node_1"]
        config["max_steps"] = self.max_steps
        config["input_requests_type"] = self.input_requests_type
        return config

    def reset(self, *, seed=None, options=None):
        # Current step.
        self.current_step = 0

        # If seed is given, overwrite the master seed. Ray will give the seed in
        # reset() only when it creates the environment for each rollout worker
        # (and local worker). Each worker has a specific seed.
        if seed is not None:
            # The master seed for the RNG. This is used in each episode (each
            # "reset()") to create a new RNG that will be used for generating
            # input requests.
            #
            # Using the master seed make sure to generate a reproducible
            # sequence of seeds.
            self.master_seed = seed
            self.master_rng = np.random.default_rng(seed=self.master_seed)

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
        limits = {}
        for agent in self.agent_ids:
            limits[agent] = {
                    "min": self.observation_space[agent]["input_requests"].low.item(),
                    "max": self.observation_space[agent]["input_requests"].high.item()
                    }
        if self.input_requests_type == "synthetic":
            self.input_requests = _synthetic_input_requests(self.max_steps,
                                                            self.agent_ids,
                                                            limits,
                                                            self.rng)
        else:  # "real"
            retval = _real_input_requests(self.max_steps, self.agent_ids,
                                          limits, self.rng, self.evaluation)

            self.input_requests = retval[0]

            # Special attribute, not returned in the observation: contains the
            # hashes of the selected input requests. It is used by the
            # callbacks.
            self.input_requests_hashes = retval[1]

        # Queue state for each agent (number of requests to process locally).
        # The queues start empty (max capacity) and can be full.
        self.queue = {agent: 0 for agent in self.agent_ids}

        self.last_info = None  # Required by _build_observation().
        obs = self._build_observation()

        # For each reset() and step() call, the info dictionary is stored in an
        # attribute and is not returned. The caller can access this attribute
        # directly (usually at the end of the episode).
        #
        # To update the dictionary, a private function is called at the end of
        # reset() and call().
        self.additional_info = None
        self._additional_info(obs)

        return obs, {}

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
                                                        (info_work["node_0"]["local_excess"], info_work["node_0"]["forward_rejects"]),
                                                        (self.queue["node_0"], self.queue_capacity_max["node_0"]))
        rewards["node_1"] = self._calculate_reward_1_v2(action_1,
                                                        (info_work["node_1"]["local_excess"],),
                                                        (info_work["node_1"]["queue_status_pre_forward"], self.queue_capacity_max["node_1"]))

        # Make sure the reward is of type float.
        for agent in self.agent_ids:
            rewards[agent] = float(rewards[agent])

        # Required by _build_observation().
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

        # Update the additional_info dictionary.
        self._additional_info(obs, self.last_info["action"], rewards, info_work)

        return obs, rewards, terminated, truncated, {}

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

        # Set the ratio of forwarded but rejected requests.
        if self.last_info is None:
            last_forward_reqs = last_forward_rejects = 0
        else:
            last_forward_reqs = self.last_info["action"]["node_0"][1]
            last_forward_rejects = self.last_info["workload"]["node_0"]["forward_rejects"]

        obs["node_0"]["last_forward_requests"] = np.array([last_forward_reqs], dtype=np.int32)
        obs["node_0"]["last_forward_rejects"] = np.array([last_forward_rejects], dtype=np.int32)

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

        # There is no incoming forwarded requests for node_0, so save the queue
        # status with the same value (to be coherent with the symmetric
        # version).
        info["node_0"]["queue_status_pre_forward"] = self.queue["node_0"]
        info["node_0"]["queue_status_post_forward"] = self.queue["node_0"]

        # Local processing for node_1. Input requests for node_1 have the
        # priority over the forwarded requests from node_0.
        info["node_1"]["local_excess"] = fill_queue("node_1", local_1)
        info["node_1"]["queue_status_pre_forward"] = self.queue["node_1"]

        # Handle now forwarded requests.
        info["node_0"]["forward_rejects"] = fill_queue("node_1", forward_0)
        info["node_1"]["queue_status_post_forward"] = self.queue["node_1"]

        return info

    def _additional_info(self, obs, action=None, rewards=None, info_work=None):
        """Update the additional_info dictionary with the current step."""
        # Initialize the additional_info dictionary with all the NumPy arrays.
        if self.additional_info is None:
            self.additional_info = {}
            for key in self.info_keys:
                self.additional_info[key] = {}
                for agent in self.agent_ids:
                    self.additional_info[key][agent] = np.empty(self.max_steps, dtype=self.info_keys[key])

        # Update the additional_info dictionary.
        for agent in self.agent_ids:
            # In the last step, do not write the observation out of bounds.
            if self.current_step < self.max_steps:
                self.additional_info["observation_input_requests"][agent][self.current_step] = obs[agent]["input_requests"]
                self.additional_info["observation_queue_capacity"][agent][self.current_step] = obs[agent]["queue_capacity"]

            if self.current_step == 0:
                # After reset() there is no action, reward and info_work.
                continue

            # These values refer to the previous step, so there is -1.
            self.additional_info["action_local"][agent][self.current_step-1] = action[agent][0]
            if agent == "node_0":
                self.additional_info["action_forward"][agent][self.current_step-1] = action[agent][1]
                self.additional_info["action_reject"][agent][self.current_step-1] = action[agent][2]
            else:  # node_1 cannot forward.
                self.additional_info["action_forward"][agent][self.current_step-1] = 0
                self.additional_info["action_reject"][agent][self.current_step-1] = action[agent][1]

            self.additional_info["excess_local"][agent][self.current_step-1] = info_work[agent]["local_excess"]
            if agent == "node_0":
                self.additional_info["excess_forward_reject"][agent][self.current_step-1] = info_work[agent]["forward_rejects"]
            else:
                self.additional_info["excess_forward_reject"][agent][self.current_step-1] = 0

            self.additional_info["queue_status_pre_forward"][agent][self.current_step-1] = info_work[agent]["queue_status_pre_forward"]
            self.additional_info["queue_status_post_forward"][agent][self.current_step-1] = info_work[agent]["queue_status_post_forward"]

            self.additional_info["reward"][agent][self.current_step-1] = rewards[agent]


class DFaaS(MultiAgentEnv):
    # String representing the type of this environment.
    type = "SYM"

    # Keys contained in the additional_info dictionary.
    info_keys = {"observation_input_requests": np.int32,
                 "observation_queue_capacity": np.int32,
                 "action_local": np.int32,
                 "action_forward": np.int32,
                 "action_reject": np.int32,
                 "excess_local": np.int32,
                 "excess_forward_reject": np.int32,
                 "queue_status_pre_forward": np.int32,
                 "queue_status_post_forward": np.int32,
                 "reward": np.float32}

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
                "input_requests": gym.spaces.Box(low=0, high=150, dtype=np.int32),

                # Queue capacity (currently a constant).
                "queue_capacity": gym.spaces.Box(low=0, high=self.queue_capacity_max["node_0"], dtype=np.int32),

                # Forwarded requests in the previous step.
                "last_forward_requests": gym.spaces.Box(low=0, high=150, dtype=np.int32),

                # Forwarded but rejected requests in the previous step. Note
                # that last_forward_rejects <= last_forward_requests.
                "last_forward_rejects": gym.spaces.Box(low=0, high=150, dtype=np.int32)
                }) for agent in self.agent_ids
            })

        # Number of steps in the environment. The default is one step for every
        # 5 minutes of a 24-hour day.
        self.max_steps = config.get("max_steps", 288)

        # Type of input requests.
        self.input_requests_type = config.get("input_requests_type", "synthetic")
        match self.input_requests_type:
            case "synthetic":
                pass
            case "real":
                assert self.max_steps == 288, f"With {self.input_requests_type = } only 288 max_steps are supported"
            case _:
                assert False, f"Unsupported {self.input_requests_type = }"

        # Is the env created for evaluation only? If so, the input requests may
        # differ from the training ones.
        self.evaluation = config.get("evaluation", False)

        super().__init__()

    def get_config(self):
        """Returns a dictionary with the current configuration of the
        environment."""
        config = {}
        config["queue_capacity_max_node_0"] = self.queue_capacity_max["node_0"]
        config["queue_capacity_max_node_1"] = self.queue_capacity_max["node_1"]
        config["max_steps"] = self.max_steps
        config["input_requests_type"] = self.input_requests_type
        return config

    def reset(self, *, seed=None, options=None):
        # Current step.
        self.current_step = 0

        # If seed is given, overwrite the master seed. Ray will give the seed in
        # reset() only when it creates the environment for each rollout worker
        # (and local worker). Each worker has a specific seed.
        if seed is not None:
            # The master seed for the RNG. This is used in each episode (each
            # "reset()") to create a new RNG that will be used for generating
            # input requests.
            #
            # Using the master seed make sure to generate a reproducible
            # sequence of seeds.
            self.master_seed = seed
            self.master_rng = np.random.default_rng(seed=self.master_seed)

        # At least one time the reset() method must be called with a seed.
        assert getattr(self, "master_seed", None) is not None

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
        limits = {}
        for agent in self.agent_ids:
            limits[agent] = {
                    "min": self.observation_space[agent]["input_requests"].low.item(),
                    "max": self.observation_space[agent]["input_requests"].high.item()
                    }
        if self.input_requests_type == "synthetic":
            self.input_requests = _synthetic_input_requests(self.max_steps,
                                                            self.agent_ids,
                                                            limits,
                                                            self.rng)
        else:  # "real"
            retval = _real_input_requests(self.max_steps, self.agent_ids,
                                          limits, self.rng, self.evaluation)

            self.input_requests = retval[0]

            # Special attribute, not returned in the observation: contains the
            # hashes of the selected input requests. It is used by the
            # callbacks.
            self.input_requests_hashes = retval[1]

        # Queue state for each agent (number of requests to process locally).
        # The queues start empty (max capacity) and can be full.
        self.queue = {agent: 0 for agent in self.agent_ids}

        self.last_info = None  # Required by _build_observation().
        obs = self._build_observation()

        # For each reset() and step() call, the info dictionary is stored in an
        # attribute and is not returned. The caller can access this attribute
        # directly (usually at the end of the episode).
        #
        # To update the dictionary, a private function is called at the end of
        # reset() and call().
        self.additional_info = None
        self._additional_info(obs)

        return obs, {}

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
                                                       (info_work[agent]["local_excess"], info_work[agent]["forward_rejects"]),
                                                       (info_work[agent]["queue_status_pre_forward"], self.queue_capacity_max[agent]))
            # Make sure the reward is of type float.
            rewards[agent] = float(rewards[agent])

        # Required by _build_observation().
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

        # Update the additional_info dictionary.
        self._additional_info(obs, action, rewards, info_work)

        return obs, rewards, terminated, truncated, {}

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

            # Set the forwarded and forwarded but rejected requests from the
            # previous step.
            if self.last_info is None:
                last_forward_reqs = last_forward_rejects = 0
            else:
                last_forward_reqs = self.last_info["action"][agent][1]
                last_forward_rejects = self.last_info["workload"][agent]["forward_rejects"]

            obs[agent]["last_forward_requests"] = np.array([last_forward_reqs], dtype=np.int32)
            obs[agent]["last_forward_rejects"] = np.array([last_forward_rejects], dtype=np.int32)

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
        info["node_0"]["forward_rejects"] = fill_queue("node_1", forward_0)
        info["node_1"]["forward_rejects"] = fill_queue("node_0", forward_1)
        info["node_0"]["queue_status_post_forward"] = self.queue["node_0"]
        info["node_1"]["queue_status_post_forward"] = self.queue["node_1"]

        return info

    def _additional_info(self, obs, action=None, rewards=None, info_work=None):
        """Update the additional_info dictionary with the current step."""
        # Initialize the additional_info dictionary with all the NumPy arrays.
        if self.additional_info is None:
            self.additional_info = {}
            for key in self.info_keys:
                self.additional_info[key] = {}
                for agent in self.agent_ids:
                    self.additional_info[key][agent] = np.empty(self.max_steps, dtype=self.info_keys[key])

        # Update the additional_info dictionary.
        for agent in self.agent_ids:
            # In the last step, do not write the observation out of bounds.
            if self.current_step < self.max_steps:
                self.additional_info["observation_input_requests"][agent][self.current_step] = obs[agent]["input_requests"]
                self.additional_info["observation_queue_capacity"][agent][self.current_step] = obs[agent]["queue_capacity"]

            if self.current_step == 0:
                # After reset() there is no action, reward and info_work.
                continue

            # These values refer to the previous step, so there is -1.
            self.additional_info["action_local"][agent][self.current_step-1] = action[agent][0]
            self.additional_info["action_forward"][agent][self.current_step-1] = action[agent][1]
            self.additional_info["action_reject"][agent][self.current_step-1] = action[agent][2]

            self.additional_info["excess_local"][agent][self.current_step-1] = info_work[agent]["local_excess"]
            self.additional_info["excess_forward_reject"][agent][self.current_step-1] = info_work[agent]["forward_rejects"]

            self.additional_info["queue_status_pre_forward"][agent][self.current_step-1] = info_work[agent]["queue_status_pre_forward"]
            self.additional_info["queue_status_post_forward"][agent][self.current_step-1] = info_work[agent]["queue_status_post_forward"]

            self.additional_info["reward"][agent][self.current_step-1] = rewards[agent]


def _synthetic_input_requests(max_steps, agent_ids, limits, rng):
    """Generates the input requests for the given agents with the given length,
    clipping the values within the given bounds and using the given rng to
    generate the synthesized data.

    limits must be a dictionary whose keys are the agent ids, and each agent has
    two sub-keys: "min" for the minimum value and "max" for the maximum value.

    Returns a dictionary whose keys are the agent IDs and whose value is an
    np.ndarray containing the input requests for each step."""
    # These two values are calculated to match the average mean of the real
    # traces.
    average_requests = 50
    amplitude_requests = 100
    noise_ratio = .1
    unique_periods = 3  # The periods changes 3 times for each episode.

    input_requests = {}
    steps = np.arange(max_steps)
    for agent in agent_ids:
        # Note: with default max_stes, the period changes every 96 steps
        # (max_steps = 288). We first generate the periods and expand the array
        # to match the max_steps. If max_steps is not a multiple of 96, some
        # elements must be appended at the end, hence the resize call.
        repeats = max_steps // unique_periods
        periods = rng.uniform(15, high=100, size=unique_periods)
        periods = np.repeat(periods, repeats)  # Expand the single values.
        periods = np.resize(periods, periods.size + max_steps - periods.size)

        base_input = average_requests + amplitude_requests * np.sin(2 * np.pi * steps / periods)
        noisy_input = base_input + noise_ratio * rng.normal(0, amplitude_requests, size=max_steps)
        requests = np.asarray(noisy_input, dtype=np.int32)

        # Clip the excess values respecting the minimum and maximum values
        # for the input requests observation.
        min = limits[agent]["min"]
        max = limits[agent]["max"]
        np.clip(requests, min, max, out=requests)

        input_requests[agent] = requests

    return input_requests


# This list contains all fourteen real input request dataset files as Pandas
# DataFrame. Each DataFrame has a special attribute "idx", a string that
# indicates which file was read.
_real_input_requests_pool = None


def _init_real_input_requests_pool():
    """Initializes the _real_input_requests_pool module variable and reads the
    record files from a known path. It stops the application if a dataset file
    cannot be found."""
    # Generates the file names.
    total_datasets = 14
    datasets = []
    for idx in range(1, total_datasets+1):
        item = (f"d{idx:02}", f"invocations_per_function_md.anon.http.scaled.selected.d{idx:02}.csv")
        datasets.append(item)

    # Read each CSV file as a data frame.
    pool = []
    for (idx, dataset) in datasets:
        # Prefer absolute paths.
        path = (Path.cwd() / "dataset" / "data" / dataset).resolve()
        if not path.exists():
            _logger.critical(f"Dataset file not found: {path.as_posix()!r}")
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        frame.idx = idx  # Special metadata to know the original file.

        pool.append(frame)

    global _real_input_requests_pool
    _real_input_requests_pool = np.array(pool, dtype=object)


def _real_input_requests(max_steps, agent_ids, limits, rng, evaluation):
    """Randomly selects a real input request from the pool for each of the given
    agents.

    Since the steps and values of the real input requests are fixed, if the
    given values don't respect the fixed values, there will be an assertion
    error.

    limits must be a dictionary whose keys are the agent ids, and each agent has
    two sub-keys: "min" for the minimum value and "max" for the maximum value.

    The boolean evaluation parameter can be used to select the subpool from
    which input requests are selected. Note that the evaluation pool is smaller
    than the training pool.

    Returns a tuple: the first element is a dictionary whose keys are the agent
    ids and whose value is a NumPy array containing the input requests for each
    step, the second element is a dictionary whose keys are the agent ids and
    whose value is the hash string of the selected function from the pool."""
    if _real_input_requests_pool is None:
        _init_real_input_requests_pool()

    # Separate the evaluation pool (two dataframes) from the training pool.
    if evaluation:
        pool = _real_input_requests_pool[-2:]
    else:
        pool = _real_input_requests_pool[:-2]

    # Randomly select a dataframe for each agent. Note: It is important to
    # avoid choosing the same dataframe to avoid correlations between functions
    # in one day.
    dataframes = rng.choice(pool, size=len(agent_ids), replace=False)
    agents = list(agent_ids)  # Make a copy because it will be modified.
    functions = {}
    for dataframe in dataframes:
        row = dataframe.sample(random_state=rng)
        functions[agents.pop()] = {"dataframe": row, "idx": dataframe.idx}

    # Extract the input requests and function hashes from the dataframe.
    input_requests, hashes = {}, {}
    for agent in agent_ids:
        dataframe = functions[agent]["dataframe"]

        # The new hash is the concatenation of the function hash and the day
        # (from 01 to 14).
        hash = f"{dataframe['HashFunction'].item()}-{functions[agent]['idx']}"

        # Get only the columns related to the input requests and convert to a
        # NumPy array.
        reqs = dataframe.loc[:, "0":].to_numpy(dtype=np.int32).flatten()

        # Do some sanity checks to avoid nasty bugs.
        assert reqs.size == max_steps, f"Unsupported given max_steps = {max_steps}"
        min = limits[agent]["min"]
        max = limits[agent]["max"]
        assert np.all(reqs >= min) and np.all(reqs <= max), f"Unsupported limits: {limits[agent]}"

        input_requests[agent] = reqs
        hashes[agent] = hash

    return input_requests, hashes


# Register the environments with Ray so that they can be used automatically when
# creating experiments.
def register(env_class):
    register_env(env_class.__name__, lambda env_config: env_class(config=env_config))


register(DFaaS_ASYM)
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

        # If the environment has real input requests, we need to store the
        # hashes of all the requests used in the episode (one for each agent) in
        # order to identify the individual requests.
        if env.input_requests_type == "real":
            episode.hist_data["hashes"] = [env.input_requests_hashes]

    def on_episode_end(self, *, episode, base_env, **kwargs):
        """Called when an episode is done (after terminated/truncated have been
        logged).

        Only the episode and base_env keyword arguments are used, other
        arguments are ignored."""
        env = base_env.envs[0]
        info = env.additional_info

        # Note that this has to be a list of length 1 because there can be
        # multiple episodes in a single iteration, so at the end Ray will append
        # the list to a general list for the iteration.
        for key in env.info_keys:
            episode.hist_data[key] = [info[key]]

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
