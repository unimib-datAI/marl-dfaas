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

        # Maximum number of requests a node can handle in a single step.
        self.queue_capacity_max = 100

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
                "queue_capacity": gym.spaces.Box(low=0, high=self.queue_capacity_max, dtype=np.int32),

                # Forwarding capacity (currently a constant).
                "forward_capacity": gym.spaces.Box(low=0, high=self.queue_capacity_max, dtype=np.int32)
                 }),

            "node_1": gym.spaces.Dict({
                # Number of input requests to process for a single step.
                "input_requests": gym.spaces.Box(low=50, high=150, dtype=np.int32),

                # Queue capacity (currently a constant).
                "queue_capacity": gym.spaces.Box(low=0, high=self.queue_capacity_max, dtype=np.int32)
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

    def reset(self, *, seed=None, options=None):
        # Current step.
        self.current_step = 0

        # Seed used for this episode.
        iinfo = np.iinfo(np.uint32)
        self.seed = self.master_rng.integers(0, high=iinfo.max, size=1)[0]

        # Create the RNG used to generate input requests.
        self.rng = np.random.default_rng(seed=self.seed)
        self.np_random = self.rng  # Required by the Gymnasium API

        # Generate all input requests for the environment.
        self.input_requests = self._get_input_requests()

        # These values refer to the last action/reward performed in the latest
        # step() invocation. They are set to None because no actions and rewards
        # was logged in reset(). Used by the _additional_info method.
        self.last_actions = None
        self.last_rewards = None

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
        self.last_actions = {}
        self.last_actions["node_0"] = {"local": action_0[0],
                                       "forward": action_0[1],
                                       "reject": action_0[2]}

        forward_capacity = self.last_obs["node_0"]["forward_capacity"]

        # Action for node_1.

        # Get the input requests from the latest observation because node_0 may
        # have forwarded some requests to it.
        input_requests_1 = self.last_obs["node_1"]["input_requests"]

        # Convert the action distribution (a distribution of probabilities) into
        # the number of requests to locally process and reject.
        action_1 = self._convert_distribution(input_requests_1, action_dict["node_1"])
        self.last_actions["node_1"] = {"local": action_1[0], "reject": action_1[1]}

        # We have the actions, not update the environment state.
        # TODO

        # Calculate the reward for both agents.
        rewards = {}
        rewards["node_1"] = self._calculate_reward(action_1)
        rewards["node_0"] = self._calculate_reward_0(action_0, forward_capacity)

        # Make sure the reward is of type float.
        for agent in self.agent_ids:
            rewards[agent] = float(rewards[agent])
        self.last_rewards = rewards

        # Go to the next step.
        self.current_step += 1

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
            obs[agent]["queue_capacity"] = np.array([self.queue_capacity_max], dtype=np.int32)

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
        forward_capacity = self.queue_capacity_max - input_requests
        if forward_capacity < 0:
            forward_capacity = 0
        obs["node_0"]["forward_capacity"] = np.array([forward_capacity], dtype=np.int32)

        return obs

    def _calculate_reward(self, action):
        """Returns the reward for the given action (a tuple containing the
        number of requests to locally process and reject) in a range between 0
        and 1.

        This function is only for node_1 agent."""
        assert len(action) == 2, "Expected (local, reject)"

        reqs_total = sum(action)
        reqs_local, reqs_reject = action

        # The agent (policy) tried to be sneaky, but it is not possible to
        # locally process more requests than the internal limit for each step.
        # This behavior must be discouraged by penalizing the reward, but not as
        # much as by rejecting too many requests (the .5 factor).
        if reqs_local > self.queue_capacity_max:
            reqs_local_exceed = reqs_local - self.queue_capacity_max
            return 1 - (reqs_local_exceed / 100) * .5

        # If there are more requests than the node can handle locally, the
        # optimal strategy should be to process all possible requests locally
        # and reject the extra ones.
        if reqs_total > self.queue_capacity_max:
            # The reward penalises the agent if the action doesn't maximise the
            # request process locally.
            if reqs_local < self.queue_capacity_max:
                # The new value is the number of rejected requests that will be
                # considered a penalty for the reward. Note that some rejections
                # are inevitable and will not be penalized, only those that can
                # be processed locally but the agent didn't.
                reqs_reject = self.queue_capacity_max - reqs_local
                reqs_total = self.queue_capacity_max
            else:
                reqs_reject = 0

        # The reward is a range from 0 to 1. It decreases as the number of
        # unnecessary rejected requests increases.
        return 1 - reqs_reject / reqs_total

    def _calculate_reward_0(self, action, forwarding_capacity):
        """Returns the reward for the given action (a tuple containing the
        number of requests to process locally, to forward and to reject) in a
        range between 0 and 1.

        This function is only for node_0 agent."""
        assert len(action) == 3, "Expected (local, forward, reject)"

        reqs_total = sum(action)
        reqs_local, reqs_forward, reqs_reject = action

        # The agent (policy) tried to be sneaky, but it is not possible to
        # locally process more requests than the internal limit for each step.
        # This behavior must be discouraged by penalizing the reward, but not as
        # much as by rejecting too many requests (the .5 factor).
        if reqs_local > self.queue_capacity_max:
            reqs_local_exceed = reqs_local - self.queue_capacity_max
            return 1 - (reqs_local_exceed / 100) * .5

        # The same also for forwarding.
        if reqs_forward > forwarding_capacity:
            reqs_forward_exceed = reqs_forward - forwarding_capacity
            return 1 - (reqs_forward_exceed / 100) * .5

        # If there are more requests than the node can handle locally, the
        # optimal strategy should be to process all possible requests locally
        # and forward or reject the extra ones.
        if reqs_total > self.queue_capacity_max:
            # The reward penalises the agent if the action doesn't maximise the
            # request process locally.
            if reqs_local < self.queue_capacity_max:
                # The new value is the number of rejected requests that will be
                # considered a penalty for the reward. Note that some rejections
                # are inevitable and will not be penalized, only those that can
                # be processed locally but the agent didn't.
                reqs_reject = self.queue_capacity_max - reqs_local - reqs_forward
                reqs_reject = np.clip(reqs_reject, a_min=0, a_max=None)
                reqs_total = self.queue_capacity_max + reqs_forward
            elif reqs_forward < forwarding_capacity:
                reqs_reject = reqs_reject - (forwarding_capacity - reqs_forward)
                reqs_reject = np.clip(reqs_reject, a_min=0, a_max=None)
                reqs_total = self.queue_capacity_max + reqs_forward
            else:
                reqs_reject = 0

        # The reward is a range from 0 to 1. It decreases as the number of
        # unnecessary rejected requests increases.
        reward = 1 - reqs_reject / reqs_total
        assert 0.0 <= reward <= 1.0

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

    def _get_input_requests(self):
        """Calculate the input requests for all agents for all steps.

        Returns a dictionary whose keys are the agent IDs and whose value is an
        np.ndarray containing the input requests for each step."""
        average_requests = 100
        period = 50
        amplitude_requests = 50
        noise_ratio = .1

        input_requests = {}
        steps = np.arange(self.queue_capacity_max)
        for agent in self.agent_ids:
            # TODO: do not directly check the value of the agent ID.
            fn = np.sin if agent == "node_0" else np.cos

            base_input = average_requests + amplitude_requests * fn(2 * np.pi * steps / period)
            noisy_input = base_input + noise_ratio * self.rng.normal(0, amplitude_requests, size=self.queue_capacity_max)
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

        if self.current_step < self.max_steps:
            # node_1 has input requests altered in the observation because
            # node_0 can forward requests to it. So we need to track also the
            # original input requests.
            input_reqs = self.input_requests["node_1"][self.current_step]
        else:
            # This is the last step, the value won't be used.
            input_reqs = 0
        info["node_1"]["original_input_requests"] = input_reqs

        # Also save the actions and rewards from the last step.
        if self.last_actions is not None:
            assert self.last_rewards is not None

            for agent in self.agent_ids:
                info[agent]["action"] = self.last_actions[agent]
                info[agent]["reward"] = self.last_rewards[agent]

        return info


# Register the environment with Ray so that it can be used automatically when
# creating experiments.
register_env("DFaaS", lambda env_config: DFaaS(config=env_config))
