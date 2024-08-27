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
                "queue_capacity": gym.spaces.Box(low=0, high=self.max_requests_step, dtype=np.int32),

                # Forwarding capacity (currently a constant).
                "forward_capacity": gym.spaces.Box(low=0, high=self.max_requests_step, dtype=np.int32)
                 }),

            "node_1": gym.spaces.Dict({
                # Number of input requests to process for a single step.
                "input_requests": gym.spaces.Box(low=50, high=150, dtype=np.int32),

                # Queue capacity (currently a constant).
                "queue_capacity": gym.spaces.Box(low=0, high=self.max_requests_step, dtype=np.int32)
                })
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
        self.last_obs = None

        obs = self._build_observation()
        self.last_obs = obs
        info = self._additional_info()

        return obs, info

    def step(self, action_dict=None):
        action_agent = self.agent_ids[self.turn]

        assert action_dict is not None, "action_dict is required"
        action_dist = action_dict.get(action_agent)
        assert action_dist is not None, f"Expected agent {action_agent!r} but {action_dict = }"

        # We need to process the action differently between agents because each
        # agent has a different action space.
        if action_agent == "node_0":
            assert len(action_dist) == 3, "Expected (local, forward, reject)"

            input_requests = self.input_requests[action_agent][self.current_step]

            # Convert the action distribution (a distribution of probabilities)
            # into the number of requests to locally process, to forward and to
            # reject.
            reqs_local, reqs_forward, reqs_reject = self._convert_distribution_0(input_requests, action_dist)
            self.last_action = {"local": reqs_local,
                                "forward": reqs_forward,
                                "reject": reqs_reject}

            forward_capacity = self.last_obs[action_agent]["forward_capacity"]

            # Calculate the reward.
            reward = self._calculate_reward_0(reqs_local, reqs_forward, reqs_reject, forward_capacity)
        else:  # node_1
            assert len(action_dist) == 2, "Expected (local, reject)"

            # Get the input requests from the latest observation because node_0
            # may have forwarded some requests to it.
            input_requests = self.last_obs[action_agent]["input_requests"]

            # Convert the action distribution (a distribution of probabilities)
            # into the number of requests to locally process and reject.
            reqs_local, reqs_reject = self._convert_distribution(input_requests, action_dist)
            self.last_action = {"local": reqs_local, "reject": reqs_reject}

            # Calculate the reward.
            reward = self._calculate_reward(reqs_local, reqs_reject)

        # Make sure the reward is of type float.
        reward = float(reward)
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

        # Reward for the agent.
        rewards = {action_agent: reward}

        if self.current_step < self.node_max_steps:
            obs = self._build_observation()
        else:
            # Return a dummy observation because this is the last step.
            obs = self.observation_space_sample([next_agent])
        self.last_obs = obs

        # Create the additional information dictionary.
        info = self._additional_info()

        return obs, rewards, terminated, truncated, info

    def _build_observation(self):
        """Returns the observation dictionary for the current agent for the
        current step."""
        agent = self.agent_ids[self.turn]

        # Initialize the observation dictionary for the current agent.
        obs = {}
        obs[agent] = {}

        # The queue capacity is always a fixed value for now.
        obs[agent]["queue_capacity"] = np.array([self.max_requests_step], dtype=np.int32)

        # node_0 also has the forwarding capacity observation.
        if agent == "node_0":
            # The forwarding capacity depends on the input requests of the
            # node_1. The value is non-negative, because if it is zero node_0
            # can't forward any requests to node_1.
            input_requests_1 = self.input_requests["node_1"][self.current_step]
            forward_capacity = self.max_requests_step - input_requests_1
            if forward_capacity < 0:
                forward_capacity = 0
            obs[agent]["forward_capacity"] = np.array([forward_capacity], dtype=np.int32)

        input_requests = self.input_requests[agent][self.current_step]
        obs[agent]["input_requests"] = np.array([input_requests], dtype=np.int32)
        if agent == "node_1":
            # node_1 has increased input requests because node_0 may have
            # forwarded some requests to this node.
            obs[agent]["input_requests"] += self.last_action["forward"]

            # node_0 may have forwarded more requests than the forwarding
            # capacity. The environment penalized this with the reward, but in
            # this case we also need to clip the update value to not exceed the
            # higher limit of input requests.
            top = self.observation_space[agent]["input_requests"].high[0]
            if obs[agent]["input_requests"] > top:
                obs[agent]["input_requests"] = np.array([top], dtype=np.int32)

        return obs

    def _calculate_reward(self, reqs_local, reqs_reject):
        """Returns the reward for the given action (the number of locally
        processed requests and rejected requests). The reward is a number in the
        range 0 to 1.

        This function is only for node_1 agent."""
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

    def _calculate_reward_0(self, reqs_local, reqs_forward, reqs_reject, forwarding_capacity):
        """Returns the reward for the given action (the number of locally
        processed requests, forwarded requests and rejected requests). The
        reward is a number in the range 0 to 1.

        This function is only for node_0 agent."""
        reqs_total = reqs_local + reqs_forward + reqs_reject

        # The agent (policy) tried to be sneaky, but it is not possible to
        # locally process more requests than the internal limit for each step.
        # This behavior must be discouraged by penalizing the reward, but not as
        # much as by rejecting too many requests (the .5 factor).
        if reqs_local > self.max_requests_step:
            reqs_local_exceed = reqs_local - self.max_requests_step
            return 1 - (reqs_local_exceed / 100) * .5

        # The same also for forwarding.
        if reqs_forward > forwarding_capacity:
            reqs_forward_exceed = reqs_forward - forwarding_capacity
            return 1 - (reqs_forward_exceed / 100) * .5

        # If there are more requests than the node can handle locally, the
        # optimal strategy should be to process all possible requests locally
        # and forward or reject the extra ones.
        if reqs_total > self.max_requests_step:
            # The reward penalises the agent if the action doesn't maximise the
            # request process locally.
            if reqs_local < self.max_requests_step:
                # The new value is the number of rejected requests that will be
                # considered a penalty for the reward. Note that some rejections
                # are inevitable and will not be penalized, only those that can
                # be processed locally but the agent didn't.
                reqs_reject = self.max_requests_step - reqs_local - reqs_forward
                reqs_reject = np.clip(reqs_reject, a_min=0, a_max=None)
                reqs_total = self.max_requests_step + reqs_forward
            elif reqs_forward < forwarding_capacity:
                reqs_reject = reqs_reject - (forwarding_capacity - reqs_forward)
                reqs_reject = np.clip(reqs_reject, a_min=0, a_max=None)
                reqs_total = self.max_requests_step + reqs_forward
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
            info["__common__"]["turn"] = agent
            info["__common__"][agent] = {
                    "obs": self.last_obs,
                    "current_step": self.current_step
                    }

            # node_1 has input requests altered in the observation because
            # node_0 can forward requests to it. So we need to track also the
            # original input requests.
            if agent == "node_1":
                input_reqs = self.input_requests[agent][self.current_step]
                info["__common__"][agent]["original_input_requests"] = input_reqs

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
