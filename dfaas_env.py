"""This module contains the DFaaS multi-agent environment, a reinforcement
learning environment to train agents using different RL (and not) algorithms. It
also includes functions or methods strictly related to the environment, like the
associated callbacks."""

import gymnasium as gym
import logging
from pathlib import Path
from collections import deque, namedtuple, defaultdict

import pandas as pd
import numpy as np
import networkx as nx

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SAC

import perfmodel
import dfaas_input_rate

# Initialize logger for this module.
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.DEBUG,
)
_logger = logging.getLogger(Path(__file__).name)


def reward_fn(action, additional_reject):
    """Returns the reward for the given action and additional rejects. The
    reward is in the range [-1, 1]."""
    assert len(action) == 3, "Expected (local, forward, reject)"
    assert len(additional_reject) == 2, "Expected (local_reject, forward_reject)"

    arrival_rate_total = sum(action)
    rate_local, rate_forward, rate_reject = action
    local_reject, forward_reject = additional_reject

    local_reward = rate_local - local_reject
    forward_reward = rate_forward - forward_reject
    reward = local_reward + forward_reward - rate_reject

    # Normalize the reward around [-1, 1].
    assert arrival_rate_total > 0
    norm_reward = reward / arrival_rate_total

    return float(norm_reward)


class DFaaS(MultiAgentEnv):
    """DFaaS multi-agent reinforcement learning environment.

    The constructor accepts a config dictionary with the environment
    configuration."""

    def __init__(self, config={}):
        # By default, the network has two interconnected agents.
        #
        # The graph is represented by NetworkX's adjacency lists. Each line is a
        # list of node labels, where the first label is the source node and the
        # following labels are the destination nodes.
        network_config = config.get("network", ["node_0 node_1"])

        # Freeze to prevent further modification of the network nodes and edges.
        self.network = nx.freeze(nx.parse_adjlist(network_config))

        # IDs of the agents in the DFaaS environment.
        self.agents = list(self.network.nodes)

        # It is the possible max and min value for the reward returned by the
        # step() call.
        self.reward_range = (-1.0, 1.0)

        # Provide full (preferred format) observation and action spaces as
        # Dicts mapping agent IDs to the individual agents' spaces.

        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict(
            {
                # Distribution of how many requests are processed locally,
                # forwarded and rejected.
                #
                # Important: forwarding requests are evenly distributed among
                # the node's neighbours.
                agent: Simplex(shape=(3,))
                for agent in self.agents
            }
        )

        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict(
            {
                # Only the first observation refers to the next step, the others
                # refers to the previous step (historical data).
                agent: gym.spaces.Dict(
                    {
                        # Arrival rate of input requests per second to process
                        # for a single step.
                        # TODO: Change the name.
                        "input_requests": gym.spaces.Box(low=1, high=150, dtype=np.int32),
                        # Incoming local requests in the previous step.
                        "prev_local_requests": gym.spaces.Box(low=0, high=150, dtype=np.float32),
                        # Incoming local requests but rejected in the previous
                        # step. Note prev_local_rejects < prev_local_requests.
                        "prev_local_rejects": gym.spaces.Box(low=0, high=150, dtype=np.float32),
                        # Forwarded requests in the previosu step.
                        "prev_forward_requests": gym.spaces.Box(
                            low=0,
                            high=150,
                            dtype=np.float32,
                        ),
                        # Forwarded requests but rejected requests in the
                        # previous step.
                        # Note prev_forward_rejects <= prev_forward_requests.
                        "prev_forward_rejects": gym.spaces.Box(low=0, high=150, dtype=np.float32),
                    }
                )
                for agent in self.agents
            }
        )

        # Save the action range that is based on the minimium and maximium
        # possibile value of input requests for all agents.
        #
        # Note: This assumes that all agents have the same input request range.
        input_requests_space = self.observation_space["node_0"]["input_requests"]
        self.action_range = (input_requests_space.low.item(), input_requests_space.high.item())

        # Number of steps in the environment. The default is one step for every
        # 5 minutes of a 24-hour day.
        self.max_steps = config.get("max_steps", 288)

        # Generation method of input rate.
        self.input_rate_same_method = config.get("input_rate_same_method", True)
        if self.input_rate_same_method:
            self.input_rate_method = config.get("input_rate_method", "synthetic-sinusoidal")
            match self.input_rate_method:
                case "synthetic-sinusoidal":
                    pass
                case "synthetic-normal":
                    pass
                case "real":
                    assert self.max_steps == 288, f"With {self.input_rate_method = } only 288 max_steps are supported"
                case _:
                    assert False, f"Unsupported {self.input_rate_method = }"
        else:
            assert False, f"Unsupported {self.input_rate_same_method = }"

        # Is the env created for evaluation only? If so, the input requests may
        # differ from the training ones.
        self.evaluation = config.get("evaluation", False)

        super().__init__()

    def get_config(self):
        """Returns a dictionary with the current configuration of the
        environment."""
        config = {
            "max_steps": self.max_steps,
            "input_rate_same_method": self.input_rate_same_method,
            "input_rate_method": self.input_rate_method,
            "evaluation": self.evaluation,
            "network": list(nx.generate_adjlist(self.network)),
        }

        return config

    def reset(self, *, seed=None, options=None):
        # Current step.
        self.current_step = 0

        # If seed is given, overwrite the master seed. Ray will give the seed in
        # reset() only when it creates the environment for each runner (and
        # local runner). Each runner has a specific seed.
        if seed is not None:
            # The master seed for the RNG. This is used in each episode (each
            # "reset()") to create a new RNG that will be used for generating
            # input requests.
            #
            # Using the master seed make sure to generate a reproducible
            # sequence of seeds.
            self.master_seed = seed
            self.master_rng = np.random.default_rng(seed=self.master_seed)

        assert getattr(self, "master_seed", None) is not None, "reset() must be called the first time with a seed"

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

        # Generate all input rates for the environment.
        limits = {}
        for agent in self.agents:
            limits[agent] = {
                "min": self.observation_space[agent]["input_requests"].low.item(),
                "max": self.observation_space[agent]["input_requests"].high.item(),
            }
        match self.input_rate_method:
            case "synthetic-sinusoidal":
                self.input_requests = dfaas_input_rate.synthetic_sinusoidal(
                    self.max_steps, self.agents, limits, self.rng
                )
                pass
            case "synthetic-normal":
                self.input_requests = dfaas_input_rate.synthetic_normal(self.max_steps, self.agents, limits, self.rng)
                pass
            case "real":
                retval = dfaas_input_rate.real(self.max_steps, self.agents, limits, self.rng, self.evaluation)

                self.input_requests = retval[0]

                # Special attribute, not returned in the observation: contains
                # the hashes of the selected input requests. It is used by the
                # callbacks.
                self.input_requests_hashes = retval[1]
            case _:
                assert False, f"Unreachable code"

        def info_init_key():
            """Helper function to automatically initialize the keys in the info
            dictionary."""
            value = {}
            for agent in self.agents:
                value[agent] = [0 for _ in range(self.max_steps)]

            return value

        # This dictionary contains the entire history of observations, rewards,
        # metrics, and logs for a single episode. It is updated step by step,
        # each value being a list with one entry for each step. It can be
        # accessed at any time, usually at the end of the episode.
        #
        # It is of type defaultdict to allow missing keys to be initialized
        # automatically.
        self.info = defaultdict(info_init_key)

        obs = self._build_observation()

        return obs, {}

    def step(self, action_dict):
        """Given an action for each agent, executes a step.

        This method overrides the parent class method, see MultiAgentEnv for
        more information."""
        # 1. Convert the action distribution to arrival rate for each action.
        action = {}
        for agent in self.agents:
            # Get total arrival rate.
            arrival_rate = self.input_requests[agent][self.current_step]

            # Convert the action distribution (a distribution of probabilities)
            # into the number of requests to locally process, to forward and to
            # reject.
            action[agent] = _convert_arrival_rate_dist(arrival_rate, action_dict[agent])

            # Log the arrival rate for each action.
            self.info["action_local"][agent][self.current_step] = action[agent][0]
            self.info["action_forward"][agent][self.current_step] = action[agent][1]
            self.info["action_reject"][agent][self.current_step] = action[agent][2]

        # 2. Manage the workload.
        self._manage_workload(action)

        # 3. Calculate the reward for both agents.
        rewards = {}
        for agent in self.agents:
            # Extract some additional info that is used to calculate the reward.
            additional_rejects = (
                self.info["incoming_rate_local_reject"][agent][self.current_step],
                self.info["forward_reject_rate"][agent][self.current_step],
            )

            reward = reward_fn(action[agent], additional_rejects)
            assert isinstance(reward, float), f"Unsupported reward type {type(reward)}"

            rewards[agent] = reward
            self.info["reward"][agent][self.current_step] = reward

        # Go to the next step.
        self.current_step += 1

        # 4. Update environment state.
        if self.current_step < self.max_steps:
            obs = self._build_observation()
        else:
            # Return a dummy observation because this is the last step.
            obs = self.observation_space.sample()

        # 5. Prepare return values.

        # Each key in the dictionary indicates whether an individual agent has
        # terminated. There is a special key "__all__" which is true only if all
        # agents have terminated.
        terminated = {agent: False for agent in self.agents}
        if self.current_step == self.max_steps:
            # We are past the last step: nothing more to do.
            terminated = {agent: True for agent in self.agents}
        terminated["__all__"] = all(terminated.values())

        # Truncated is always set to False because it is not used.
        truncated = {agent: False for agent in self.agents}
        truncated["__all__"] = False

        # Postprocess the info dictionary by converting all NumPy arrays to
        # Python types. Note that this is done on the current and previous step
        # to avoid copying this code into the reset() method.
        #
        # TODO: This is not very efficient, but it works.
        for key in self.info:
            for agent in self.agents:
                # Warning: this is also executed at the end of the episode!
                if self.current_step < self.max_steps:
                    value = self.info[key][agent][self.current_step]
                    if isinstance(value, (np.ndarray, np.float64)):
                        self.info[key][agent][self.current_step] = value.item()

                value = self.info[key][agent][self.current_step - 1]
                if isinstance(value, (np.ndarray, np.float64)):
                    self.info[key][agent][self.current_step - 1] = value.item()

        return obs, rewards, terminated, truncated, {}

    def _build_observation(self):
        """Builds and returns the observation for the current step."""
        assert self.current_step < self.max_steps

        # Initialize the observation dictionary.
        obs = {agent: {} for agent in self.agents}

        def update_info(obs):
            """Helper function that populates the info dictionary for the
            current step only for the observation keys."""
            for agent in self.agents:
                for key in obs[agent].keys():
                    info_key = "observation_" + key

                    for agent in self.agents:
                        self.info[info_key][agent][self.current_step] = obs[agent][key]

        # Special case: there is no data from the previous step at the start.
        if self.current_step == 0:
            for agent in self.agents:
                input_requests = self.input_requests[agent][self.current_step]
                obs[agent] = {
                    "input_requests": np.array([input_requests], dtype=np.float32),
                    "prev_local_requests": np.array([0], dtype=np.float32),
                    "prev_local_rejects": np.array([0], dtype=np.float32),
                    "prev_forward_requests": np.array([0], dtype=np.float32),
                    "prev_forward_rejects": np.array([0], dtype=np.float32),
                }

            update_info(obs)

            return obs

        # Normal case.
        for agent in self.agents:
            input_requests = self.input_requests[agent][self.current_step]
            prev_local_reqs = self.info["action_local"][agent][self.current_step - 1]
            prev_local_rejects = self.info["incoming_rate_local_reject"][agent][self.current_step - 1]
            prev_forward_reqs = self.info["action_forward"][agent][self.current_step - 1]
            prev_forward_rejects = self.info["forward_reject_rate"][agent][self.current_step - 1]

            obs[agent]["input_requests"] = np.array([input_requests], dtype=np.float32)

            obs[agent]["prev_local_requests"] = np.array([prev_local_reqs], dtype=np.float32)
            obs[agent]["prev_local_rejects"] = np.array([prev_local_rejects], dtype=np.float32)

            obs[agent]["prev_forward_requests"] = np.array([prev_forward_reqs], dtype=np.float32)
            obs[agent]["prev_forward_rejects"] = np.array([prev_forward_rejects], dtype=np.float32)

        update_info(obs)
        return obs

    def _manage_workload(self, action):
        """Simulate one step of the workload management for all agents.

        At the end, updates the self.info dictionary with all information on the
        simulation step."""
        assert len(action) == len(self.agents), f"Expected {len(self.agents)} entries, found {len(action)}"

        warm_service_time = 15
        cold_service_time = 30
        idle_time_before_kill = 600

        # Extract arrival rate for each action.
        local = {agent: action[agent][0] for agent in self.agents}
        forward = {agent: action[agent][1] for agent in self.agents}
        reject = {agent: action[agent][2] for agent in self.agents}

        # Rate of incoming requests for each agent and action. The first is
        # always the local requests.
        incoming_rate = {agent: [] for agent in self.agents}

        # Total rate of incoming requests for each agent (local + forwarded).
        incoming_rate_total = {agent: 0 for agent in self.agents}

        # Label (ID) of each agent in the corresponding incoming_reqs key.
        incoming_rate_agents = {agent: [] for agent in self.agents}

        # Before calling the pacsltk's function, collect the total rate of
        # requests for each agent.
        for agent in self.agents:
            incoming_rate[agent].append(local[agent])
            incoming_rate_agents[agent].append(agent)

            for neighbor in self.network.neighbors(agent):
                # The rate of forwarded requests are distributed equally to all
                # neighbouring nodes, so I need to calculate the share for the
                # current agent.
                forwarded_reqs = forward[neighbor] / self.network.degree(neighbor)
                incoming_rate[agent].append(forwarded_reqs)
                incoming_rate_agents[agent].append(neighbor)

            incoming_rate_total[agent] = sum(incoming_rate[agent])

            self.info["incoming_rate"][agent][self.current_step] = incoming_rate_total[agent]
            self.info["incoming_rate_local"][agent][self.current_step] = incoming_rate[agent][0]
            self.info["incoming_rate_forward"][agent][self.current_step] = sum(incoming_rate[agent][1:])

        # Then call the pacsltk's function for each agent.
        for agent in self.agents:
            if incoming_rate_total[agent] == 0:
                # Skip this agent since there a no requests to handle.
                continue

            result_props, _ = perfmodel.get_sls_warm_count_dist(
                incoming_rate_total[agent],
                warm_service_time,
                cold_service_time,
                idle_time_before_kill,
            )
            rejection_rate = result_props["rejection_rate"]
            self.info["incoming_rate_reject"][agent][self.current_step] = rejection_rate

            # Distribute the rejection rate to all agents (itself and its
            # neighbors), proportionally to the incoming rate for each agent.
            for idx in range(len(incoming_rate[agent])):
                # Portions of incoming rate.
                incoming_dist_agent = incoming_rate_agents[agent][idx]
                incoming_dist_rate = incoming_rate[agent][idx]

                reject = round(rejection_rate * incoming_dist_rate / incoming_rate_total[agent])

                # Current agent (local incoming rate).
                if incoming_dist_agent == agent:
                    self.info["incoming_rate_local_reject"][agent][self.current_step] = reject
                else:  # Neighbor agent (forwarded requests).
                    neighbor = incoming_dist_agent

                    # Update the forwarded reject rate for the neighbor and the
                    # incoming forward reject rate for the receiving agent.
                    self.info["forward_reject_rate"][neighbor][self.current_step] += reject
                    self.info["incoming_rate_forward_reject"][agent][self.current_step] += reject


def _convert_arrival_rate_dist(arrival_rate, action_dist):
    """Distribute the arrival rate to the given action distribution.

    Returns a tuple of three elements: the arrival rate of local requests,
    forwarded and rejected.

    This function expects an action distribution (e.g. [.7, .2, .1])."""
    assert len(action_dist) == 3, "Expected (local, forward, reject)"

    # Extract the three actions from the action distribution.
    prob_local, prob_forward, prob_reject = action_dist

    # Get the corresponding arrival rate for each action.
    rate_local = arrival_rate * prob_local
    rate_forward = arrival_rate * prob_forward
    rate_reject = arrival_rate * prob_reject

    return tuple([rate_local, rate_forward, rate_reject])


# Register the environments with Ray so that they can be used automatically when
# creating experiments.
def register(env_class):
    register_env(env_class.__name__, lambda env_config: env_class(config=env_config))


register(DFaaS)


class DFaaSCallbacks(DefaultCallbacks):
    """User defined callbacks for the DFaaS environment.

    These callbacks can be used with other environments, both multi-agent and
    single-agent.

    See the Ray's API documentation for DefaultCallbacks, the custom class
    overrides (and uses) only a subset of callbacks and keyword arguments."""

    def on_episode_start(self, *, episode, base_env, **kwargs):
        """Callback run right after an episode has started.

        Only the episode and base_env keyword arguments are used, other
        arguments are ignored."""
        # Make sure this episode has just been started (only initial obs logged
        # so far).
        assert (
            episode.length <= 0
        ), f"'on_episode_start()' callback should be called right after env reset! {episode.length = }"

        try:
            env = base_env.envs[0]
        except AttributeError:
            # With single-agent environment the wrapper env is an instance of
            # VectorEnvWrapper and it doesn't have envs attribute. With
            # multi-agent the wrapper is MultiAgentEnvWrapper.
            env = base_env.get_sub_environments()[0]

        # Save environment seed directly in hist_data.
        episode.hist_data["seed"] = [env.seed]

        # If the environment has real input requests, we need to store the
        # hashes of all the requests used in the episode (one for each agent) in
        # order to identify the individual requests.
        if env.input_rate_same_method and env.input_rate_method == "real":
            episode.hist_data["hashes"] = [env.input_requests_hashes]

    def on_episode_end(self, *, episode, base_env, **kwargs):
        """Called when an episode is done (after terminated/truncated have been
        logged).

        Only the episode and base_env keyword arguments are used, other
        arguments are ignored."""
        try:
            env = base_env.envs[0]
        except AttributeError:
            env = base_env.get_sub_environments()[0]

        assert (
            env.current_step == env.max_steps
        ), f"'on_episode_end()' callback should be called at the end of the episode! {env.current_step = } != {env.max_steps = }"

        for key in env.info.keys():
            # Note that this has to be a list of length 1 because there can be
            # multiple episodes in a single iteration, so at the end Ray will
            # append the list to a general list for the iteration.
            episode.hist_data[key] = [env.info[key]]

    def on_evaluate_end(self, *, algorithm, evaluation_metrics, **kwargs):
        """Called at the end of Algorithm.evaluate()."""
        evaluation_metrics["callbacks_ok"] = True

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Called at the end of Algorithm.train()."""
        if algorithm.__class__ == SAC:
            # Only for SAC: log also the status of the replay buffer.
            result["info"]["replay_buffer"] = {}
            result["info"]["replay_buffer"]["capacity_per_policy"] = algorithm.local_replay_buffer.capacity
            result["info"]["replay_buffer"].update(algorithm.local_replay_buffer.stats())

        result["callbacks_ok"] = True


def _run_one_episode(verbose=False):
    """Run a test episode of the DFaaS environment."""
    config = {"network": ["node_0 node_1", "node_1"]}
    # config = {"network": ["node_0 node_1 node_2", "node_3 node_2 node_0", "node_1 node_4"]}
    env = DFaaS(config=config)
    _ = env.reset(seed=42)

    if verbose:
        from tqdm import trange

        range = trange

    for step in range(env.max_steps):
        env.step(action_dict=env.action_space.sample())


if __name__ == "__main__":
    _run_one_episode()
