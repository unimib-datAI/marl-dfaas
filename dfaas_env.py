"""This module contains the DFaaS multi-agent environment, a reinforcement
learning environment to train agents using different RL (and not) algorithms. It
also includes functions or methods strictly related to the environment, like the
associated callbacks."""

import gymnasium as gym
import logging
from pathlib import Path

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


def _total_network_delay(access_delay_ms, data_size_bytes, bandwidth_mbps):
    """
    Calculate total network delay (in milliseconds).

    Parameters:
        access_delay_ms (float): Access delay in milliseconds
        data_size_bytes (float): Size of data to send in bytes
        bandwidth_mbps (float): Bandwidth in megabits per second (Mbps)

    Returns:
        float: Total delay in milliseconds
    """
    data_size_bits = data_size_bytes * 8  # From bytes to bits.
    bandwidth_bps = bandwidth_mbps * 1_000_000  # From Mbps to bps.

    # Transmission delay in seconds.
    transmission_delay_s = data_size_bits / bandwidth_bps

    # Convert to milliseconds.
    transmission_delay_ms = transmission_delay_s * 1000

    # Total delay in milliseconds.
    total_delay_ms = access_delay_ms + transmission_delay_ms
    return total_delay_ms


def reward_fn(action, additional_reject):
    """Returns the reward for the given action and additional rejects. The
    reward is in the range [-1, 1]."""
    # Extract local and reject actions from the action tuple.
    local_action = action[0]
    reject_action = action[-1]

    # Sum up all forwarding actions (could be multiple neighbors).
    forward_action = sum(action[1:-1])

    if len(additional_reject) != 2:
        raise ValueError("Expected additional_reject = (local_reject, forward_reject)")

    arrival_rate_total = sum(action)
    local_reject, forward_reject = additional_reject

    local_reward = local_action - local_reject
    forward_reward = forward_action - forward_reject
    reward = local_reward + forward_reward - reject_action

    # Normalize the reward around [-1, 1].
    if arrival_rate_total <= 0:
        raise ValueError("The sum of actions must be positive!")
    norm_reward = reward / arrival_rate_total

    return float(norm_reward)


class DFaaS(MultiAgentEnv):
    """DFaaS multi-agent reinforcement learning environment.

    The constructor accepts a config dictionary with the environment
    configuration.

    Metrics can be accessed from the self.info dictionary. This dictionary
    contains an entry for each agent, and for every agent, a list of metrics
    recorded at each step. Note that some metrics are redundant (starting with
    "observation_prev"). The default callback associated with this environment
    automatically skips these redundant metrics.
    """

    def __init__(self, config={}):
        # By default, the network has two interconnected agents.
        #
        # The graph is represented by NetworkX's adjacency lists. Each line is a
        # list of node labels, where the first label is the source node and the
        # following labels are the destination nodes.
        network_config = config.get("network", ["node_0 node_1"])

        self.network = nx.parse_adjlist(network_config)

        # Network link properties (delay, bandwidth).
        #
        # The configuration expects a dictionary with keys (source_node,
        # dest_node), and as value a dictionary with custom properties (keys
        # "access_delay_ms" and "bandwidth_mbps").
        network_links = config.get("network_links", {})
        for (u, v), params in network_links.items():
            if self.network.has_edge(u, v):
                nx.set_edge_attributes(self.network, {(u, v): params})
                # Assume bidirectional link
                if self.network.has_edge(v, u):
                    nx.set_edge_attributes(self.network, {(v, u): params})

        # Set default link parameters if not specified.
        default_link_params = {"access_delay_ms": 10, "bandwidth_mbps": 10}
        for u, v in self.network.edges():
            if "access_delay_ms" not in self.network[u][v]:
                nx.set_edge_attributes(self.network, {(u, v): default_link_params})

        # Freeze to prevent further modification of the network nodes and edges.
        self.network = nx.freeze(self.network)

        # Average data size in bytes for a single function execution.
        #
        # Currently it is a constant, because we only have a single type of
        # function.
        self.data_size_bytes = config.get("data_size_bytes", 2_000_000)  # 2MB

        # IDs of the agents in the DFaaS environment.
        self.agents = list(self.network.nodes)

        # It is the possible max and min value for the reward returned by the
        # step() call.
        self.reward_range = (-1.0, 1.0)

        # Store the neighbor list for each agent for easy access.
        self.agent_neighbors = {agent: list(self.network.neighbors(agent)) for agent in self.agents}

        # Provide full (preferred format) observation and action spaces as
        # Dicts mapping agent IDs to the individual agents' spaces.

        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict(
            {
                # Distribution of how many requests are processed locally,
                # forwarded to specific neighbors, and rejected.
                #
                # For each agent, the action space size is:
                # 1 (local) + number_of_neighbors (forwarding) + 1 (reject)
                agent: Simplex(shape=(1 + len(self.agent_neighbors[agent]) + 1,))
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
                        "input_rate": gym.spaces.Box(low=1, high=150, dtype=np.int32),
                        # Arrival rate of the previous step.
                        "prev_input_rate": gym.spaces.Box(low=1, high=150, dtype=np.int32),
                        # How much rate has been forwarded and how much of this
                        # has been rejected in the previous step, for each
                        # neighbor (a node may have different neighbors).
                        # We unpack the temporary dictionaries (**).
                        **{
                            f"prev_forward_to_{neighbor}": gym.spaces.Box(low=0, high=150, dtype=np.int32)
                            for neighbor in self.agent_neighbors[agent]
                        },
                        **{
                            f"prev_forward_rejects_to_{neighbor}": gym.spaces.Box(low=0, high=150, dtype=np.int32)
                            for neighbor in self.agent_neighbors[agent]
                        },
                    }
                )
                for agent in self.agents
            }
        )

        # Save the action range that is based on the minimium and maximium
        # possibile value of input rate for all agents.
        #
        # Note: This assumes that all agents have the same input rate range.
        input_rate_space = self.observation_space["node_0"]["input_rate"]
        self.action_range = (input_rate_space.low.item(), input_rate_space.high.item())

        # Number of steps in the environment. The default is one step for every
        # 5 minutes of a 24-hour day.
        self.max_steps = config.get("max_steps", 288)

        # All nodes receive their input rates from the same generation method by
        # default.
        self.input_rate_same_method = config.get("input_rate_same_method", True)

        # Verify that the specified generation method exists. Otherwise, an
        # exception will be raised.
        self.input_rate_method = config.get("input_rate_method", "synthetic-sinusoidal")
        dfaas_input_rate.generator(self.input_rate_method)

        # Is the env created for evaluation only? If so, the input requests may
        # differ from the training ones.
        self.evaluation = config.get("evaluation", False)

        # Perfmodel parameters for each node.
        self.perfmodel_params = {}
        default_params = (15, 30, 600)  # warm_service_time, cold_service_time, idle_time_before_kill
        node_params = config.get("perfmodel_params", {})
        for agent in self.agents:
            # Set parameters for each node, or the defaults if not given.
            params = node_params.get(agent, default_params)
            if not isinstance(params, (list, tuple)) or len(params) != 3:
                raise ValueError(f"perfmodel_params for {agent} must be a tuple or list of 3 elements")

            self.perfmodel_params[agent] = {
                "warm_service_time": params[0],
                "cold_service_time": params[1],
                "idle_time_before_kill": params[2],
            }

        # Initialize the info dictionary with node as first level keys and
        # metrics as second level.
        self.info = {}
        for agent in self.agents:
            self.info[agent] = self._init_agent_metrics()

        super().__init__()

    def _init_agent_metrics(self):
        """Initialize all metrics for a single agent with zero values for each step."""
        # List of some metrics to track for each agent. Additional metrics are
        # defined below.
        metrics = [
            "action_local",
            "action_forward",
            "action_reject",
            "incoming_rate",
            "incoming_rate_local",
            "incoming_rate_forward",
            "incoming_rate_reject",
            "incoming_rate_local_reject",
            "incoming_rate_forward_reject",
            "forward_reject_rate",
            "reward",
            "response_time_avg",
            "network_delay_avg",
        ]

        # Create a zero-filled array for each metric.
        result = {metric: [0 for _ in range(self.max_steps)] for metric in metrics}

        # Add specific metrics for forward actions and rejects to each neighbor.
        for agent, neighbors in self.agent_neighbors.items():
            for neighbor in neighbors:
                result[f"action_forward_to_{neighbor}"] = [0 for _ in range(self.max_steps)]
                result[f"forward_rejects_to_{neighbor}"] = [0 for _ in range(self.max_steps)]
                result[f"network_delay_avg_to_{neighbor}"] = [0 for _ in range(self.max_steps)]

        return result

    def get_config(self):
        """Returns a dictionary with the current configuration of the
        environment."""
        config = {
            "max_steps": self.max_steps,
            "input_rate_same_method": self.input_rate_same_method,
            "input_rate_method": self.input_rate_method,
            "evaluation": self.evaluation,
            "network": list(nx.generate_adjlist(self.network)),
            "data_size_bytes": self.data_size_bytes,
            "network_links": {
                (u, v): {
                    "access_delay_ms": self.network[u][v]["access_delay_ms"],
                    "bandwidth_mbps": self.network[u][v]["bandwidth_mbps"],
                }
                for u, v in self.network.edges()
                if u < v  # Avoid duplicates for bidirectional links.
            },
            "perfmodel_params": {
                agent: (
                    params["warm_service_time"],
                    params["cold_service_time"],
                    params["idle_time_before_kill"],
                )
                for agent, params in self.perfmodel_params.items()
            },
        }

        return config

    def reset(self, *, seed=None, options=None):
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

        if self.evaluation:
            # See self.set_master_seed() method.
            self.seed = self.eval_seeds[self.eval_seeds_index]
            self.eval_seeds_index = (self.eval_seeds_index + 1) % len(self.eval_seeds)

        # Create the RNG used to generate input requests.
        self.rng = np.random.default_rng(seed=self.seed)
        self.np_random = self.rng  # Required by the Gymnasium API

        # Generate all input rates for the environment.
        generator = dfaas_input_rate.generator(self.input_rate_method)
        match self.input_rate_method:  # Each generator has its own signature.
            case "synthetic-sinusoidal" | "synthetic-normal":
                input_rate = generator(self.max_steps, self.agents, self.rng)

                # Scale down the traces to ensure that the input rate does not
                # exceed the total capacity in a single step.
                self.input_rate = dfaas_input_rate.scale_down(input_rate)
            case "synthetic-constant" | "synthetic-step-change" | "synthetic-linear-growth":
                # It's not necessary to scale down since the generator already
                # do that.
                self.input_rate = generator(self.max_steps, self.agents)
            case "synthetic-double-linear-growth":
                # The max_per_agent argument is taken from perfmodel: with the
                # current input values a node starts to drop requests from about
                # 63 input requests per step.
                #
                # It's not necessary to scale down since the generator already
                # do that.
                self.input_rate = dfaas_input_rate.synthetic_double_linear_growth(
                    self.max_steps, self.agents, max_per_agent=63, rng=self.rng
                )
            case "real":
                limits = {}
                for agent in self.agents:
                    limits[agent] = {
                        "min": self.observation_space[agent]["input_rate"].low.item(),
                        "max": self.observation_space[agent]["input_rate"].high.item(),
                    }

                retval = dfaas_input_rate.real(self.max_steps, self.agents, limits, self.rng, self.evaluation)

                self.input_rate = retval[0]

                # Special attribute, not returned in the observation: contains
                # the hashes of the selected input requests. It is used by the
                # callbacks.
                self.input_rate_hashes = retval[1]
            case _:
                raise AssertionError("Unreachable code")

        # Reset the info dictionary
        for agent in self.agents:
            self.info[agent] = self._init_agent_metrics()

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
            arrival_rate = self.input_rate[agent][self.current_step]

            # Convert the action distribution (a distribution of probabilities)
            # into the number of requests to locally process, to forward to specific neighbors, and to reject.
            action[agent] = _convert_arrival_rate_dist(arrival_rate, action_dict[agent])

            # Log the arrival rate for each action.
            self.info[agent]["action_local"][self.current_step] = action[agent][0]

            # Log forward actions for each neighbor
            for i, neighbor in enumerate(self.agent_neighbors[agent]):
                forward_key = f"action_forward_to_{neighbor}"
                self.info[agent][forward_key][self.current_step] = action[agent][i + 1]

            # Total forwarded requests (sum of all neighbor forwards)
            self.info[agent]["action_forward"][self.current_step] = sum(action[agent][1:-1])

            # Rejected requests
            self.info[agent]["action_reject"][self.current_step] = action[agent][-1]

        # 2. Manage the workload.
        self._manage_workload(action)

        # 3. Calculate the reward for both agents.
        rewards = {}
        for agent in self.agents:
            # Extract some additional info that is used to calculate the reward.
            additional_rejects = (
                self.info[agent]["incoming_rate_local_reject"][self.current_step],
                self.info[agent]["forward_reject_rate"][self.current_step],
            )

            reward = reward_fn(action[agent], additional_rejects)
            assert isinstance(reward, float), f"Unsupported reward type {type(reward)}"

            rewards[agent] = reward
            self.info[agent]["reward"][self.current_step] = reward

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
        for agent in self.agents:
            for key in self.info[agent]:
                # Warning: this is also executed at the end of the episode!
                if self.current_step < self.max_steps:
                    value = self.info[agent][key][self.current_step]
                    if isinstance(value, (np.ndarray, np.float64)):
                        self.info[agent][key][self.current_step] = value.item()

                value = self.info[agent][key][self.current_step - 1]
                if isinstance(value, (np.ndarray, np.float64)):
                    self.info[agent][key][self.current_step - 1] = value.item()

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

                    # Create the metric if it doesn't exist yet
                    if info_key not in self.info[agent]:
                        self.info[agent][info_key] = [0 for _ in range(self.max_steps)]

                    self.info[agent][info_key][self.current_step] = obs[agent][key]

        # Special case: there is no data from the previous step at the start.
        if self.current_step == 0:
            for agent in self.agents:
                input_rate = self.input_rate[agent][self.current_step]
                obs[agent] = {
                    "input_rate": np.array([input_rate], dtype=np.float32),
                    "prev_input_rate": np.array([1], dtype=np.float32),
                }

                # Add zero entries for each neighbor
                for neighbor in self.agent_neighbors[agent]:
                    obs[agent][f"prev_forward_to_{neighbor}"] = np.array([0], dtype=np.float32)
                    obs[agent][f"prev_forward_rejects_to_{neighbor}"] = np.array([0], dtype=np.float32)

            update_info(obs)

            return obs

        # Normal case.
        for agent in self.agents:
            input_rate = self.input_rate[agent][self.current_step]
            prev_input_rate = self.input_rate[agent][self.current_step - 1]

            obs[agent]["input_rate"] = np.array([input_rate], dtype=np.int32)
            obs[agent]["prev_input_rate"] = np.array([prev_input_rate], dtype=np.int32)

            # Add specific values for each neighbor
            for neighbor in self.agent_neighbors[agent]:
                # Previous forward requests to this neighbor
                prev_forward_key = f"action_forward_to_{neighbor}"
                prev_forward_reqs = self.info[agent][prev_forward_key][self.current_step - 1]
                obs[agent][f"prev_forward_to_{neighbor}"] = np.array([prev_forward_reqs], dtype=np.int32)

                # Previous forward rejects to this neighbor
                prev_rejects_key = f"forward_rejects_to_{neighbor}"
                prev_forward_rejects = (
                    self.info[agent][prev_rejects_key][self.current_step - 1]
                    if prev_rejects_key in self.info[agent]
                    else 0
                )
                obs[agent][f"prev_forward_rejects_to_{neighbor}"] = np.array([prev_forward_rejects], dtype=np.int32)

        update_info(obs)
        return obs

    def _manage_workload(self, action):
        """Simulate one step of the workload management for all agents.

        At the end, updates the self.info dictionary with all information on the
        simulation step."""
        assert len(action) == len(self.agents), f"Expected {len(self.agents)} entries, found {len(action)}"

        # Incoming rate for each agent. Each value is a list of incoming rates
        # for that agent.
        incoming_rate = {agent: [] for agent in self.agents}

        # The label (ID) of the agent for each incoming rate.
        #
        # E.g.: incoming_rate["node_0"] = [30, 10, 15]
        #       incoming_rate_agents["node_0"] = ["node_1", "node_0", "node_2"]
        incoming_rate_agents = {agent: [] for agent in self.agents}

        # Before calling the pacsltk's function, collect the total incoming rate
        # of requests for each agent.
        for agent in self.agents:
            # First element is local processing
            local_rate = action[agent][0]
            incoming_rate[agent].append(local_rate)
            incoming_rate_agents[agent].append(agent)
            self.info[agent]["incoming_rate_local"][self.current_step] = local_rate

        # Process the forwarded requests
        for agent in self.agents:
            neighbors = self.agent_neighbors[agent]

            # For each neighbor, forward the specific amount (no longer evenly distributed)
            for i, neighbor in enumerate(neighbors):
                # Index 1 to len(neighbors) are the forwarding actions
                forward_rate = action[agent][i + 1]

                if forward_rate > 0:
                    incoming_rate[neighbor].append(forward_rate)
                    incoming_rate_agents[neighbor].append(agent)
                    self.info[neighbor]["incoming_rate_forward"][self.current_step] += forward_rate

                    # Calculate network delay
                    link_params = self.network[agent][neighbor]
                    delay = _total_network_delay(
                        link_params["access_delay_ms"],
                        self.data_size_bytes,
                        link_params["bandwidth_mbps"],
                    )
                    delay_key = f"network_delay_avg_to_{neighbor}"
                    self.info[agent][delay_key][self.current_step] = delay

        # Then call the pacsltk's function for each agent.
        node_avg_resp_time = {}  # Store avg_resp_time for each node
        for agent in self.agents:
            incoming_rate_total = sum(incoming_rate[agent])
            self.info[agent]["incoming_rate"][self.current_step] = incoming_rate_total
            if incoming_rate_total == 0:
                node_avg_resp_time[agent] = 0
                # Skip this agent since there are no requests to handle.
                continue

            params = self.perfmodel_params[agent]
            result_props, _ = perfmodel.get_sls_warm_count_dist(
                incoming_rate_total,
                params["warm_service_time"],
                params["cold_service_time"],
                params["idle_time_before_kill"],
            )
            rejection_rate = result_props["rejection_rate"]
            node_avg_resp_time[agent] = result_props.get("avg_resp_time", 0)

            # Distribute the rejection rate to all agents (itself and its
            # neighbors).
            rejects = _distribute_rejects(rejection_rate, incoming_rate[agent])

            self.info[agent]["incoming_rate_reject"][self.current_step] = sum(rejects)
            for idx in range(len(incoming_rate[agent])):
                reject_share = rejects[idx]
                reject_agent = incoming_rate_agents[agent][idx]

                assert reject_share <= incoming_rate[agent][idx]

                # Current agent (local incoming rate).
                if reject_agent == agent:
                    self.info[agent]["incoming_rate_local_reject"][self.current_step] = reject_share
                else:  # Neighbor agent (forwarded requests).
                    sender = reject_agent

                    # Update the forwarded reject rate for the sending agent and track per-neighbor reject
                    self.info[sender]["forward_reject_rate"][self.current_step] += reject_share

                    # Update the per-neighbor forward rejects
                    reject_key = f"forward_rejects_to_{agent}"
                    if reject_key not in self.info[sender]:
                        self.info[sender][reject_key] = [0 for _ in range(self.max_steps)]
                    self.info[sender][reject_key][self.current_step] += reject_share

                    # Update the incoming forward reject rate for the receiving agent
                    self.info[agent]["incoming_rate_forward_reject"][self.current_step] += reject_share

        # Calculate response_time_avg for each link for each agent.
        for agent in self.agents:
            total_resp_time = 0
            total_reqs = 0
            delays_sum = 0

            # Local requests
            local_reqs = self.info[agent]["incoming_rate_local"][self.current_step]
            if local_reqs > 0:
                total_resp_time += node_avg_resp_time[agent] * local_reqs
                total_reqs += local_reqs

            # Forwarded requests originated from this agent
            for i, neighbor in enumerate(self.agent_neighbors[agent]):
                forwarded_reqs = action[agent][i + 1]
                if forwarded_reqs > 0:
                    delay_key = f"network_delay_avg_to_{neighbor}"
                    network_delay = self.info[agent][delay_key][self.current_step]
                    neighbor_resp_time = node_avg_resp_time[neighbor]
                    total_resp_time += (network_delay + neighbor_resp_time) * forwarded_reqs
                    total_reqs += forwarded_reqs

                    # Save for network_delay_avg across all links
                    delays_sum += network_delay

            if total_reqs > 0:
                avg_resp_time = total_resp_time / total_reqs
                self.info[agent]["response_time_avg"][self.current_step] = avg_resp_time

            if (num_neighbors := len(self.agent_neighbors[agent])) > 0:
                self.info[agent]["network_delay_avg"][self.current_step] = delays_sum / num_neighbors
            else:
                self.info[agent]["network_delay_avg"][self.current_step] = 0

    def set_master_seed(self, master_seed, episodes):
        """Set the master seed of the environment. This function is usually
        called once when creating the environment to ensure to use the same
        seeds for every evaluation iteration."""
        self.master_seed = master_seed
        self.master_rng = np.random.default_rng(seed=self.master_seed)
        iinfo = np.iinfo(np.uint32)
        self.eval_seeds = self.master_rng.integers(0, high=iinfo.max, size=episodes)
        self.eval_seeds_index = 0


def _distribute_rejects(reject_rate, incoming_rate):
    """Distributes a given number of rejects proportionally among agents based
    on their incoming rates.

    Args:
        - reject_rate (float or int): the total number of rejects to distribute.
        - incoming_rate (list of int): the list of incoming rates from each
          agent.

    Returns:
        - list of int: a list where each element represents the number of rejects
        allocated to each agent.
    """
    agents = len(incoming_rate)
    reject_rate = round(reject_rate)

    if reject_rate == 0:  # Special (and rare) case.
        return np.repeat(0, agents).tolist()

    # First, we need to convert the raw incoming rate, which is a list of
    # integers, into proportions of the incoming rate.
    incoming_rate = np.array(incoming_rate)
    proportions = incoming_rate / incoming_rate.sum()
    assert incoming_rate.sum() >= reject_rate

    # Distribute the rejects proportionally among all agents according to the
    # incoming rate. Since we want integers, we must account for the fractional
    # parts.
    raw_rates = proportions * reject_rate
    floored = np.floor(raw_rates).astype(int)
    remainder = int(round(reject_rate - floored.sum()))

    # Find the indices with the largest fractional parts and add one more rate
    # until the remainder is zero.
    fractional_parts = raw_rates - floored
    for _ in range(remainder):
        max_idx = np.argmax(fractional_parts)
        floored[max_idx] += 1
        fractional_parts[max_idx] = -1  # Dummy value to avoid picking this id again.
    assert floored.sum() == reject_rate

    return floored.tolist()


def _convert_arrival_rate_dist(arrival_rate, action_dist):
    """Distribute the arrival rate to the given action distribution.

    Returns a tuple with the arrival rate for each action:

    - First element: local processing rate
    - Middle elements: forwarding rates to specific neighbors
    - Last element: rejection rate

    All values are integers.

    This function expects an action distribution that sums to 1."""
    # The action space is now:
    # [local, forward_to_neighbor_1, forward_to_neighbor_2, ..., forward_to_neighbor_n, reject]

    # Calculate the raw rates based on the action distribution.
    raw_rates = [int(prob * arrival_rate) for prob in action_dist]

    # Calculate how many requests are currently assigned.
    total_assigned = sum(raw_rates)

    # Distribute the remaining requests due to integer rounding.
    remaining = int(arrival_rate - total_assigned)
    if remaining > 0:
        # Find the actions with the highest fractional parts.
        fractions = [(prob * arrival_rate) - raw_rates[i] for i, prob in enumerate(action_dist)]

        # Sort indices by fraction.
        sorted_indices = sorted(range(len(fractions)), key=lambda i: fractions[i], reverse=True)

        # Assign remaining requests to actions with highest fractional parts.
        for i in range(remaining):
            raw_rates[sorted_indices[i % len(sorted_indices)]] += 1

    assert sum(raw_rates) == arrival_rate
    return tuple(raw_rates)


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
        assert episode.length <= 0, (
            f"'on_episode_start()' callback should be called right after env reset! {episode.length = }"
        )

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
            episode.hist_data["hashes"] = [env.input_rate_hashes]

    def on_episode_end(self, *, episode, base_env, **kwargs):
        """Called when an episode is done (after terminated/truncated have been
        logged).

        Only the episode and base_env keyword arguments are used, other
        arguments are ignored."""
        try:
            env = base_env.envs[0]
        except AttributeError:
            env = base_env.get_sub_environments()[0]

        assert env.current_step == env.max_steps, (
            f"'on_episode_end()' callback should be called at the end of the episode! {env.current_step = } != {env.max_steps = }"
        )

        # Add all metrics from self.info to episode.hist_data
        for agent in env.agents:
            agent_data = env.info[agent]
            for metric, values in agent_data.items():
                # Skip metrics starting with "observation_prev", because they
                # are redundant.
                if metric.startswith("observation_prev"):
                    continue

                # Create the key in hist_data if it doesn't exist
                if metric not in episode.hist_data:
                    episode.hist_data[metric] = [{agent: values}]
                else:
                    # Add to existing key
                    episode.hist_data[metric][0][agent] = values

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


def _run_one_episode(verbose=False, config=None, seed=None):
    """Run a test episode of the DFaaS environment."""
    if config is None:
        # config = {"network": ["node_0 node_1", "node_1"]}
        config = {"network": ["node_0 node_1 node_2", "node_3 node_2 node_0", "node_1 node_4"]}
    if seed is None:
        seed = 42

    env = DFaaS(config=config)
    _ = env.reset(seed=seed)

    if verbose:
        from tqdm import trange

        loop = trange
    else:
        loop = range

    for step in loop(env.max_steps):
        env.step(action_dict=env.action_space.sample())


def _main():
    """Main entry point for running a single DFaaS episode with random
    actions."""
    # Import these modules only if this module is called as main script.
    import argparse
    import dfaas_utils

    desc = "Run a single DFaaS episode with random actions"

    parser = argparse.ArgumentParser(prog="dfaas_env", description=desc)

    parser.add_argument("--env-config", help="Override default environment configuration (TOML file)", type=Path)
    parser.add_argument("--seed", type=int, help="Override default seed of input rate generation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.env_config is not None:
        config = dfaas_utils.toml_to_dict(args.env_config)
    else:
        config = None

    _run_one_episode(args.verbose, config, args.seed)


if __name__ == "__main__":
    _main()
