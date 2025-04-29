# This file contains the DFaaS multi-agent environment and associated callbacks.
import gymnasium as gym
import logging
from pathlib import Path
from collections import deque, namedtuple, defaultdict

import pandas as pd
import numpy as np
import networkx as nx
import pacsltk

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SAC

# Initialize logger for this module.
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.DEBUG,
)
_logger = logging.getLogger(Path(__file__).name)


def reward_fn(action, additional_reject):
    """Reward function for the agents in the DFaaS environment."""
    assert len(action) == 3, "Expected (local, forward, reject)"
    assert len(additional_reject) == 2, "Expected (local_reject, forward_reject)"

    arrival_rate_total = sum(action)
    rate_local, rate_forward, rate_reject = action
    local_reject, forward_reject = additional_reject

    local_reward = rate_local - local_reject
    forward_reward = rate_forward - forward_reject
    return float(local_reward + forward_reward - rate_reject)


class DFaaS(MultiAgentEnv):
    """DFaaS environment.

    The constructor accepts a config dictionary with the following keys (see the
    source code for defaults):

    - network: the graph structure of the DFaaS network, given as adjacency
      lists to be parsed with NetworkX.
    - max_steps
    - input_requests_type
    - evaluation
    """

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
        self.reward_range = (0.0, 1.0)

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
                        "input_requests": gym.spaces.Box(low=0, high=150, dtype=np.int32),
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

        # Number of steps in the environment. The default is one step for every
        # 5 minutes of a 24-hour day.
        self.max_steps = config.get("max_steps", 288)

        # Type of input requests.
        self.input_requests_type = config.get("input_requests_type", "synthetic-sinusoidal")
        match self.input_requests_type:
            case "synthetic-sinusoidal":
                pass
            case "synthetic-normal":
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
        config = {
            "max_steps": self.max_steps,
            "input_requests_type": self.input_requests_type,
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

        # Generate all input requests for the environment.
        limits = {}
        for agent in self.agents:
            limits[agent] = {
                "min": self.observation_space[agent]["input_requests"].low.item(),
                "max": self.observation_space[agent]["input_requests"].high.item(),
            }
        if self.input_requests_type == "synthetic-sinusoidal" or self.input_requests_type == "synthetic":
            self.input_requests = _synthetic_sinusoidal_input_requests(self.max_steps, self.agents, limits, self.rng)
        elif self.input_requests_type == "synthetic-normal":
            self.input_requests = _synthetic_normal_input_requests(self.max_steps, self.agents, limits, self.rng)
        else:  # real
            retval = _real_input_requests(self.max_steps, self.agents, limits, self.rng, self.evaluation)

            self.input_requests = retval[0]

            # Special attribute, not returned in the observation: contains the
            # hashes of the selected input requests. It is used by the
            # callbacks.
            self.input_requests_hashes = retval[1]

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

            result_props, _ = pacsltk.perfmodel.get_sls_warm_count_dist(
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


def _synthetic_normal_input_requests(max_steps, agents, limits, rng):
    """Generates the input requests for the given agents with the given length,
    clipping the values within the given bounds and using the given rng to
    generate the synthesized data.

    limits must be a dictionary whose keys are the agent ids, and each agent has
    two sub-keys: "min" for the minimum value and "max" for the maximum value.

    Returns a dictionary whose keys are the agent IDs and whose value is an
    np.ndarray containing the input requests for each step."""
    mean = 61
    std = 32

    input_requests = {}
    for agent in agents:
        requests = rng.normal(loc=mean, scale=std, size=max_steps)
        input_requests[agent] = np.asarray(requests, dtype=np.int32)

        # Clip the excess values respecting the minimum and maximum values
        # for the input requests observation.
        min = limits[agent]["min"]
        max = limits[agent]["max"]
        np.clip(input_requests[agent], min, max, out=input_requests[agent])

    return input_requests


def _synthetic_sinusoidal_input_requests(max_steps, agents, limits, rng):
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
    noise_ratio = 0.1
    unique_periods = 3  # The periods changes 3 times for each episode.

    input_requests = {}
    steps = np.arange(max_steps)
    for agent in agents:
        # Note: with default max_steps, the period changes every 96 steps
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


def _synthetic_sinusoidal_input_requests_new(max_steps, agents, limits, rng):
    """Generates the input requests for the given agents with the given length,
    clipping the values within the given bounds and using the given rng to
    generate the synthesized data.

    limits must be a dictionary whose keys are the agent ids, and each agent has
    two sub-keys: "min" for the minimum value and "max" for the maximum value.

    Returns a dictionary whose keys are the agent IDs and whose value is an
    np.ndarray containing the input requests for each step."""
    average_requests = np.clip(rng.normal(loc=70), 60, 80)
    amplitude_requests = 60
    noise_ratio = 0.2

    input_requests = {}
    steps = np.arange(max_steps)
    for agent in agents:
        # Sample the function.
        function = rng.choice([np.sin, np.cos])

        # Sample the period in a fixed range.
        periods = rng.uniform(5, high=30)

        # Sample the requests.
        base_input = average_requests + amplitude_requests * function(steps / periods)

        # Add some noise.
        noisy_input = base_input + noise_ratio * rng.normal(0, amplitude_requests, size=max_steps)
        requests = np.asarray(noisy_input, dtype=np.int32)

        # Clip the excess values respecting the minimum and maximum values
        # for the input requests observation.
        min = limits[agent]["min"]
        max = limits[agent]["max"]
        assert min == 0 and max == 150, "Unsupported [min, max] input request range!"
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
    for idx in range(1, total_datasets + 1):
        item = (
            f"d{idx:02}",
            f"invocations_per_function_md.anon.http.scaled.selected.d{idx:02}.csv",
        )
        datasets.append(item)

    # Read each CSV file as a data frame.
    pool = []
    for idx, dataset in datasets:
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


def _real_input_requests(max_steps, agents, limits, rng, evaluation):
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
    dataframes = rng.choice(pool, size=len(agents), replace=False)
    functions = {}
    for agent, dataframe in zip(agents, dataframes):
        row = dataframe.sample(random_state=rng)
        functions[agent] = {"dataframe": row, "idx": dataframe.idx}

    # Extract the input requests and function hashes from the dataframe.
    input_requests, hashes = {}, {}
    for agent in agents:
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
        if env.input_requests_type == "real":
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


def _run_episode():
    """Run a test episode of the DFaaS environment."""
    # config = {"network": ["node_0 node_1", "node_1"]}
    config = {"network": ["node_0 node_1 node_2", "node_3 node_2 node_0", "node_1 node_4"]}
    env = DFaaS(config=config)
    _ = env.reset(seed=42)

    for step in range(env.max_steps):
        env.step(action_dict=env.action_space.sample())


if __name__ == "__main__":
    _run_episode()
