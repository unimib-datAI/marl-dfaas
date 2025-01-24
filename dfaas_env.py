# This file contains the DFaaS multi-agent environment and associated callbacks.
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

# Initialize logger for this module.
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.DEBUG,
)
_logger = logging.getLogger(Path(__file__).name)


def _reward_fw(action, excess, queue):
    """Reward function for the agents in the DFaaS environment.

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
        return 1.0

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

    # Calculate the number of rejected requests that could have been forwarded.
    if forward_reject == 0 and valid_reject > 0:

        # Assume that all rejected requests could have been forwarded because no
        # forwarded requests were rejected.
        reject_excess += valid_reject
        valid_reject = 0

    assert (
        local_excess >= 0
        and forward_reject >= 0
        and forward_excess >= 0
        and reject_excess >= 0
    )
    wrong_reqs = local_excess + forward_reject + forward_excess + reject_excess
    assert (
        wrong_reqs <= reqs_total
    ), f"({local_excess = } + {forward_reject = } + {forward_excess = } + {reject_excess = }) <= {reqs_total}"

    return 1 - (wrong_reqs / reqs_total)


class DFaaS(MultiAgentEnv):
    def __init__(self, config={}):
        """Create the DFaaS environment with the given config. The config
        supports the following keys (see the source code for defaults):

            - network: the graph structure of the DFaaS network, given as
              adjacency lists to be parsed with NetworkX.
            - queue_capacity
            - max_steps
            - input_requests_type
            - evaluation

        """
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
                agent: Simplex(shape=(3,))
                for agent in self.agents
            }
        )

        # Request queue capacity for each agent.
        self.queue_capacity = config.get("queue_capacity", 100)

        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict(
            {
                agent: gym.spaces.Dict(
                    {
                        # Number of input requests to process for a single step.
                        "input_requests": gym.spaces.Box(
                            low=0, high=150, dtype=np.int32
                        ),
                        # Queue current size.
                        "queue_size": gym.spaces.Box(
                            low=0,
                            high=self.queue_capacity,
                            dtype=np.int32,
                        ),
                        # Forwarded requests in the previous step.
                        "prev_forward_requests": gym.spaces.Box(
                            low=0, high=150, dtype=np.int32
                        ),
                        # Forwarded but rejected requests in the previous step.
                        # Note last_forward_rejects <= last_forward_requests.
                        "prev_forward_rejects": gym.spaces.Box(
                            low=0, high=150, dtype=np.int32
                        ),
                    }
                )
                for agent in self.agents
            }
        )

        # Number of steps in the environment. The default is one step for every
        # 5 minutes of a 24-hour day.
        self.max_steps = config.get("max_steps", 288)

        # Type of input requests.
        self.input_requests_type = config.get(
            "input_requests_type", "synthetic-sinusoidal"
        )
        match self.input_requests_type:
            case "synthetic-sinusoidal":
                pass
            case "synthetic-normal":
                pass
            case "real":
                assert (
                    self.max_steps == 288
                ), f"With {self.input_requests_type = } only 288 max_steps are supported"
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
            "queue_capacity": self.queue_capacity,
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

        assert (
            getattr(self, "master_seed", None) is not None
        ), "reset() must be called the first time with a seed"

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
        if (
            self.input_requests_type == "synthetic-sinusoidal"
            or self.input_requests_type == "synthetic"
        ):
            self.input_requests = _synthetic_sinusoidal_input_requests(
                self.max_steps, self.agents, limits, self.rng
            )
        elif self.input_requests_type == "synthetic-normal":
            self.input_requests = _synthetic_normal_input_requests(
                self.max_steps, self.agents, limits, self.rng
            )
        else:  # real
            retval = _real_input_requests(
                self.max_steps, self.agents, limits, self.rng, self.evaluation
            )

            self.input_requests = retval[0]

            # Special attribute, not returned in the observation: contains the
            # hashes of the selected input requests. It is used by the
            # callbacks.
            self.input_requests_hashes = retval[1]

        # Waiting queue of requests for each agent to process locally. The queue
        # has a limited capacity (self.queue_capacity) and the requests are
        # served in FIFO order (left -> first, right -> last).
        self.queues = {agent: deque() for agent in self.agents}

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
        # 1. Convert the action distribution to absolute number of requests.
        action = {}
        for agent in self.agents:
            # Get input requests.
            input_requests = self.input_requests[agent][self.current_step]

            # Convert the action distribution (a distribution of probabilities)
            # into the number of requests to locally process, to forward and to
            # reject.
            action[agent] = _convert_distribution_fw(input_requests, action_dict[agent])

            # Log the absolute number of requests in the info dict.
            self.info["action_local"][agent][self.current_step] = action[agent][0]
            self.info["action_forward"][agent][self.current_step] = action[agent][1]
            self.info["action_reject"][agent][self.current_step] = action[agent][2]

        # 2. Manage the workload.
        additional_rejects = self._manage_workload(action)

        # 3. Calculate the reward for both agents.
        rewards = {}
        for agent in self.agents:
            excess = additional_rejects[agent]
            queue_status = (len(self.queues[agent]), self.queue_capacity)

            reward = _reward_fw(action[agent], excess, queue_status)
            assert isinstance(reward, float), "Unsupported reward type {type(reward)}"

            # Make sure the reward is of type float.
            # rewards[agent] = float(rewards[agent])

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
                    if isinstance(value, np.ndarray):
                        self.info[key][agent][self.current_step] = value.item()

                value = self.info[key][agent][self.current_step - 1]
                if isinstance(value, np.ndarray):
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
                    "queue_size": np.array([0], dtype=np.int32),
                    "input_requests": np.array([input_requests], dtype=np.int32),
                    "prev_forward_requests": np.array([0], dtype=np.int32),
                    "prev_forward_rejects": np.array([0], dtype=np.int32),
                }

            update_info(obs)

            return obs

        # Normal case.
        for agent in self.agents:
            queue_size = self.info["queue_size"][agent][self.current_step - 1]
            input_requests = self.input_requests[agent][self.current_step]
            prev_forward_reqs = self.info["action_forward"][agent][
                self.current_step - 1
            ]
            prev_forward_rejects = self.info["forward_rejects"][agent][
                self.current_step - 1
            ]

            obs[agent]["queue_size"] = np.array([queue_size], dtype=np.int32)
            obs[agent]["input_requests"] = np.array([input_requests], dtype=np.int32)
            obs[agent]["prev_forward_requests"] = np.array(
                [prev_forward_reqs], dtype=np.int32
            )
            obs[agent]["prev_forward_rejects"] = np.array(
                [prev_forward_rejects], dtype=np.int32
            )

        update_info(obs)
        return obs

    def _manage_workload(self, action):
        """Manages the workload for the agents in the current step. It takes the
        input request actions for each agent and returns a dictionary whose keys
        are the agent IDs and whose value is a tuple of two elements: the number
        of local input requests rejected and the number of forwarded requests
        rejected by the other agent because the queue is full.

        The rejected requests for an agent is the sum of: rejected requests
        indicated by the action, the number of local requests that could not be
        processed or added to the queue, the number of forwarded requests to the
        other agent that could not be processed or added to the queue."""
        assert len(action) == len(
            self.agents
        ), f"Expected {len(self.agents)} entries, found {len(action)}"

        # CPU shares and RAM available for all requests in a single step.
        cpu_capacity = {agent: 1000 for agent in self.agents}
        ram_capacity = {agent: 8000 for agent in self.agents}

        # Extract absolute number of requests.
        local = {agent: action[agent][0] for agent in self.agents}
        forward = {agent: action[agent][1] for agent in self.agents}
        reject = {agent: action[agent][2] for agent in self.agents}

        def process_request(agent, request):
            """Process the specified request for the specified agent. Returns
            True if the request has been processed in the current step,
            otherwise returns False."""
            if (
                cpu_capacity[agent] >= request.cpu_shares
                and ram_capacity[agent] >= request.ram_mb
            ):
                cpu_capacity[agent] -= request.cpu_shares
                ram_capacity[agent] -= request.ram_mb
                return True  # Simulate the processing of the request.

            return False

        def append_queue(agent, request):
            """Appends the specified requests to the queue of the specified
            agent. Returns True if the request was appended, otherwise False
            (the queue is full)."""
            if len(self.queues[agent]) < self.queue_capacity:
                self.queues[agent].append(request)
                return True

            return False  # No available space.

        # 1. First, process the requests in the queue from the previous step.
        for agent in self.agents:
            not_processed = deque()
            for request in self.queues[agent]:
                if process_request(agent, request):
                    self.info["processed_local"][agent][self.current_step] += 1
                    # Count the number of processed requests forwarded by the
                    # opposite node.
                    if request.forwarded:
                        self.info["processed_local_forward"][agent][
                            self.current_step
                        ] += 1
                else:
                    not_processed.append(request)

            self.queues[agent] = not_processed
            self.info["queue_size_pre_incoming_local"][agent][self.current_step] = len(
                not_processed
            )

        # 2. Handle incoming local requests.
        for agent in self.agents:
            for request in _sample_workload(local[agent], self.rng):
                # Try to process incoming requests only when the queue is empty.
                if len(self.queues[agent]) == 0 and process_request(agent, request):
                    self.info["processed_local"][agent][self.current_step] += 1
                    continue

                # Queue is not empty or system does not have enough resources,
                # try to add the request to the queue.
                if append_queue(agent, request):
                    continue

                # Insufficient requests and full queue: the only option left is
                # to reject.
                reject[agent] += 1
                self.info["local_rejects_queue_full"][agent][self.current_step] += 1

            self.info["queue_size_pre_incoming_forward"][agent][self.current_step] = (
                len(self.queues[agent])
            )

        # 3. Handle incoming forwarded requests. Same strategy as for incoming
        # local requests.
        for agent in self.agents:
            # TODO: Order matter!
            for neighbor in self.network.neighbors(agent):
                # The forwarded requests are distributed equally to all
                # neighbouring nodes, so I need to calculate the share for the
                # current agent.
                #
                # TODO: This can be improved by using the dictionary structure
                # of the network when processing the action directly instead of
                # this function!
                #
                # FIXME: The division truncates the fractional part, a request
                # may be missing!
                incoming_reqs = forward[neighbor] // self.network.degree(neighbor)

                for request in _sample_workload(incoming_reqs, self.rng):
                    request = request._replace(forwarded=True)  # Mark the request.
                    if len(self.queues[agent]) == 0 and process_request(agent, request):
                        self.info["processed_local"][agent][self.current_step] += 1
                        if request.forwarded:
                            self.info["processed_local_forward"][agent][
                                self.current_step
                            ] += 1
                        continue
                    if append_queue(agent, request):
                        continue

                    self.info["forward_rejects"][neighbor][self.current_step] += 1

            self.info["queue_size"][agent][self.current_step] = len(self.queues[agent])

        # Fill the dictionary to return.
        retval = {}
        for agent in self.agents:
            retval[agent] = (
                self.info["local_rejects_queue_full"][agent][self.current_step],
                self.info["forward_rejects"][agent][self.current_step],
            )

        return retval


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
        # Note: with default max_stes, the period changes every 96 steps
        # (max_steps = 288). We first generate the periods and expand the array
        # to match the max_steps. If max_steps is not a multiple of 96, some
        # elements must be appended at the end, hence the resize call.
        repeats = max_steps // unique_periods
        periods = rng.uniform(15, high=100, size=unique_periods)
        periods = np.repeat(periods, repeats)  # Expand the single values.
        periods = np.resize(periods, periods.size + max_steps - periods.size)

        base_input = average_requests + amplitude_requests * np.sin(
            2 * np.pi * steps / periods
        )
        noisy_input = base_input + noise_ratio * rng.normal(
            0, amplitude_requests, size=max_steps
        )
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
        assert np.all(reqs >= min) and np.all(
            reqs <= max
        ), f"Unsupported limits: {limits[agent]}"

        input_requests[agent] = reqs
        hashes[agent] = hash

    return input_requests, hashes


def _convert_distribution_fw(input_requests, action_dist):
    """Converts the given action distribution (e.g. [.7, .2, .1]) into the
    absolute number of requests to process locally, to forward and to reject.
    Returns the result as a tuple."""
    assert len(action_dist) == 3, "Expected (local, forward, reject)"

    # Extract the three actions from the action distribution
    prob_local, prob_forwarded, prob_rejected = action_dist

    # Get the corresponding number of requests for each action. Note: the number
    # of requests is a discrete number, so there is a fraction of the action
    # probabilities that is left out of the calculation.
    actions = [
        int(prob_local * input_requests),  # local requests
        int(prob_forwarded * input_requests),  # forwarded requests
        int(prob_rejected * input_requests),
    ]  # rejected requests

    processed_requests = sum(actions)

    # There is a fraction of unprocessed input requests. We need to fix this
    # problem by assigning the remaining requests to the higher fraction for the
    # three action probabilities, because that action is the one that loses the
    # most.
    if processed_requests < input_requests:
        # Extract the fraction for each action probability.
        fractions = [
            prob_local * input_requests - actions[0],
            prob_forwarded * input_requests - actions[1],
            prob_rejected * input_requests - actions[2],
        ]

        # Get the highest fraction index and and assign remaining requests to
        # that action.
        max_fraction_index = np.argmax(fractions)
        actions[max_fraction_index] += input_requests - processed_requests

    assert sum(actions) == input_requests
    return tuple(actions)


def _sample_workload(num_requests, rng):
    """Samples the given number of requests using the given NumPy RNG generator.

    Returns a deque, each element of which is a namedtuple with the following
    fields:

        * "type": the class type of the request ("A", "B", or "C"),
        * "forwarded": True if the request was forwarded by the opposite node,
        * "cpu_shares": CPU shares used by the request,
        * "ram_mb": RAM used by the request in megabytes.
    """
    workload = deque()

    request_tuple = namedtuple("Request", "type, forwarded, cpu_shares, ram_mb")

    for request_class in rng.choice(["A", "B", "C"], num_requests):
        # The distribution values are taken from the code of the original
        # article, see here: https://github.com/unimib-datAI/rl-dfaas-seated24
        match request_class:
            case "A":
                cpu_min, cpu_mean, cpu_max = 1, 5.5, 10
                ram_min, ram_mean, ram_max = 1, 13, 25
            case "B":
                cpu_min, cpu_mean, cpu_max = 11, 15.5, 20
                ram_min, ram_mean, ram_max = 26, 38, 50
            case "C":
                cpu_min, cpu_mean, cpu_max = 21, 25.5, 30
                ram_min, ram_mean, ram_max = 51, 63, 75
            case _:
                assert False, "Unreachable code"

        # TODO: Rewrite the distribution to use only integers.
        # See here: https://stackoverflow.com/a/50004451

        std_dev = 2.5  # Fixed standard deviation for both CPU and RAM.
        cpu_shares = np.clip(rng.normal(cpu_mean, std_dev), cpu_min, cpu_max)
        ram_mb = np.clip(rng.normal(ram_mean, std_dev), ram_min, ram_max)

        workload.append(request_tuple(request_class, False, cpu_shares, ram_mb))

    return workload


# Register the environments with Ray so that they can be used automatically when
# creating experiments.
def register(env_class):
    register_env(env_class.__name__, lambda env_config: env_class(config=env_config))


register(DFaaS)


class DFaaSCallbacks(DefaultCallbacks):
    """User defined callbacks for the DFaaS environment.

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
        result["callbacks_ok"] = True
