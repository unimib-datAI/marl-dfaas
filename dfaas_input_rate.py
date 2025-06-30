"""This module contains several methods for generating the average rate of input
requests, also called the input rate.

You can call a generator directly or use the registry, which maps generator
names to their respective implementation functions.

The DFaaS environment uses this module at the beginning of an episode to obtain
the input rate for all steps."""

from pathlib import Path
import errno
from copy import deepcopy

import numpy as np
import pandas as pd


# This is a registry that holds all the supported input rate generators. The key
# is the generator string ID and the value is the associated function.
_generator = {}


def _register_generator(name):
    """Register a generator function with the given name to _generator."""

    def decorator(function):
        _generator[name] = function
        return function

    return decorator


def generator(name):
    """Returns the generator with the given name.

    Args:
        name (str): Generator name.

    Raises:
        ValueError: If the generator is not found on the registry.
    """
    if name not in _generator:
        e = ValueError(f"Unsupported {name!r} input rate generation method")
        e.add_note(f"Supported generators are {list(_generator.keys())}")
        raise e

    return _generator[name]


def scale_down(traces, max_per_agent=63, min_rate_per_agent=1, max_rate_per_agent=150):
    """Proportionally scale down input rate traces per agent when the total capacity is exceeded.

    Args:
        traces (dict): Dictionary mapping agent identifiers to 1D numpy arrays
            of rates. All arrays must have the same length.
        max_per_agent (int): Maximum allowed rate per agent (default is 63).
        min_rate_per_agent (int): Minimum allowed rate per agent (default is 1).
        max_rate_per_agent (int): Maximum allowed rate per agent (default is 150).

    Returns:
        scaled_traces (dict): A dictionary in the same format as `traces`, but
            with rates scaled down where necessary to ensure that the sum across all
            agents does not exceed the total system capacity at any time step, and
            all rates satisfy the per-agent min/max constraints.
    """
    agents = list(traces.keys())
    max_steps = len(next(iter(traces.values())))  # All traces have the same length, just get the first.
    total_capacity_step = max_per_agent * len(agents)
    scaled_traces = deepcopy(traces)

    for step in range(max_steps):
        input_rate_step = np.array([traces[agent][step] for agent in agents])
        if input_rate_step.sum() <= total_capacity_step:
            continue  # Capacity not exceeded, nothing to do.

        # Traces for this step must be scaled down proportionally.
        scaled = np.round(input_rate_step * total_capacity_step / input_rate_step.sum())

        # Clip values to ensure they're in the correct range.
        scaled = np.clip(scaled, min_rate_per_agent, max_rate_per_agent).astype(int)

        # Adjust single input rates to match total_capacity_step. Rounding and
        # clipping can cause the sum to be slightly off.
        diff = total_capacity_step - scaled.sum()
        while diff != 0:
            if diff > 0:
                # Add to agents below max_rate_per_agent.
                for i in range(len(scaled)):
                    if scaled[i] < max_rate_per_agent:
                        scaled[i] += 1
                        diff -= 1
                        if diff == 0:
                            break
            else:
                # Subtract from agents above min_rate_per_agent.
                for i in range(len(scaled)):
                    if scaled[i] > min_rate_per_agent:
                        scaled[i] -= 1
                        diff += 1
                        if diff == 0:
                            break

        # Update the scaled_traces with the new rates. Each agent is mapped to
        # an integer (the orders is the same across all iterations).
        for agent, idx in zip(agents, range(len(agents)), strict=True):
            scaled_traces[agent][step] = scaled[idx]

    return scaled_traces


@_register_generator("synthetic-normal")
def synthetic_normal(max_steps, agents, limits, rng):
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


def _gen_synthetic_sinusoidal(rng):
    """Returns a single generated synthetic input rate trace. Usually used for
    tests."""
    # These are just fixed values for one trace.
    agents = ["node_0"]
    max_steps, min_reqs, max_reqs = 288, 1, 150

    trace = synthetic_sinusoidal(max_steps, agents, rng)

    return trace["node_0"]


@_register_generator("synthetic-sinusoidal")
def synthetic_sinusoidal(max_steps, agents, rng=None):
    """Generate synthetic sinusoidal input rate traces for a set of agents.

    Args:
        max_steps (int): The number of time steps for which to generate the
            input rate trace.
        agents (list): A list of agent identifiers for which to generate traces.
        rng: A NumPy RNG instance. If None, a new default_rng() is created.

    Returns
        traces (dict): A dictionary mapping each agent identifier to its
            corresponding 1D numpy array of input rates (length `max_steps`),
            representing a noisy, phase-shifted sinusoidal trace.

    Raises:
        ValueError: If `max_steps` or `max_per_agent` is not greater than zero.

    Notes:
        - All agents have the same baseline and amplitude for the sinusoid, but
          noise ratio is randomized for each agent (in range [0.05, 0.1]).
        - Phases are evenly spaced to avoid global overload events (overlapping
          peaks).
        - Input rates are clipped to the interval [1, 150].
    """
    if rng is None:
        rng = np.random.default_rng()
    if max_steps <= 0:
        raise ValueError(f"Expected > 0, found {max_steps = }!")

    # All agents have the same basline_rate and amplitude_rate.
    baseline_rate = np.repeat(70, len(agents))
    amplitude_rate = np.repeat(65, len(agents))
    noise_ratio = rng.uniform(0.05, 0.1, len(agents))  # But different noise ratio.
    steps = np.arange(max_steps)
    rate_min, rate_max = 1, 150

    # Avoid overlapping phases (that may cause a global overload) by evenly
    # spacing the phase for each agent.
    phi = np.linspace(0, 2 * np.pi, len(agents), endpoint=False)

    # Generate the input rate traces for each agent.
    input_rates = []
    for idx in range(len(agents)):
        base_rate = amplitude_rate[idx] * np.sin(2 * np.pi * steps / max_steps + phi[idx]) + baseline_rate[idx]

        noisy_rate = base_rate + noise_ratio[idx] * rng.normal(0, amplitude_rate[idx], size=max_steps)

        clipped_rate = np.clip(np.round(noisy_rate), rate_min, rate_max)

        input_rates.append(clipped_rate)

    # Randomly assign an input rate trace for each agent.
    traces = {}
    for agent, input_rate in zip(agents, rng.permutation(input_rates)):
        traces[agent] = input_rate

    return traces


@_register_generator("synthetic-constant")
def synthetic_constant(max_steps, agents):
    """Generates a constant input rate trace for each agent for the given
    length.

    Current limitations: only two-agent environments are supported, and the
    constat rates are hardcoded as 5 and 100."""
    if len(agents) != 2:
        raise ValueError("Only two agents supported by this input rate generation method")

    input_rate = {}
    constant_rates = np.array([5, 100], dtype=np.int32)
    for agent, rate in zip(agents, [5, 100]):
        input_rate[agent] = np.repeat(rate, max_steps)

    return input_rate


@_register_generator("synthetic-linear-growth")
def synthetic_linear_growth(max_steps, agents):
    """Generates an input rate trace where the first agent's rate is constant
    (5), and the second agent's rate grows linearly from 1 to 150.

    Only two-agent environments are supported.
    """
    if len(agents) != 2:
        raise ValueError("Only two agents supported by this input rate generation method")

    input_rate = {}
    input_rate[agents[0]] = np.repeat(5, max_steps)

    # Generate linear growth from 1 to 150 (inclusive) over max_steps, using
    # integers.
    linear = np.linspace(1, 150, max_steps)
    input_rate[agents[1]] = np.round(linear).astype(np.int32)

    return input_rate


@_register_generator("synthetic-double-linear-growth")
def synthetic_double_linear_growth(max_steps, agents, max_per_agent=63, rng=None):
    """Generates input rate traces for two agents, both following linear growth
    with random start/end points and slopes (in [1, 150]). The sum of requests
    for both agents at any step does not exceed 2*max_per_agent.

    Args:
        max_steps: Number of steps in the trace.
        agents: List of agent IDs.
        max_per_agent: Maximum requests per agent per step (single value for all).
        rng: An optional Numpy RNG for reproducibility.

    Returns:
        dict: agent -> array of input rates.
    """
    if len(agents) != 2:
        raise ValueError("Only two agents supported by this input rate generation method")

    if rng is None:
        rng = np.random.default_rng()

    # Min and max rates taken from the environment observation space.
    min_rate, max_rate = 1, 150

    # First agent: random linear growth in [1, 150]
    start1, end1 = rng.integers(min_rate, max_rate, size=2)
    trace1 = np.round(np.linspace(start1, end1, max_steps)).astype(np.int32)

    # Second agent: random linear growth in [1, 150], but clipped.
    start2, end2 = rng.integers(min_rate, max_rate, size=2)
    trace2 = np.round(np.linspace(start2, end2, max_steps)).astype(np.int32)
    for step in range(max_steps):
        # Clip the input trace based on the other trace and the max for step.
        allowed = min(trace2[step], 2 * max_per_agent - trace1[step], max_per_agent)

        # Ensure that the min value is contained in the min range. This may
        # exceed the condition that both input traces must be lesser or equal
        # than 2*max_per_agent, but one rate is excess it not problematic.
        trace2[step] = max(1, allowed)

    # Randomly assign the traces to the agents.
    agents = rng.permutation(agents)
    input_rate = {}
    input_rate[str(agents[0])] = trace1
    input_rate[str(agents[1])] = trace2

    return input_rate


@_register_generator("synthetic-step-change")
def synthetic_step_change(max_steps, agents, rates_before=[5, 100], rates_after=[70, 30]):
    """Generates a step-change input rate trace for each agent for the given
    length.

    Limitations: at the midpoint of the episode, the input rate for each agent
    switches from an initial value to a final value. Only two-agent environments
    are supported, and the rates must be specified. The sum of the input rates
    for both agents at any time must not exceed 120.

    Args:
        max_steps: Length of the trace.
        agents: List of agent IDs.
        rates_before (list): List of rates before change.
        rates_after (list): List of rates after change.

    Returns:
        dict: agent -> array of input rates
    """
    if len(agents) != 2:
        raise ValueError("Only two agents supported by this input rate generation method")
    if len(rates_before) != len(rates_after) != 2:
        raise ValueError("Rates must be of length 2")
    if sum(rates_before) > 120 or sum(rates_after) > 120:
        raise ValueError("Sum of initial or final rates exceeds 120")

    change_point = max_steps // 2

    input_rate = {}
    for agent, before, after in zip(agents, rates_before, rates_after):
        trace = np.concatenate([np.repeat(before, change_point), np.repeat(after, max_steps - change_point)])
        input_rate[agent] = trace

    return input_rate


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
            raise FileNotFoundError(errno.ENOENT, "Dataset file not found", path)

        frame = pd.read_csv(path)
        frame.idx = idx  # Special metadata to know the original file.

        pool.append(frame)

    global _real_input_requests_pool
    _real_input_requests_pool = np.array(pool, dtype=object)


@_register_generator("real")
def real(max_steps, agents, limits, rng, evaluation):
    """Randomly selects a real input request from the pool for each of the given
    agents.

    Since the steps and values of the real input requests are fixed, if the
    given values don't respect the fixed values, an exception will be raised.

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
        if reqs.size != max_steps:
            raise ValueError(f"Unsupported given max_steps = {max_steps}")
        min = limits[agent]["min"]
        max = limits[agent]["max"]
        if np.all(reqs < min) or np.all(reqs > max):
            raise ValueError(f"Unsupported limits: {limits[agent]}")

        input_requests[agent] = reqs
        hashes[agent] = hash

    return input_requests, hashes


def _gen_example_input_rate(seed):
    """Generate a sample trace and print it to standard output."""
    rng = np.random.default_rng(seed=seed)  # RNG used to generate the requests.

    trace = _gen_synthetic_sinusoidal(rng)

    print(trace)


if __name__ == "__main__":
    import argparse

    description = """Generate an input rate trace and print to the standard
    output."""

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=42, help="RNG seed used to generate the trace")

    args = parser.parse_args()

    _gen_example_input_rate(args.seed)
