"""This module generates the average rate of input requests for the agent's
observation dictionary. The generated rates can be from real or synthetic data.
This module is used by the DFaaS environment at the beginning of an episode to
get the input rate for all steps."""

from pathlib import Path

import numpy as np
import pandas as pd


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
    limits = {"node_0": {"min": min_reqs, "max": max_reqs}}

    trace = synthetic_sinusoidal(max_steps, agents, limits, rng)

    return trace["node_0"]


def synthetic_sinusoidal(max_steps, agents, limits, rng):
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


def synthetic_constant(max_steps, agents):
    """Generates a constant input rate trace for each agent for the given
    length.

    Current limitations: only two-agent environments are supported, and the
    constat rates are hardcoded as 5 and 100."""
    assert len(agents) == 2, "Only two agents supported by this input rate generation method"

    input_rate = {}
    constant_rates = np.array([5, 100], dtype=np.int32)
    for agent, rate in zip(agents, [5, 100]):
        input_rate[agent] = np.repeat(rate, max_steps)

    return input_rate


def synthetic_linear_growth(max_steps, agents):
    """Generates an input rate trace where the first agent's rate is constant
    (5), and the second agent's rate grows linearly from 1 to 150.

    Only two-agent environments are supported.
    """
    assert len(agents) == 2, "Only two agents supported by this input rate generation method"

    input_rate = {}
    input_rate[agents[0]] = np.repeat(5, max_steps)

    # Generate linear growth from 1 to 150 (inclusive) over max_steps, using
    # integers.
    linear = np.linspace(1, 150, max_steps)
    input_rate[agents[1]] = np.round(linear).astype(np.int32)

    return input_rate


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
    assert len(agents) == 2, "Only two agents supported"
    assert len(rates_before) == len(rates_after) == 2, "Rates must be length 2"
    assert sum(rates_before) <= 120, "Sum of initial rates exceeds 120"
    assert sum(rates_after) <= 120, "Sum of final rates exceeds 120"

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
            _logger.critical(f"Dataset file not found: {path.as_posix()!r}")
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        frame.idx = idx  # Special metadata to know the original file.

        pool.append(frame)

    global _real_input_requests_pool
    _real_input_requests_pool = np.array(pool, dtype=object)


def real(max_steps, agents, limits, rng, evaluation):
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
