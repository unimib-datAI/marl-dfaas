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

    # These are just fixed values for one trace.
    agents = ["node_0"]
    max_steps, min_reqs, max_reqs = 288, 0, 150
    limits = {"node_0": {"min": min_reqs, "max": max_reqs}}

    # Generate a specific seed for this trace.
    iinfo = np.iinfo(np.uint32)
    seed = rng.integers(0, high=iinfo.max, size=1)[0].item()
    specific_rng = np.random.default_rng(seed=seed)

    trace = synthetic_sinusoidal_input_requests(max_steps, agents, limits, specific_rng)

    print(trace["node_0"])


if __name__ == "__main__":
    import argparse

    description = """Generate an input rate trace and print to the standard
    output."""

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=42, help="RNG seed used to generate the trace")

    args = parser.parse_args()

    _gen_example_input_rate(args.seed)
