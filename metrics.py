# This Python script calculates additional metrics from the 'result.json' file
# in the experiment output directory and saves the data to the 'metrics.json'
# file in the same directory.
#
# Some scripts related to plotting require this script to be run before
# plotting.
import argparse
from pathlib import Path
import logging
import json

import numpy as np

import dfaas_utils

# Initialize logger for this module.
logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s", level=logging.DEBUG)
logger = logging.getLogger(Path(__file__).name)


def get_agents(iters):
    """Returns the agent IDs from the given list of iterations. Assumes at least
    one iteration and one episode for iteration."""
    return list(iters[0]["hist_stats"]["input_requests"][0].keys())


def average_reward_per_step(iters, metrics):
    """Calculates average reward per step per episode."""
    agents = get_agents(iters)

    iter = 0
    while iter < len(metrics["iterations"]):
        # Temporally accumulate the average reward of each episode for that
        # iteration.
        avg_reward_iter = []
        for episode in iters[iter]["hist_stats"]["reward"]:
            # The average reward for each step in a single episode, for each
            # agent.
            avg_reward = {}
            for agent in agents:
                avg_reward[agent] = np.average(episode[agent])

            avg_reward_iter.append(avg_reward)

        metrics["iterations"][iter]["reward_average"] = avg_reward_iter

        iter += 1


def main(exp_dir):
    exp_dir = dfaas_utils.to_pathlib(exp_dir)

    # Each element is one iteration.
    iters = dfaas_utils.parse_result_file(exp_dir / "result.json")

    # The metrics dictionary: it contains all calculated metrics for each
    # iteration and episode.
    metrics = {"iterations": [None] * len(iters)}

    # Initialize internal dicts for all iterations.
    for idx in range(len(iters)):
        metrics["iterations"][idx] = {}

    average_reward_per_step(iters, metrics)

    # Save the metrics dictionary to disk as a JSON file.
    metrics_path = exp_dir / "metrics.json"
    dfaas_utils.dict_to_json(metrics, metrics_path)
    logger.info(f"Metrics data saved to: {metrics_path.as_posix()!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="metrics")

    parser.add_argument(dest="exp_directory",
                        help="Directory with the results.json file")

    args = parser.parse_args()

    main(args.exp_directory)
