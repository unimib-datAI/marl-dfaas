# This Python script calculates additional metrics from the 'result.json' file
# in the experiment output directory and saves the data to the 'metrics.json'
# file in the same directory.
#
# Some scripts related to plotting require this script to be run before
# plotting.
import argparse
from pathlib import Path
import logging

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


def rejected_reqs_percent_exceed_per_step(iters, metrics):
    """For each step, for each episode, for each iteration, calculates the
    percentage of rejected requests that could have been handled locally but
    were rejected by the agent."""
    agents = get_agents(iters)

    for iter in range(len(metrics["iterations"])):
        iter_data = []
        for episode in range(len(iters[iter]["hist_stats"]["seed"])):
            episode_data = {}
            for agent in agents:
                steps = len(iters[iter]["hist_stats"]["input_requests"][episode][agent])
                input_reqs = np.empty(steps, dtype=np.int32)
                reject_reqs = np.empty(steps, dtype=np.int32)
                reject_reqs_exc_percent = np.empty(steps, dtype=np.float32)

                input_reqs = np.asarray(iters[iter]["hist_stats"]["input_requests"][episode][agent],
                                        dtype=np.int32)
                reject_reqs = np.asarray(iters[iter]["hist_stats"]["action"][episode]["reject"][agent],
                                         dtype=np.int32)

                for step in range(steps):
                    # TODO: get the max requests to process locally dinamically
                    # from the environment or some config.
                    if input_reqs[step] > 100:
                        tmp = reject_reqs[step] - (input_reqs[step] - 100)
                    else:  # input_reqs[step] < 100 -> all reject requests are too much
                        tmp = reject_reqs[step] * 100 / input_reqs[step]

                    reject_reqs_exc_percent[step] = tmp

                episode_data[agent] = reject_reqs_exc_percent

            iter_data.append(episode_data)

        metrics["iterations"][iter]["rejected_reqs_percent_exceed_per_step"] = iter_data


def average_rejected_reqs_percent_exceed_per_episode(iters, metrics):
    """For each episode, for each iteration, calculates the average percentage
    of rejected requests that could have been handled locally but were rejected
    by the agent."""
    agents = get_agents(iters)

    for iter in range(len(metrics["iterations"])):
        iter_data = []
        for episode in range(len(metrics["iterations"][iter]["rejected_reqs_percent_exceed_per_step"])):
            episode_data = {}
            for agent in agents:
                tmp = np.average(metrics["iterations"][iter]["rejected_reqs_percent_exceed_per_step"][episode][agent])
                episode_data[agent] = tmp

            iter_data.append(episode_data)

        metrics["iterations"][iter]["rejected_reqs_percent_exceed_per_episode"] = iter_data


def average_rejected_reqs_percent_exceed_per_iteration(iters, metrics):
    """For each iteration, calculates the average percentage of rejected
    requests that could have been handled locally but were rejected by the
    agent."""
    agents = get_agents(iters)

    for iter in range(len(metrics["iterations"])):
        episodes = len(metrics["iterations"][iter]["rejected_reqs_percent_exceed_per_episode"])

        iter_data = {}
        for agent in agents:
            iter_data[agent] = np.empty(episodes, np.float32)

            for episode in range(episodes):
                iter_data[agent][episode] = metrics["iterations"][iter]["rejected_reqs_percent_exceed_per_episode"][episode][agent]

        tmp = {}
        for agent in agents:
            tmp[agent] = np.average(iter_data[agent])
        metrics["iterations"][iter]["rejected_reqs_percent_exceed_per_iteration"] = tmp


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

    rejected_reqs_percent_exceed_per_step(iters, metrics)

    average_rejected_reqs_percent_exceed_per_episode(iters, metrics)

    average_rejected_reqs_percent_exceed_per_iteration(iters, metrics)

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
