import argparse
import json
import sys

import numpy as np

import utils
sys.stdout = utils.OutputDuplication()
sys.stderr = sys.stdout

from RL4CC.utilities.logger import Logger

# Logger used in this file.
logger = Logger(name="DFAAS-METRICS", verbose=2)

def aggregate_data(data):
    """Returns the following aggregated data given the evaluations for a single
    scenario as data:

    - Mean of the rewards,
    - Standard deviation of the rewards,
    - Mean of steps in congested state,
    - Mean of rejected requests.

    All values are calculated for episode, not for steps in all episodes (e.g.
    mean of rejected requests for episode).
    """
    # Prepare the empty arrays to store the metrics for each episode.
    num_episodes = len(data)
    steps_cong = np.empty(shape=num_episodes, dtype=np.int64)
    rejected_reqs = np.empty(shape=num_episodes, dtype=np.int64)
    rewards = np.empty(shape=num_episodes)
    idx = 0

    # Iterate over all episodes.
    for episode in data:
        tmp_steps_cong = 0
        tmp_rejected_reqs = 0
        tmp_rewards = 0

        for step in episode["evaluation_steps"]:
            tmp_steps_cong += step["obs_info"]["congested"]
            tmp_rejected_reqs += step["obs_info"]["actions"]["rejected"]
            tmp_rewards += step["reward"]

        steps_cong[idx] = tmp_steps_cong
        rejected_reqs[idx] = tmp_rejected_reqs
        rewards[idx] = tmp_rewards
        idx += 1

    # Calculate the means and standard deviation.
    rewards_mean = np.mean(rewards)
    rewards_std = np.std(rewards)
    steps_cong_mean = np.mean(steps_cong)
    rejected_reqs_mean = np.mean(rejected_reqs)

    return rewards_mean, rewards_std, steps_cong_mean, rejected_reqs_mean


def calculate_metrics(results_directory):
    # This dictionary contains the raw data resulted from the previous phase.
    data = {}

    eval_file = "evaluations_scenarios.json"
    try:
        with open(f"{results_directory}/{eval_file}") as file:
            data = json.load(file)
    except IOError as e:
        logger.err(f"Failed to read {eval_file}: {e}")
        sys.exit(1)

    # We do not set the log file until after we have read the evaluation JSON
    # file, because now we are sure that the directory exists.
    sys.stdout.set_logfile(f"{results_directory}/metrics.log")

    logger.log(f"START metrics calculation")
    logger.log(f"  Train scenario: {data['train_scenario']!r}")
    logger.log(f"  Episodes for each scenario: {data['num_episodes_for_scenario']}")
    logger.log(f"  Exploration allowed? {data['allow_exploration']}")
    logger.breakline()

    # This dictionary contains the metrics data that will be saved at the end.
    metrics_data = {"train_scenario": data["train_scenario"],
                    "num_episodes_for_scenario": data["num_episodes_for_scenario"],
                    "allow_exploration": data["allow_exploration"],
                    "metrics": {}
                    }

    for scenario in data["evaluations"]:
        logger.log(f"Metrics for scenario {scenario!r}")

        mean, std, steps_cong, rejected_reqs = aggregate_data(data["evaluations"][scenario])

        metrics_data["metrics"][scenario] = {"mean": mean,
                                             "std": std,
                                             "steps_cong": steps_cong,
                                             "rejected_reqs": rejected_reqs
                                             }

        logger.log(f"  Average reward: {mean}")
        logger.log(f"  Dev. Std. of reward: {std}")
        logger.log(f"  Steps in congestion state: {steps_cong}")
        logger.log(f"  Rejected requests: {rejected_reqs}")
        logger.breakline()

    # Save metrics data as JSON file to disk.
    metrics_file = "evaluations_metrics.json"
    try:
        with open(f"{results_directory}/{metrics_file}", "w") as file:
            json.dump(metrics_data, file)
    except IOError as e:
      logger.err(f"Failed to write metrics results to {metrics_file!r}: {e}")
      sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="metrics",
                                     description="Metrics extraction from evaluation phase")

    parser.add_argument(dest="results_directory",
                        help="Directory where the evaluation results are saved.")

    args = parser.parse_args()

    calculate_metrics(args.results_directory)
