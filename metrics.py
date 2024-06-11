import argparse
from pathlib import Path

import utils

import numpy as np

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

    assert idx == num_episodes, "The ndarrays have some elements not accessed"

    # Calculate the means and standard deviation.
    rewards_mean = np.mean(rewards)
    rewards_std = np.std(rewards)
    steps_cong_mean = np.mean(steps_cong)
    rejected_reqs_mean = np.mean(rejected_reqs)

    return rewards_mean, rewards_std, steps_cong_mean, rejected_reqs_mean


def calculate_metrics(exp_dir):
    """Returns a dictionary with the calculated metrics from the given
    experiment directory where the metrics.json file is located.

    This function calculates the metrics for a single experiment, while
    calculate_aggregate_metrics() calculates the aggregate metrics for a list of
    experiments."""
    # Read raw evaluations data from the experiment.
    eval_path = Path(exp_dir, "evaluations_scenarios.json")
    data = utils.json_to_dict(eval_path)

    # A dictionary containing some general information about the reports and the
    # calculated metrics data.
    metrics = {"train_scenario": data["train_scenario"],
               "num_episodes_for_scenario": data["num_episodes_for_scenario"],
               "allow_exploration": data["allow_exploration"],
               "scenarios": {}
               }

    # Each scenario has its own metrics, we cannot mix them.
    for (scenario, scenario_data) in data["scenarios"].items():
        mean, std, steps_cong, rejected_reqs = aggregate_data(scenario_data)

        metrics["scenarios"][scenario] = {"rewards_mean": mean,
                                          "rewards_std": std,
                                          "steps_cong_mean": steps_cong,
                                          "rejected_reqs_mean": rejected_reqs
                                          }

    return metrics


def calculate_aggregate_metrics(exp_dirs):
    """Returns a dictionary with the computed aggregated metrics from the given
    experiment directories where the metrics.json file exists."""
    assert len(exp_dirs) >= 1
    num_exp = len(exp_dirs)

    # Read only the metrics file data of the first experiment to know how many
    # scenarios there are and to set up the arrays.
    metrics_path = Path(exp_dirs[0], "metrics.json")
    metrics = utils.json_to_dict(metrics_path)
    raw_data = {}
    for (scenario, scenario_data) in metrics["scenarios"].items():
        # This array contains the individual metrics for each experiment and
        # scenario (each scenario has its own metrics).
        raw_data[scenario] = {}
        raw_data[scenario]["rewards_mean"] = np.empty(shape=num_exp)
        raw_data[scenario]["rewards_std"] = np.empty(shape=num_exp)
        raw_data[scenario]["steps_cong_mean"] = np.empty(shape=num_exp)
        raw_data[scenario]["rejected_reqs_mean"] = np.empty(shape=num_exp)

        # Start filling the arrays.
        raw_data[scenario]["rewards_mean"][0] = scenario_data["rewards_mean"]
        raw_data[scenario]["rewards_std"][0] = scenario_data["rewards_std"]
        raw_data[scenario]["steps_cong_mean"][0] = scenario_data["steps_cong_mean"]
        raw_data[scenario]["rejected_reqs_mean"][0] = scenario_data["rejected_reqs_mean"]

    # Iterate over all experiment metrics and finish to fill the raw_data
    # arrays.
    idx = 1
    for exp_dir in exp_dirs[1:]:
        metrics_path = Path(exp_dir, "metrics.json")
        metrics = utils.json_to_dict(metrics_path)

        for (scenario, scenario_data) in metrics["scenarios"].items():
            raw_data[scenario]["rewards_mean"][idx] = scenario_data["rewards_mean"]
            raw_data[scenario]["rewards_std"][idx] = scenario_data["rewards_std"]
            raw_data[scenario]["steps_cong_mean"][idx] = scenario_data["steps_cong_mean"]
            raw_data[scenario]["rejected_reqs_mean"][idx] = scenario_data["rejected_reqs_mean"]

        idx += 1

    assert idx == num_exp, "The ndarrays have some unitialized data"

    # Now it is possible to calculate the aggregated metrics.
    aggr_metrics = {}
    for (scenario, scenario_data) in raw_data.items():
        aggr_metrics[scenario] = {
                "rewards_mean": scenario_data["rewards_mean"].mean(),
                "rewards_min": scenario_data["rewards_mean"].min(),
                "rewards_max": scenario_data["rewards_mean"].max(),

                "rewards_std_mean": scenario_data["rewards_std"].mean(),
                "rewards_std_min": scenario_data["rewards_std"].min(),
                "rewards_std_max": scenario_data["rewards_std"].max(),

                "steps_cong_mean": scenario_data["steps_cong_mean"].mean(),
                "steps_cong_min": scenario_data["steps_cong_mean"].min(),
                "stesp_cong_max": scenario_data["steps_cong_mean"].max(),

                "rejected_reqs_mean": scenario_data["rejected_reqs_mean"].mean(),
                "rejected_reqs_min": scenario_data["rejected_reqs_mean"].min(),
                "rejected_reqs_max": scenario_data["rejected_reqs_mean"].max(),
                }

    return aggr_metrics


def main(experiments_directory):
    experiments_path = Path(experiments_directory, "experiments.json")
    experiments = utils.json_to_dict(experiments_path)

    # Dictionary of aggregated metrics, stored as "metrics.json" in the main
    # experiments directory.
    aggr_metrics = {}

    for (algo, algo_value) in experiments.items():
        for (params, params_value) in algo_value.items():
            for (scenario, scenario_value) in params_value.items():
                all_done = True
                exp_dirs = []

                # Calculate metrics for a single experiment.
                for exp in scenario_value.values():
                    if not exp["done"]:
                        logger.warn(f"Skipping experiment {exp['id']!r} because it is not done")
                        all_done = False
                        continue
                    logger.log(f"Calculating metrics for {exp['id']!r}")

                    exp_directory = Path(experiments_directory, exp["directory"])
                    exp_dirs.append(exp_directory)
                    metrics = calculate_metrics(exp_directory)

                    # Save the metrics to disk.
                    metrics_path = Path(exp_directory, "metrics.json")
                    utils.dict_to_json(metrics, metrics_path)

                # When calculating aggregate metrics, all sub-experiments must
                # be run.
                exp_id = f"{algo}:{params}:{scenario}"
                if not all_done:
                    logger.warn(f"Skipping aggregate experiment {exp_id!r} because not all experiments are done")
                    continue
                logger.log(f"Calculating aggregate metrics for {exp_id!r}")

                # Calculate aggregate metrics for all the same experiments but
                # with different seeds.
                metrics = calculate_aggregate_metrics(exp_dirs)
                aggr_metrics[exp_id] = metrics
                aggr_metrics[exp_id]["id"] = exp_id

                # Save the updated metrics data to disk.
                aggr_metrics_path = Path(experiments_directory, "metrics.json")
                utils.dict_to_json(aggr_metrics, aggr_metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="metrics",
                                     description="Metrics extraction from evaluation phase")

    parser.add_argument(dest="experiments_directory",
                        help="Directory with the experiments.json file")

    args = parser.parse_args()

    main(args.experiments_directory)
