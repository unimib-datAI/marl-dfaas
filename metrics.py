import argparse
from pathlib import Path

import utils

import numpy as np

from RL4CC.utilities.logger import Logger

# Logger used in this file.
logger = Logger(name="DFAAS-METRICS", verbose=2)


def eval_single_exp_single_scenario(eval_data, scenario):
    """Returns metrics calculated for the given evaluation data (the contents of
    "evaluations_scenarios.json") for the given scenario for a single experiment
    with a single seed.

    Since each evaluation consists of a set of episodes, the returned metrics is
    a dict with the following contents:

        - reward_total_mean: the average of the total reward for all steps for
          each episode,
        - reward_total_std: the standard deviation of the total reward for each
          episode,
        - congested_total_mean: the mean of the number of congested steps for
          each episode,
        - congested_total_std: the standard deviation of the number of congested
          steps for each episode.
        - rejected_req_total_mean: the mean of the total number of rejected
          requests for each episode,
        - rejected_req_total_std: the standard deviation of the total number of
          rejected requests for each episode.
        """
    episodes = eval_data["scenarios"][scenario]
    num_episodes = len(episodes)

    # Initialize arrays.
    reward_total = np.empty(shape=num_episodes)
    steps_cong_total = np.empty(shape=num_episodes, dtype=np.int64)
    rejected_reqs_total = np.empty(shape=num_episodes, dtype=np.int64)
    rejected_reqs_total_percent = np.empty(shape=num_episodes)

    # Fill the arrays with data.
    idx = 0
    for episode in episodes:
        tmp_rewards_total = 0
        tmp_steps_cong_total = 0
        tmp_rejected_reqs_total = 0
        tmp_requests_total = 0

        for step in episode["evaluation_steps"]:
            tmp_rewards_total += step["reward"]
            tmp_steps_cong_total += step["obs_info"]["congested"]
            tmp_rejected_reqs_total += step["obs_info"]["actions"]["rejected"]
            tmp_requests_total += sum(step["obs_info"]["actions"].values())

        reward_total[idx] = tmp_rewards_total
        steps_cong_total[idx] = tmp_steps_cong_total
        rejected_reqs_total[idx] = tmp_rejected_reqs_total
        rejected_reqs_total_percent[idx] = tmp_rejected_reqs_total / tmp_requests_total * 100
        idx += 1

    assert idx == num_episodes, "The ndarrays have some elements not accessed"

    # Calculate the means and standard deviation.
    result = {
            "reward_total": {
                "mean": reward_total.mean(),
                "std": reward_total.std(),
                "values": reward_total,
                },
            "congested_total": {
                "mean": steps_cong_total.mean(),
                "std": steps_cong_total.std(),
                "values": steps_cong_total,
                },
            "rejected_reqs_total": {
                "mean": rejected_reqs_total.mean(),
                "std": rejected_reqs_total.std(),
                "values": rejected_reqs_total,
                "percent_mean": rejected_reqs_total_percent.mean(),
                "percent_std": rejected_reqs_total_percent.std(),
                "percent_values": rejected_reqs_total_percent,
                },
            }

    return result


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
    for scenario in data["scenarios"]:
        result = eval_single_exp_single_scenario(data, scenario)

        metrics["scenarios"][scenario] = result

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
        raw_data[scenario] = {
                "reward_total": {
                    "mean": np.empty(shape=num_exp),
                    "std": np.empty(shape=num_exp),
                    },
                "congested_total": {
                    "mean": np.empty(shape=num_exp),
                    "std": np.empty(shape=num_exp),
                    },
                "rejected_reqs_total": {
                    "mean": np.empty(shape=num_exp),
                    "std": np.empty(shape=num_exp),
                    "percent_mean": np.empty(shape=num_exp),
                    "percent_std": np.empty(shape=num_exp),
                    }
                }

    # Iterate over all experiment metrics and finish to fill the raw_data
    # arrays.
    idx = 0
    for exp_dir in exp_dirs:
        metrics_path = Path(exp_dir, "metrics.json")
        metrics = utils.json_to_dict(metrics_path)

        for (scenario, scenario_data) in metrics["scenarios"].items():
            raw_data[scenario]["reward_total"]["mean"][idx] = scenario_data["reward_total"]["mean"]
            raw_data[scenario]["reward_total"]["std"][idx] = scenario_data["reward_total"]["std"]
            raw_data[scenario]["congested_total"]["mean"][idx] = scenario_data["congested_total"]["mean"]
            raw_data[scenario]["congested_total"]["std"][idx] = scenario_data["congested_total"]["std"]
            raw_data[scenario]["rejected_reqs_total"]["mean"][idx] = scenario_data["rejected_reqs_total"]["mean"]
            raw_data[scenario]["rejected_reqs_total"]["std"][idx] = scenario_data["rejected_reqs_total"]["std"]
            raw_data[scenario]["rejected_reqs_total"]["percent_mean"][idx] = scenario_data["rejected_reqs_total"]["percent_mean"]
            raw_data[scenario]["rejected_reqs_total"]["percent_std"][idx] = scenario_data["rejected_reqs_total"]["percent_std"]

        idx += 1

    assert idx == num_exp, "The ndarrays have some unitialized data"

    # Now it is possible to calculate the aggregated metrics.
    aggr_metrics = {}
    for (scenario, scenario_data) in raw_data.items():

        aggr_metrics[scenario] = {
                "reward_total": {
                    "mean": scenario_data["reward_total"]["mean"].mean(),
                    "std": scenario_data["reward_total"]["std"].mean(),
                    },
                "congested_total": {
                    "mean": scenario_data["congested_total"]["mean"].mean(),
                    "std": scenario_data["congested_total"]["std"].mean(),
                    },
                "rejected_reqs": {
                    "mean": scenario_data["rejected_reqs_total"]["mean"].mean(),
                    "std": scenario_data["rejected_reqs_total"]["std"].mean(),
                    "percent_mean": scenario_data["rejected_reqs_total"]["percent_mean"].mean(),
                    "percent_std": scenario_data["rejected_reqs_total"]["percent_std"].mean(),
                    },
                }

    return aggr_metrics


def main(experiments_directory):
    experiments_path = Path(experiments_directory, "experiments.json")
    experiments = utils.json_to_dict(experiments_path)

    # Dictionary of aggregated metrics, stored as "metrics.json" in the main
    # experiments directory.
    aggr_metrics = {}

    for (algo, algo_value) in experiments.items():
        aggr_metrics[algo] = {}

        for (params, params_value) in algo_value.items():
            aggr_metrics[algo][params] = {}

            for (scenario, scenario_value) in params_value.items():
                aggr_metrics[algo][params][scenario] = {}
                all_done = False
                exp_dirs = []

                # Calculate metrics for a single experiment.
                for exp in scenario_value.values():
                    if not exp["done"]:
                        continue
                    logger.log(f"Calculating metrics for {exp['id']!r}")

                    exp_directory = Path(experiments_directory, exp["directory"])
                    exp_dirs.append(exp_directory)
                    metrics = calculate_metrics(exp_directory)

                    # Save the metrics to disk.
                    metrics_path = Path(exp_directory, "metrics.json")
                    utils.dict_to_json(metrics, metrics_path)

                    all_done = True

                # When calculating aggregate metrics, all sub-experiments must
                # be run.
                if not all_done:
                    continue

                exp_id = f"{algo}:{params}:{scenario}"
                logger.log(f"Calculating aggregate metrics for {exp_id!r}")

                aggr_metrics[algo][params][scenario]["id"] = exp_id
                aggr_metrics[algo][params][scenario]["trained_on"] = scenario

                # Calculate aggregate metrics for all the same experiments but
                # with different seeds.
                metrics = calculate_aggregate_metrics(exp_dirs)
                aggr_metrics[algo][params][scenario]["scenarios"] = metrics

    # Save the metrics data to disk.
    aggr_metrics_path = Path(experiments_directory, "metrics.json")
    utils.dict_to_json(aggr_metrics, aggr_metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="metrics",
                                     description="Metrics extraction from evaluation phase")

    parser.add_argument(dest="experiments_directory",
                        help="Directory with the experiments.json file")

    args = parser.parse_args()

    main(args.experiments_directory)
