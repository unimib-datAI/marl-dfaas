from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

if __name__ == "__main__":
    import sys
    import os
    import argparse
    import matplotlib

    # Add the current directory (where Python is called) to sys.path. This
    # assumes this script is called in the project root directory, not inside
    # the directory where the script is.
    #
    # Required when calling this module directly as main.
    sys.path.append(os.getcwd())

import utils
from traffic_env import TrafficManagementEnv

from RL4CC.utilities.logger import Logger

logger = Logger(name="DFAAS-PLOTS", verbose=2)


def _make_plot(subplot, metrics, group_labels):
    """Creates a grouped bar chart using the given metrics and the given group
    labels. The plot is made on the given subplot object."""
    x = np.arange(len(group_labels))  # Label locations on X axis.
    width = 0.15  # Width of the bars
    multiplier = 0

    # For each seed, plot the bar at different locations on the X axis.
    for attribute, measurement in metrics.items():
        offset = width * multiplier
        bars = subplot.bar(x + offset, measurement, width, label=attribute)
        subplot.bar_label(bars, fmt="{:.0f}", padding=3)
        multiplier += 1

    # Set labels for the X axis (the train scenario/eval scenario pairs). The
    # labels must be centers across each group of bars.
    subplot.set_xticks(x + (width * multiplier)/2.5, group_labels)


def _get_metrics(exp_dir, algo):
    """Returns the metrics extracted from the metrics.json file in the given
    dfaas directory for the given algorithm."""
    metrics = utils.json_to_dict(Path(exp_dir, "metrics.json"))

    # Create the dictionaries that will contain the data, each key is the
    # choosen parameter and the value is a list of metrics.
    params = list(metrics[algo])
    reward_total_mean = {param: [] for param in params}
    congested_total_mean = {param: [] for param in params}
    rejected_reqs_total_percent_mean = {param: [] for param in params}

    # We assume every experiment is done, the caller must check it.
    for param in metrics[algo]:
        for train_scenario in metrics[algo][param]:
            eval_scenarios = metrics[algo][param][train_scenario]["scenarios"]

            for eval_scenario, data in eval_scenarios.items():
                reward_total_mean[param].append(data["reward_total"]["mean"])
                congested_total_mean[param].append(data["congested_total"]["mean"])
                rejected_reqs_total_percent_mean[param].append(data["rejected_reqs"]["percent_mean"])

    result = {
            "reward_total_mean": reward_total_mean,
            "congested_total_mean": congested_total_mean,
            "rejected_reqs_total_percent_mean": rejected_reqs_total_percent_mean
            }

    return result


def _abbreviate_scenario(scenario):
    """Returns the abbreviated form of the given scenario."""
    match scenario:
        case "scenario1":
            return "S1"
        case "scenario2":
            return "S2"
        case "scenario3":
            return "S3"
        case _:
            assert False, f"Unrecognized scenario {scenario!r}"


def make(exp_dir, algo):
    """Makes the evaluation plot over trained scenarios and parameters for the
    given algorithm and dfaas directory where the metrics.json file is located.
    Saves the plot to the specified dfaas directory."""
    logger.log(f"Making summary evaluation across parameters for {algo!r} algorithm")

    # Get the data.
    metrics = _get_metrics(exp_dir, algo)

    # Each parameter is a bar grouped together, we need to precalculate the
    # labels.
    group_labels = []
    for train_scenario in TrafficManagementEnv.get_scenarios():
        train = _abbreviate_scenario(train_scenario)
        for eval_scenario in TrafficManagementEnv.get_scenarios():
            eval = _abbreviate_scenario(eval_scenario)
            group_labels.append(f"Train {train}\nTest {eval}")

    # A figure with three plots (3 columns). Each column is for one metric.
    fig = plt.figure(figsize=(30, 7), dpi=300, layout="constrained")
    fig.suptitle(f"Evaluation of {algo!r} across experiments")
    axs = fig.subplots(ncols=3, nrows=1)

    # Average of total reward plot.
    _make_plot(axs[0], metrics["reward_total_mean"], group_labels)
    axs[0].set_ylabel('Reward')
    axs[0].set_title('Average of Total Reward')

    # Average of total congested steps plot.
    _make_plot(axs[1], metrics["congested_total_mean"], group_labels)
    axs[1].set_ylabel('Steps')
    axs[1].set_title('Average of Total Congested Steps')

    # Average percent of total rejected requests plot.
    _make_plot(axs[2], metrics["rejected_reqs_total_percent_mean"], group_labels)
    axs[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    axs[2].set_ylim(top=100)  # Percent.
    axs[2].set_ylabel('Percent Requests')
    axs[2].set_title('Average Percent of Total Rejected Requests')

    # Common settings for all plots.
    for ax in axs:
        # Visualize only the grid for Y axis, because the X axis is the #
        # scenarios.
        ax.grid(axis="y")
        ax.set_axisbelow(True)  # Place the grid behind the lines and bars.

        # Show the legend.
        ax.legend()

    # Save the plot.
    plot_path = Path(exp_dir, "plots", "evaluation", "params", f"{algo}.pdf")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    logger.log(f"{algo}: {plot_path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser(prog="algo_eval_params")

    parser.add_argument("experiment_directory",
                        help="DFaaS Experiment directory")
    parser.add_argument("-a", "--algorithm",
                        help="Which algorithm to make plot (example 'PPO')",
                        default=None)
    args = parser.parse_args()

    # Read and parse the experiments.json file.
    experiments_path = Path(args.experiment_directory, "experiments.json")
    experiments = utils.json_to_dict(experiments_path)

    for (algo, algo_values) in experiments.items():
        if args.algorithm is not None and algo != args.algorithm:
            continue

        # All sub-experiments must be done, otherwise can't make the general
        # plot for the algorithm.
        all_done = True
        for (params, params_values) in algo_values.items():
            for (scenario, scenario_value) in params_values.items():
                for exp in scenario_value.values():
                    if not exp["done"]:
                        all_done = False

        if all_done:
            make(args.experiment_directory, algo)
