from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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


def _make_plot(subplot, metrics, scenarios):
    x = np.arange(len(scenarios))  # Label locations on X axis.
    width = 0.15  # Width of the bars
    multiplier = 0

    # For each seed, plot the bar at different locations on the X axis.
    for attribute, measurement in metrics.items():
        offset = width * multiplier
        bars = subplot.bar(x + offset, measurement, width, label=attribute)
        subplot.bar_label(bars, padding=1)
        multiplier += 1

    # Set labels for the X axis (the scenarios). The labels must be centers
    # across each group of bars.
    subplot.set_xticks(x + (width * multiplier)/2.5, scenarios)

    # Visualize only the grid for Y axis, because the X axis is the scenarios.
    subplot.grid(axis="y")

    # Make sure that the labels on the X axis will be shown, even if this
    # subplot shares the axis with others (see main function).
    subplot.tick_params(axis="x", labelbottom=True)

    # Show the legend.
    subplot.legend()


def _make_boxplot(subplot, metrics, scenarios):
    seeds = len(metrics)

    # subplot.set_xlim(0, 30)
    x = np.arange(len(scenarios))
    width = 0.15  # the width of the bars
    multiplier = 0

    for seed in range(seeds):
        offset = width * multiplier
        subplot.boxplot(metrics[seed], positions=x + offset, widths=width)
        multiplier += 1

    # Set labels for the X axis (the scenarios). The labels must be centers
    # across each group of bars.
    subplot.set_xticks(x + (width * multiplier)/2.5, scenarios)

    # Visualize grid for both axis.
    subplot.grid(axis="both")


def make(exp_dirs, exp_id, res_dir):
    logger.log(f"Making summary evaluation across seeds plots for {exp_id!r}")

    seeds = len(exp_dirs)

    scenarios = TrafficManagementEnv.get_scenarios()

    # Get the data.
    reward_total_mean = {f"Seed nr. {seed}": [] for seed in range(seeds)}
    congested_total_mean = {f"Seed nr. {seed}": [] for seed in range(seeds)}
    rejected_reqs_total_mean = {f"Seed nr. {seed}": [] for seed in range(seeds)}
    reward_total = [[] for _ in range(seeds)]
    congested_total = [[] for _ in range(seeds)]
    rejected_reqs_total = [[] for _ in range(seeds)]
    idx = 0
    for seed in reward_total_mean:
        metrics = utils.json_to_dict(Path(exp_dirs[idx], "metrics.json"))

        # Initialize sub-lists for this seed.
        reward_total[idx] = [[] for _ in range(len(scenarios))]
        congested_total[idx] = [[] for _ in range(len(scenarios))]
        rejected_reqs_total[idx] = [[] for _ in range(len(scenarios))]

        # The total are indexed by integers, whereas the means by dictionaries.
        scen_idx = 0
        for scenario, scenario_data in metrics["scenarios"].items():
            reward_total_mean[seed].append(scenario_data["reward_total_mean"])
            congested_total_mean[seed].append(scenario_data["congested_total_mean"])
            rejected_reqs_total_mean[seed].append(scenario_data["rejected_reqs_total_mean"])

            reward_total[idx][scen_idx] = scenario_data["reward_total"]
            congested_total[idx][scen_idx] = scenario_data["congested_total"]
            rejected_reqs_total[idx][scen_idx] = scenario_data["rejected_reqs_total"]
            scen_idx += 1

        idx += 1

    # A figure with six plots (2 rows, 3 columns). Each column is for one
    # metric, the upper plot is the mean, the other plot is for the
    # distribution.
    fig = plt.figure(figsize=(30, 14), dpi=300, layout="constrained")
    fig.suptitle(f"Evaluation of {exp_id} across seeds")
    axs = fig.subplots(ncols=3, nrows=2, sharex="col")

    # Mean of total reward plot.
    _make_plot(axs[0, 0], reward_total_mean, scenarios)
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].set_title('Mean Total Reward')

    # Distribution of the total reward plot.
    _make_boxplot(axs[1, 0], reward_total, scenarios)
    axs[1, 0].set_ylabel('Reward')
    axs[1, 0].set_title('Distribution of Total Reward')

    # Mean of total congested steps plot.
    _make_plot(axs[0, 1], congested_total_mean, scenarios)
    axs[0, 1].set_ylabel('Steps')
    axs[0, 1].set_title('Mean Total Congested Steps')

    # Distribution of the total congested steps plot.
    _make_boxplot(axs[1, 1], congested_total, scenarios)
    axs[1, 1].set_ylabel('Steps')
    axs[1, 1].set_title('Distribution of Total Congsted Steps')

    # Mean of total rejected requests plot.
    _make_plot(axs[0, 2], rejected_reqs_total_mean, scenarios)
    axs[0, 2].set_ylabel('Requests')
    axs[0, 2].set_title('Mean Total Rejected Requests')

    # Distribution of the total rejected requests plot.
    _make_boxplot(axs[1, 2], rejected_reqs_total, scenarios)
    axs[1, 2].set_ylabel('Requests')
    axs[1, 2].set_title('Distribution of Total Rejected Requests')

    # Common settings for all plots.
    for ax in axs.flat:
        ax.set_axisbelow(True)  # Place the grid behind the lines and bars.

    # Save the plot.
    plot_path = Path(res_dir, "plots", "evaluation", "seeds", f"{exp_id}.pdf")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    logger.log(f"{exp_id}: {plot_path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser(prog="multiple_exp_eval_seeds")

    parser.add_argument("experiment_directory",
                        help="DFaaS Experiment directory")
    parser.add_argument("--experiment_id", "-e",
                        help="Which main experiment make plots (example 'PPO:standard:scenario1')",
                        default=None)
    args = parser.parse_args()

    # Read and parse the experiments.json file.
    experiments_path = Path(args.experiment_directory, "experiments.json")
    experiments = utils.json_to_dict(experiments_path)

    # Directory that will contains plots related to macro-experiments (not the
    # single experiment).
    Path(args.experiment_directory, "plots", "training").mkdir(parents=True, exist_ok=True)

    # Make plots for all experiments.
    for (algo, algo_values) in experiments.items():
        for (params, params_values) in algo_values.items():
            for (scenario, scenario_value) in params_values.items():
                exp_id = f"{algo}:{params}:{scenario}"
                if args.experiment_id is not None and exp_id != args.experiment_id:
                    continue

                exp_dirs = []
                all_done = True
                for exp in scenario_value.values():
                    if not exp["done"]:
                        all_done = False
                        continue
                    exp_dirs.append(Path(args.experiment_directory, exp["directory"]))

                if all_done:
                    make(exp_dirs, exp_id, args.experiment_directory)
