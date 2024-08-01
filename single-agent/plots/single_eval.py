from pathlib import Path

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


def make(exp_dir, exp_id):
    """Creates a single plot for the given experiment for evaluation across
    multiple scenarios.

    Reads data from the metrics.json file."""
    metrics = utils.json_to_dict(Path(exp_dir, "metrics.json"))

    # A figure with six sub-plots. Each row is for a metric (Total Reward, Total
    # Congested Steps, and Total Rejected Requests), with two columns: one for
    # the average total (across multiple evaluation episodes) and one for the
    # standard deviation.
    fig = plt.figure(figsize=(10, 15), dpi=600, layout="constrained")
    fig.suptitle(f"Evaluation of {exp_id} (averages across eval. episodes)")
    axs = fig.subplots(ncols=2, nrows=3)

    # Fixed scenarios to be placed on the x-axis.
    scenarios = ["Scenario 1", "Scenario 2", "Scenario 3"]

    # Data retrieved from the metrics.json file. Each list has a length of three
    # (the number of scenarios).
    reward_total_mean = []
    reward_total_std = []
    congested_total_mean = []
    congested_total_std = []
    rejected_reqs_total_percent_mean = []
    rejected_reqs_total_percent_std = []
    for scenario in TrafficManagementEnv.get_scenarios():
        reward_total_mean.append(metrics["scenarios"][scenario]["reward_total"]["mean"])
        reward_total_std.append(metrics["scenarios"][scenario]["reward_total"]["std"])
        congested_total_mean.append(metrics["scenarios"][scenario]["congested_total"]["mean"])
        congested_total_std.append(metrics["scenarios"][scenario]["congested_total"]["std"])
        rejected_reqs_total_percent_mean.append(metrics["scenarios"][scenario]["rejected_reqs_total"]["percent_mean"])
        rejected_reqs_total_percent_std.append(metrics["scenarios"][scenario]["rejected_reqs_total"]["percent_std"])

    bars = axs[0, 0].bar(scenarios, reward_total_mean)
    axs[0, 0].bar_label(bars, fmt="{:.2f}", padding=1)
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].set_title("Average total reward")

    bars = axs[0, 1].bar(scenarios, reward_total_std)
    axs[0, 1].bar_label(bars, fmt="{:.2f}", padding=1)
    axs[0, 1].set_ylabel("Reward")
    axs[0, 1].set_title("SD total reward")

    bars = axs[1, 0].bar(scenarios, congested_total_mean, color="g")
    axs[1, 0].bar_label(bars, fmt="{:.2f}", padding=1)
    axs[1, 0].set_ylabel("Steps")
    axs[1, 0].set_title("Average total congested steps")

    bars = axs[1, 1].bar(scenarios, congested_total_std, color="g")
    axs[1, 1].bar_label(bars, fmt="{:.2f}", padding=1)
    axs[1, 1].set_ylabel("Steps")
    axs[1, 1].set_title("SD total congested steps")

    bars = axs[2, 0].bar(scenarios, rejected_reqs_total_percent_mean, color="r")
    axs[2, 0].bar_label(bars, fmt="{:.2f} %", padding=1)
    axs[2, 0].set_ylim(top=100)  # Set Y axis range from 0 to 100 (percent).
    # Set the Y axis formatter.
    axs[2, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    axs[2, 0].set_ylabel("Percent Requests")
    axs[2, 0].set_title("Average percent of total rejected requests")

    bars = axs[2, 1].bar(scenarios, rejected_reqs_total_percent_std, color="r")
    axs[2, 1].bar_label(bars, fmt="{:.2f} %", padding=1)
    axs[2, 1].set_ylabel("Percent Requests")
    axs[2, 1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    axs[2, 1].set_title("SD average percent of total rejected requests")

    # Common settings for all plots.
    for ax in axs.flat:
        ax.set_xlabel("Scenarios evaluated")

        ax.grid(which="both")
        ax.set_axisbelow(True)  # Place the grid behind the lines and bars.

    path = Path(exp_dir, "plots", "evaluation.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.log(f"{exp_id}: {path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser(prog="single_eval")

    parser.add_argument(dest="experiment_directory",
                        help="Experiment directory")
    parser.add_argument(dest="experiment_id",
                        help="Experiment ID")

    args = parser.parse_args()

    make(Path(args.experiment_directory, args.experiment_id), args.experiment_id)
