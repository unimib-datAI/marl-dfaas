# This Python script generates a graph showing the metrics related to the reward
# of the the training phase.
from pathlib import Path
import sys
import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add the current directory (where Python is called) to sys.path. This is
# required to load modules in the project root directory (like dfaas_utils.py).
sys.path.append(os.getcwd())

import dfaas_utils
import plot_utils


def _get_num_episodes(iters):
    """Returns the number of episodes in each iteration.

    The value is extracted from the given iter data. It is assumed that each
    iteration has the same number of episodes and that there is at least one
    iteration in the given data."""
    return iters[0]["episodes_this_iter"]


def _get_data(exp_dir):
    data = {}

    # Read data from experiment directory.
    iters = dfaas_utils.parse_result_file(exp_dir / "result.json")
    metrics = dfaas_utils.json_to_dict(exp_dir / "metrics-result.json")
    agents = plot_utils.get_env(exp_dir).agent_ids
    episodes = _get_num_episodes(iters)

    # Average total reward per episode for each iteration.
    reward_total_avg = {}

    # Average reward per step in an episode (average across episodes in each
    # iteration).
    reward_step_avg = {}

    # Average percent of rejected requests per iteration.
    percent_reject_reqs_excess = {}

    # Percentage of local requests exceeding local buffer per step (average
    # across steps for each episode and episodes for each iteration)
    percent_local_reqs_excess = {}

    percent_forward_reqs_excess = {}

    # The data is grouped in agents.
    for agent in agents:
        reward_total_avg[agent] = np.empty(len(iters), dtype=np.float32)
        reward_step_avg[agent] = np.empty(len(iters), dtype=np.float32)
        percent_reject_reqs_excess[agent] = np.empty(len(iters), dtype=np.float32)
        percent_local_reqs_excess[agent] = np.empty(len(iters), dtype=np.float32)
        percent_forward_reqs_excess[agent] = np.empty(len(iters), dtype=np.float32)

    for iter in range(len(iters)):
        for agent in agents:
            reward_total_avg[agent][iter] = np.average(iters[iter]["hist_stats"][f"policy_policy_{agent}_reward"])

            tmp = np.empty(episodes, dtype=np.float32)
            for epi_id in range(episodes):
                tmp[epi_id] = metrics["reward_average_per_step"][iter][epi_id][agent]

            reward_step_avg[agent][iter] = np.average(tmp)

    data["agents"] = agents
    data["reward_total_avg"] = reward_total_avg
    data["reward_step_avg"] = reward_step_avg
    data["iterations"] = len(iters)

    return data


def make(exp_dir):
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    data = _get_data(exp_dir)
    env = plot_utils.get_env(exp_dir)

    fig = plt.figure(figsize=(17, 5), dpi=600, layout="constrained")
    axs = fig.subplots(ncols=2)

    # Limits for the y axis, both for total and single step.
    bottom, top = env.reward_range
    bottom_total = bottom * env.max_steps
    top_total = top * env.max_steps

    for agent in data["agents"]:
        axs[0].plot(data["reward_total_avg"][agent], label=agent)
    axs[0].set_ylim(bottom=bottom_total, top=top_total)
    axs[0].set_title("Average total reward per episode")
    axs[0].set_ylabel("Total reward")
    # Show y-axis ticks every 20 reward points.
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(20))

    for agent in data["agents"]:
        axs[1].plot(data["reward_step_avg"][agent], label=agent)
    axs[1].set_ylim(bottom=bottom, top=top)
    axs[1].set_title("Average reward per step in an episode")
    axs[1].set_ylabel("Average reward")
    # Show y-axis ticks every .1 reward point.
    axs[1].yaxis.set_major_locator(ticker.MultipleLocator(.1))

    # Common settings for the plots.
    for ax in axs.flat:
        ax.legend()

        ax.set_xlabel("Iteration")
        # Show x-axis ticks every X iterations.
        multiple = 50 if data["iterations"] > 200 else 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(multiple))

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = plots_dir / "train_summary_reward.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(dest="experiment_dir",
                        help="DFaaS experiment directory")

    args = parser.parse_args()

    make(Path(args.experiment_dir))
