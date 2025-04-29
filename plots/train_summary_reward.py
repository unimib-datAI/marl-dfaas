# This Python script generates a plot showing the metrics related to the reward
# of the the training phase.
from pathlib import Path
import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add the current directory (where Python is called) to sys.path. This is
# required to load modules in the project root directory (like dfaas_utils.py).
sys.path.append(os.getcwd())

import dfaas_env
import dfaas_utils
import plot_utils


def _average_reward_step(iter, agent):
    """Returns the average reward per step for the given iteration and agent."""
    episodes = iter["env_runners"]["episodes_this_iter"]

    tmp = np.empty(episodes, dtype=np.float32)
    for epi_idx in range(episodes):
        tmp[epi_idx] = np.average(iter["env_runners"]["hist_stats"]["reward"][epi_idx][agent])

    return np.average(tmp)


def _get_data(exp_dir):
    data = {}

    # Read data from experiment directory.
    iters = dfaas_utils.parse_result_file(exp_dir / "result.json")
    agents = plot_utils.get_env(exp_dir).agents

    data["agents"] = agents
    data["iterations"] = len(iters)
    data["episodes"] = iters[0]["env_runners"]["episodes_this_iter"]

    reward_total_avg = {}  # Average total reward per episode.
    reward_step_avg = {}  # Average reward per step.

    reward_total_avg["all"] = np.empty(data["iterations"], dtype=np.float32)
    for agent in data["agents"]:
        reward_total_avg[agent] = np.empty(data["iterations"], dtype=np.float32)
        reward_step_avg[agent] = np.empty(data["iterations"], dtype=np.float32)

    # For each iteration, get the average reward, since there are multiple
    # episodes played in each iteration.
    for iter in iters:
        # Index starts from one in log files, but Python list from zero.
        iter_idx = iter["training_iteration"] - 1

        reward_total_avg["all"][iter_idx] = np.average(iter["env_runners"]["hist_stats"]["episode_reward"])

        for agent in data["agents"]:
            reward_total_avg[agent][iter_idx] = np.average(
                iter["env_runners"]["hist_stats"][f"policy_policy_{agent}_reward"]
            )
            reward_step_avg[agent][iter_idx] = _average_reward_step(iter, agent)

    data["reward_total_avg"] = reward_total_avg
    data["reward_step_avg"] = reward_step_avg

    return data


def make(exp_dir):
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    data = _get_data(exp_dir)
    env = plot_utils.get_env(exp_dir)

    fig = plt.figure(figsize=(25, 6), dpi=600, layout="constrained")
    title = f"Training reward ({data['episodes']} episodes for each iteration)"
    fig.suptitle(title)
    axs = fig.subplots(ncols=3)

    # Limits for the y axis, both for total and single step.
    bottom, top = env.reward_range
    bottom_total = bottom * env.max_steps
    top_total = top * env.max_steps

    for agent in data["agents"]:
        axs[0].plot(data["reward_total_avg"][agent], label=agent)
    axs[0].set_ylim(bottom=bottom_total, top=top_total)
    axs[0].set_title("Average total reward per episode")
    # Show y-axis ticks every 20 reward points.
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(20))
    axs[0].legend()

    for agent in data["agents"]:
        axs[1].plot(data["reward_step_avg"][agent], label=agent)
    axs[1].set_ylim(bottom=bottom, top=top)
    axs[1].set_title("Average reward per step")
    # Show y-axis ticks every .1 reward point.
    axs[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    axs[1].legend()

    axs[2].plot(data["reward_total_avg"]["all"])
    axs[2].set_ylim(bottom=bottom_total, top=top_total * len(data["agents"]))
    axs[2].set_title("Average total reward per episode (all agents)")
    axs[2].yaxis.set_major_locator(ticker.MultipleLocator(50))

    # Common settings for the plots.
    for ax in axs.flat:
        ax.set_ylabel("Reward")

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
    types = ["real", "sinusoidal", "normal"]

    # Create parser and parse command line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("experiment_dir", nargs="+", type=Path, help="DFaaS experiment directory")

    args = parser.parse_args()

    for exp_dir in args.experiment_dir:
        make(exp_dir.resolve())
