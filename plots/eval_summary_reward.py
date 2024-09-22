# This Python script generates a graph showing the metrics related to the reward
# of the the evaluation phase.
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


def _get_data(eval_dir):
    data = {}

    # Read data from the given evaluation directory
    eval = dfaas_utils.parse_result_file(eval_dir / "evaluation.json")
    eval = eval[0]["evaluation"]
    agents = plot_utils.get_env(eval_dir).agent_ids

    data["agents"] = agents
    data["episodes"] = eval["episodes_this_iter"]

    reward_total = {}  # Total reward per episode.
    reward_total["all"] = eval["hist_stats"]["episode_reward"]
    for agent in data["agents"]:
        reward_total[agent] = eval["hist_stats"][f"policy_policy_{agent}_reward"]

    reward_step_avg = {}  # Average reward per step.
    for agent in data["agents"]:
        reward_step_avg[agent] = np.empty(data["episodes"], dtype=np.float32)

        for epi_idx in range(data["episodes"]):
            reward_step_avg[agent][epi_idx] = np.average(eval["hist_stats"]["reward"][epi_idx][agent])

    data["reward_total"] = reward_total
    data["reward_step_avg"] = reward_step_avg

    return data


def make(eval_dir):
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    data = _get_data(eval_dir)
    eval_env = plot_utils.get_env(eval_dir)

    fig = plt.figure(figsize=(25, 5), dpi=600, layout="constrained")
    title = "Evaluation reward"
    fig.suptitle(title)
    axs = fig.subplots(ncols=3)

    # Limits for the y axis, both for total and single step.
    bottom, top = eval_env.reward_range
    bottom_total = bottom * eval_env.max_steps
    top_total = top * eval_env.max_steps

    for agent in data["agents"]:
        axs[0].plot(data["reward_total"][agent], label=agent)
    axs[0].set_ylim(bottom=bottom_total, top=top_total)
    axs[0].set_title("Total reward per episode")
    # Show y-axis ticks every 20 reward points.
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(20))
    axs[0].legend()

    for agent in data["agents"]:
        axs[1].plot(data["reward_step_avg"][agent], label=agent)
    axs[1].set_ylim(bottom=bottom, top=top)
    axs[1].set_title("Average reward per step per episode")
    # Show y-axis ticks every .1 reward point.
    axs[1].yaxis.set_major_locator(ticker.MultipleLocator(.1))
    axs[1].legend()

    axs[2].plot(data["reward_total"]["all"])
    axs[2].set_ylim(bottom=bottom_total, top=top_total*len(data["agents"]))
    axs[2].set_title("Total reward per episode (all agents)")
    axs[2].yaxis.set_major_locator(ticker.MultipleLocator(50))

    # Common settings for the plots.
    for ax in axs.flat:
        ax.set_ylabel("Reward")

        ax.set_xlabel("Episodes")
        # Show x-axis ticks every X episodes.
        if data["episodes"] > 200:
            multiple = 50
        elif data["episodes"] > 10:
            multiple = 10
        else:
            multiple = 1
        ax.xaxis.set_major_locator(ticker.MultipleLocator(multiple))

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = plots_dir / "eval_summary_reward.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("evaluation_dir",
                        help="Evaluation directory (for the default evaluation, give the experiment directory")

    args = parser.parse_args()

    make(Path(args.evaluation_dir))
