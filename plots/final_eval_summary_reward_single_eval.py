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


def make_total_reward(eval_dir):
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    data = _get_data(eval_dir)
    env = plot_utils.get_env(eval_dir)

    fig = plt.figure(figsize=(7, 5), dpi=600, layout="constrained")
    ax = fig.subplots()

    # Limits for the y axis, both for total and single step.
    # Note the reward for each step is in [0, 1].
    _, top = env.reward_range
    max_reward = top * env.max_steps * env.agents
    bottom = 250  # Heuristically selected.
    ax.set_ylim(bottom=0, top=max_reward+10)

    ax.plot(data["reward_total"]["all"])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.set_ylabel("Ricompensa")
    ax.set_xlabel("Episodio")

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

    parser.add_argument("evaluation_dir", nargs="+", type=Path,
                        help="Evaluation directory (for the default evaluation, give the experiment directory")

    args = parser.parse_args()

    for eval_dir in args.evaluation_dir:
        make_total_reward(eval_dir.resolve())
