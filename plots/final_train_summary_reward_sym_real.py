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


def _average_reward_step(iter, agent):
    """Returns the average reward per step for the given iteration and agent."""
    episodes = iter["episodes_this_iter"]

    tmp = np.empty(episodes, dtype=np.float32)
    for epi_idx in range(episodes):
        tmp[epi_idx] = np.average(iter["hist_stats"]["reward"][epi_idx][agent])

    return np.average(tmp)


def _get_data(exp_dir):
    data = {}

    # Read data from experiment directory.
    iters = dfaas_utils.parse_result_file(exp_dir / "result.json")
    agents = plot_utils.get_env(exp_dir).agent_ids

    data["agents"] = agents
    data["iterations"] = len(iters)
    data["episodes"] = iters[0]["episodes_this_iter"]

    reward_total_avg = {}  # Average total reward per episode.
    reward_step_avg = {}  # Average reward per step.

    reward_total_avg["all"] = np.empty(data["iterations"], dtype=np.float32)
    for agent in data["agents"]:
        reward_total_avg[agent] = np.empty(data["iterations"], dtype=np.float32)
        reward_step_avg[agent] = np.empty(data["iterations"], dtype=np.float32)

    for iter in iters:
        iter_idx = iter["training_iteration"] - 1  # Indexes start from zero.

        reward_total_avg["all"][iter_idx] = np.average(iter["hist_stats"]["episode_reward"])

        for agent in data["agents"]:
            reward_total_avg[agent][iter_idx] = np.average(iter["hist_stats"][f"policy_policy_{agent}_reward"])
            reward_step_avg[agent][iter_idx] = _average_reward_step(iter, agent)

    data["reward_total_avg"] = reward_total_avg
    data["reward_step_avg"] = reward_step_avg

    return data


def make(exp_dirs, out):
    data_ppo = _get_data(exp_dirs[0])
    data_ppo_cc = _get_data(exp_dirs[1])
    env = plot_utils.get_env(exp_dirs[0])
    train_type = dfaas_utils.json_to_dict(exp_dirs[0] / "env_config.json")
    train_type = train_type["input_requests_type"]

    fig = plt.figure(figsize=(7, 5), dpi=600, layout="constrained")
    ax = fig.subplots()

    # Limits for the y axis, both for total and single step.
    # Note the reward for each step is in [0, 1].
    _, top = env.reward_range
    max_reward = top * env.max_steps * env.agents
    bottom = 160  # Heuristically selected.
    ax.set_ylim(bottom=160, top=max_reward+10)

    ax.plot(data_ppo["reward_total_avg"]["all"], label="PPO")
    ax.plot(data_ppo_cc["reward_total_avg"]["all"], label="PPO-CC")

    # In the Y axis show ticks with interval of 50 reward points.
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

    ax.set_ylabel("Ricompensa", fontsize="large")
    ax.set_xlabel("Iterazione", fontsize="large")

    # Show x-axis ticks every X iterations.
    multiple = 50 if data_ppo["iterations"] > 200 else 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(multiple))

    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.
    ax.legend()

    # Save the plot.
    path = out / f"train_{train_type}_summary_reward.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    exp_dirs = [Path(f"results/final/DFAAS-MA_SYM_500_real"),
                Path(f"results/final/DFAAS-MA_SYM_500_cc_real")]

    out = Path("results/final/plots")

    make(exp_dirs, out)
