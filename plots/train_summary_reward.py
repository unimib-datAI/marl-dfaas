# This Python script generates a graph showing the metrics related to the reward
# of the the training phase.
import sys
import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

# Add the current directory (where Python is called) to sys.path. This is
# required to load modules in the project root directory (like dfaas_utils.py).
sys.path.append(os.getcwd())

import dfaas_utils


def _get_agents():
    """Returns the agent IDs from the given list of iterations. Assumes at least
    one iteration and one episode for iteration."""
    return ["node_0", "node_1"]


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
    agents = _get_agents()
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
    data["iterations"] = len(iters)
    data["reward_total_avg"] = reward_total_avg
    data["reward_step_avg"] = reward_step_avg

    return data


def make(exp_dir):
    exp_dir = dfaas_utils.to_pathlib(exp_dir)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _get_data(exp_dir)

    fig = plt.figure(figsize=(17, 5), dpi=600, layout="constrained")
    axs = fig.subplots(ncols=2)

    # For the ylim, the total reward for one episode cannot exceed the possible
    # max and min of one episode. The limits ensure a bit of space for both
    # bottom and top.
    #
    # TODO: extract dynamically the values from the env.
    bottom = 0.0 - 1
    top = 100.0 + 1

    for agent in data["agents"]:
        axs[0].plot(data["reward_total_avg"][agent], label=agent)
    axs[0].set_ylim(bottom=bottom, top=top)
    axs[0].set_title("Average total reward per episode")
    axs[0].set_ylabel("Total reward")
    # Show y-axis ticks every 10 reward points.
    # TODO: make dynamic.
    axs[0].set_yticks(np.arange(0, 100+1, 10))
    axs[0].legend()

    for agent in data["agents"]:
        axs[1].plot(data["reward_step_avg"][agent], label=agent)
    # Reward range.
    # TODO: make dynamic.
    axs[1].set_ylim(bottom=.0-.1, top=1.0+.1)
    axs[1].set_title("Average reward per step in an episode")
    axs[1].set_ylabel("Average reward")
    # Show y-axis ticks every .1 reward point.
    # TODO: make dynamic.
    axs[1].set_yticks(np.arange(start=.0, stop=1.+.1, step=.1))
    axs[1].legend()

    # Common settings for the plots.
    for ax in axs.flat:
        ax.set_xlabel("Iteration")
        # Because the plots shares the x-axis, only the fourth plots will
        # show the ticks, but I want to write the ticks also for the first
        # three plots.
        ax.tick_params(axis="x", labelbottom=True)
        # Show x-axis ticks every 10 iterations.
        ax.set_xticks(np.arange(0, data["iterations"]+1, 10))

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

    parser.add_argument(dest="exp_dir",
                        help="DFaaS experiment directory")

    args = parser.parse_args()

    make(args.exp_dir)
