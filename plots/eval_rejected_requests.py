# This script generates a plot showing the details of the rejected requests in
# the episodes in the evaluation process.
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
    return ["node_0", "node_1"]  # FIXME: Get dynamically.


def _get_data(exp_dir):
    # Read data from the experiment directory.
    iters = dfaas_utils.parse_result_file(exp_dir / "final_evaluation.json")
    iter = iters[0]["evaluation"]  # There is only one iteration.
    episodes = iter["episodes_this_iter"]
    agents = _get_agents()

    data = {}
    for agent in agents:
        data[agent] = {}
        data[agent]["reject_ratios"] = np.empty(episodes)

    for epi_idx in range(episodes):
        for agent in agents:
            input_reqs = np.sum(iter["hist_stats"]["observation_input_requests"][epi_idx][agent], dtype=np.int32)
            action_reject = np.sum(iter["hist_stats"]["action_reject"][epi_idx][agent], dtype=np.int32)
            excess_local = np.sum(iter["hist_stats"]["excess_local"][epi_idx][agent], dtype=np.int32)
            forward_reject = np.sum(iter["hist_stats"]["excess_forward_reject"][epi_idx][agent], dtype=np.int32)

            rejected_reqs = action_reject + excess_local + forward_reject
            reject_ratio = rejected_reqs / input_reqs

            data[agent]["reject_ratios"][epi_idx] = reject_ratio

    for agent in agents:
        data[agent]["reject_ratio_avg"] = np.average(data[agent]["reject_ratios"])

    data["steps"] = iter["episode_len_mean"]
    data["episodes"] = episodes

    return data


def make(exp_dir):
    exp_dir = dfaas_utils.to_pathlib(exp_dir)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _get_data(exp_dir)

    fig = plt.figure(figsize=(10, 10), dpi=600, layout="constrained")
    fig.suptitle("Rejected requests in evaluation phase")
    axs = fig.subplots(nrows=2)

    # Coordinates of the x-bar for the bar plots. Must be explicitly calculated
    # from the number of episodes.
    steps_x = np.arange(stop=data["episodes"])

    idx = 0
    for agent in _get_agents():
        # Note: the ratio is in [0, 1], must be converted to percentual.
        axs[idx].bar(x=steps_x, height=data[agent]["reject_ratios"] * 100, label=agent)

        avg_line = np.full(data["episodes"], data[agent]["reject_ratio_avg"] * 100)
        axs[idx].plot(avg_line, linewidth=2, color="r", label="Average")

        # Draw the average number directly above the average curve.
        axs[idx].annotate(f'{data[agent]["reject_ratio_avg"]:.2%}',
                          (0, data[agent]["reject_ratio_avg"] * 100 + 1),
                          bbox={"boxstyle": "square", "alpha": .7, "facecolor": "white"})

        axs[idx].set_title(f"Total rejected requests ({agent})")

        idx += 1

    # Common settings for the plots.
    for ax in axs.flat:
        ax.set_xlabel("Episode")

        ax.set_ylabel("Percentage")
        ax.set_ylim(bottom=0, top=100)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        ax.set_yticks(np.arange(0, 100+1, 10))

        # Show x-axis ticks every X steps.
        ax.set_xticks(np.arange(0, data["episodes"]+1, 50))

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

        ax.legend()

    # Save the plot.
    path = plots_dir / "eval_rejected_requests.pdf"
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
