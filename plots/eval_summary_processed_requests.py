# This script generates a plot showing the details of the processed requests in
# the episodes of the given evaluation.
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
    # Read data from the evaluation directory.
    iters = dfaas_utils.parse_result_file(eval_dir / "evaluation.json")
    iter = iters[0]["evaluation"]  # There is only one iteration.
    episodes = iter["episodes_this_iter"]
    env = plot_utils.get_env(eval_dir)

    data = {}
    for agent in env.agent_ids:
        data[agent] = {"processed_reqs_ratio": np.empty(episodes)}

    data["steps"] = iter["episode_len_mean"]
    data["episodes"] = episodes

    for epi_idx in range(episodes):
        for agent in env.agent_ids:
            input_reqs = np.sum(iter["hist_stats"]["observation_input_requests"][epi_idx][agent], dtype=np.int32)

            action_local = np.sum(iter["hist_stats"]["action_local"][epi_idx][agent], dtype=np.int32)
            action_forward = np.sum(iter["hist_stats"]["action_forward"][epi_idx][agent], dtype=np.int32)

            excess_local = np.sum(iter["hist_stats"]["excess_local"][epi_idx][agent], dtype=np.int32)
            forward_reject = np.sum(iter["hist_stats"]["excess_forward_reject"][epi_idx][agent], dtype=np.int32)

            processed_reqs = (action_local - excess_local)
            processed_reqs += (action_forward - forward_reject)

            processed_ratio = processed_reqs / input_reqs

            data[agent]["processed_reqs_ratio"][epi_idx] = processed_ratio

    for agent in env.agent_ids:
        data[agent]["processed_reqs_ratio_avg"] = np.average(data[agent]["processed_reqs_ratio"])

    return data


def make(eval_dir):
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    data = _get_data(eval_dir)
    eval_env = plot_utils.get_env(eval_dir)

    if not (eval_dir.parent / "exp_config.json").exists():
        # If the user give the experiment directory instead of the evaluation
        # directory (a subdirectory of experiment directory), do not load a new
        # environment.
        train_env = eval_env
    else:
        train_env = plot_utils.get_env(eval_dir.parent)  # Experiment directory.

    fig = plt.figure(figsize=(10, 10), dpi=600, layout="constrained")
    title = f"Processed requests for evaluation (eval type {eval_env.input_requests_type}, train type {train_env.input_requests_type})"
    fig.suptitle(title)
    axs = fig.subplots(nrows=2)

    # Coordinates of the x-bar for the bar plots. Must be explicitly calculated
    # from the number of episodes.
    steps_x = np.arange(stop=data["episodes"])

    idx = 0
    for agent in eval_env.agent_ids:
        # Note: the ratio is in [0, 1], must be converted to percentual.
        axs[idx].bar(x=steps_x, height=data[agent]["processed_reqs_ratio"] * 100, label=agent)

        avg_line = np.full(data["episodes"], data[agent]["processed_reqs_ratio_avg"] * 100)
        axs[idx].plot(avg_line, linewidth=2, color="r", label="Average")

        # Draw the average number directly above the average curve.
        axs[idx].annotate(f'{data[agent]["processed_reqs_ratio_avg"]:.2%}',
                          (0, data[agent]["processed_reqs_ratio_avg"] * 100 + 1),
                          bbox={"boxstyle": "square", "alpha": .7, "facecolor": "white"})

        axs[idx].set_title(f"Total processed requests ({agent})")

        idx += 1

    # Common settings for the plots.
    for ax in axs.flat:
        ax.set_xlabel("Episode")

        ax.set_ylabel("Percentage")
        ax.set_ylim(bottom=0, top=100)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        # Show percent tick every 10%.
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

        # Show x-axis ticks every X steps.
        if data["episodes"] <= 10:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

        ax.legend()

    # Save the plot.
    path = plots_dir / "eval_summary_processed_requests.pdf"
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
