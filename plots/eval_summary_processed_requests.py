# This script generates two plots showing the details of the processed requests in
# the episodes of the given evaluation (two plots: one in percentual and one
# with absolute values).
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
        data[agent] = {"local_reqs_ratio": np.empty(episodes),
                       "forward_reqs_ratio": np.empty(episodes),
                       "processed_reqs_ratio": np.empty(episodes),
                       "local_reqs": np.empty(episodes, dtype=np.int32),
                       "forward_reqs": np.empty(episodes, dtype=np.int32),
                       "input_reqs": np.empty(episodes, dtype=np.int32)}

    data["total"] = {"local_reqs_ratio": np.empty(episodes),
                     "forward_reqs_ratio": np.empty(episodes),
                     "processed_reqs_ratio": np.empty(episodes),
                     "local_reqs": np.empty(episodes, dtype=np.int32),
                     "forward_reqs": np.empty(episodes, dtype=np.int32),
                     "input_reqs": np.empty(episodes, dtype=np.int32)}

    data["steps"] = iter["episode_len_mean"]
    data["episodes"] = episodes

    for epi_idx in range(episodes):
        total_input_reqs = total_local_reqs = total_forward_reqs = 0

        for agent in env.agent_ids:
            input_reqs = np.sum(iter["hist_stats"]["observation_input_requests"][epi_idx][agent], dtype=np.int32)
            data[agent]["input_reqs"][epi_idx] = input_reqs
            total_input_reqs += input_reqs

            action_local = np.sum(iter["hist_stats"]["action_local"][epi_idx][agent], dtype=np.int32)
            action_forward = np.sum(iter["hist_stats"]["action_forward"][epi_idx][agent], dtype=np.int32)

            excess_local = np.sum(iter["hist_stats"]["excess_local"][epi_idx][agent], dtype=np.int32)
            forward_reject = np.sum(iter["hist_stats"]["excess_forward_reject"][epi_idx][agent], dtype=np.int32)

            local_reqs = action_local - excess_local
            data[agent]["local_reqs"][epi_idx] = local_reqs
            total_local_reqs += local_reqs
            local_ratio = local_reqs / input_reqs

            forward_reqs = action_forward - forward_reject
            data[agent]["forward_reqs"][epi_idx] = forward_reqs
            total_forward_reqs += forward_reqs
            forward_ratio = forward_reqs / input_reqs

            processed_ratio = local_ratio + forward_ratio

            data[agent]["local_reqs_ratio"][epi_idx] = local_ratio
            data[agent]["forward_reqs_ratio"][epi_idx] = forward_ratio
            data[agent]["processed_reqs_ratio"][epi_idx] = processed_ratio

        data["total"]["local_reqs_ratio"][epi_idx] = total_local_reqs / total_input_reqs
        data["total"]["forward_reqs_ratio"][epi_idx] = total_forward_reqs / total_input_reqs
        data["total"]["processed_reqs_ratio"][epi_idx] = (total_local_reqs + total_forward_reqs) / total_input_reqs

        data["total"]["local_reqs"][epi_idx] = total_local_reqs
        data["total"]["forward_reqs"][epi_idx] = total_forward_reqs
        data["total"]["input_reqs"][epi_idx] = total_input_reqs

    for agent in env.agent_ids:
        data[agent]["local_reqs_ratio_avg"] = np.average(data[agent]["local_reqs_ratio"])
        data[agent]["forward_reqs_ratio_avg"] = np.average(data[agent]["forward_reqs_ratio"])
        data[agent]["processed_reqs_ratio_avg"] = np.average(data[agent]["processed_reqs_ratio"])

    data["total"]["local_reqs_ratio_avg"] = np.average(data["total"]["local_reqs_ratio"])
    data["total"]["forward_reqs_ratio_avg"] = np.average(data["total"]["forward_reqs_ratio"])
    data["total"]["processed_reqs_ratio_avg"] = np.average(data["total"]["processed_reqs_ratio"])

    return data


def make_percentual(data, plots_dir, eval_env, train_env):
    fig = plt.figure(figsize=(20, 6), dpi=600, layout="constrained")
    title = f"Processed requests for evaluation (eval type {eval_env.input_requests_type}, train type {train_env.input_requests_type})"
    fig.suptitle(title)
    axs = fig.subplots(ncols=3)

    # Coordinates of the x-bar for the bar plots. Must be explicitly calculated
    # from the number of episodes.
    steps_x = np.arange(stop=data["episodes"])

    idx = 0
    for agent in eval_env.agent_ids:
        # Note: the ratio is in [0, 1], must be converted to percentual.
        axs[idx].bar(x=steps_x,
                     height=data[agent]["local_reqs_ratio"] * 100,
                     color="g",
                     label="Local")
        axs[idx].bar(x=steps_x,
                     height=data[agent]["forward_reqs_ratio"] * 100,
                     bottom=data[agent]["local_reqs_ratio"] * 100,
                     color="b",
                     label="Forward")

        avg_line = np.full(data["episodes"], data[agent]["processed_reqs_ratio_avg"] * 100)
        axs[idx].plot(avg_line, linewidth=2, color="r", label="Average")

        # Draw the average number directly above the average curve.
        axs[idx].annotate(f'{data[agent]["processed_reqs_ratio_avg"]:.2%}',
                          (0, data[agent]["processed_reqs_ratio_avg"] * 100 + 1),
                          bbox={"boxstyle": "square", "alpha": .7, "facecolor": "white"})

        axs[idx].set_title(f"Total processed requests ({agent})")

        idx += 1

    # Total (sum of the two nodes) plot.
    axs[idx].bar(x=steps_x,
                 height=data["total"]["local_reqs_ratio"] * 100,
                 color="g",
                 label="Local")
    axs[idx].bar(x=steps_x,
                 height=data["total"]["forward_reqs_ratio"] * 100,
                 bottom=data["total"]["local_reqs_ratio"] * 100,
                 color="b",
                 label="Forward")

    avg_line = np.full(data["episodes"], data["total"]["processed_reqs_ratio_avg"] * 100)
    axs[idx].plot(avg_line, linewidth=2, color="r", label="Average")

    axs[idx].annotate(f'{data["total"]["processed_reqs_ratio_avg"]:.2%}',
                      (0, data["total"]["processed_reqs_ratio_avg"] * 100 + 1),
                      bbox={"boxstyle": "square", "alpha": .7, "facecolor": "white"})

    axs[idx].set_title("Total processed requests (all agents)")

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


def make_absolute(data, plots_dir, eval_env, train_env):
    fig = plt.figure(figsize=(20, 6), dpi=600, layout="constrained")
    title = f"Processed requests for evaluation (eval type {eval_env.input_requests_type}, train type {train_env.input_requests_type})"
    fig.suptitle(title)
    axs = fig.subplots(ncols=3)

    # Coordinates of the x-bar for the bar plots. Must be explicitly calculated
    # from the number of episodes.
    steps_x = np.arange(stop=data["episodes"])

    idx = 0
    for agent in eval_env.agent_ids:
        axs[idx].bar(x=steps_x,
                     height=data[agent]["local_reqs"],
                     color="g",
                     label="Local")
        axs[idx].bar(x=steps_x,
                     height=data[agent]["forward_reqs"],
                     bottom=data[agent]["local_reqs"],
                     color="b",
                     label="Forward")

        axs[idx].plot(data[agent]["input_reqs"], linewidth=2, color="r", label="Total input requests")

        axs[idx].set_title(f"Total processed requests ({agent})")

        idx += 1

    # Total (sum of the two nodes) plot.
    base = np.zeros(data["episodes"], dtype=np.int32)
    for agent in eval_env.agent_ids:
        axs[idx].bar(x=steps_x,
                     height=data[agent]["local_reqs"],
                     bottom=base,
                     label=f"Local ({agent})")
        base = data[agent]["local_reqs"]

        axs[idx].bar(x=steps_x,
                     height=data[agent]["forward_reqs"],
                     bottom=base,
                     label=f"Forward ({agent})")
        base += data[agent]["forward_reqs"]

    axs[idx].plot(data["total"]["input_reqs"], linewidth=2, color="r", label="Total input requests")

    axs[idx].set_title("Total processed requests (all agents)")

    # Common settings for the plots.
    for ax in axs.flat:
        ax.set_xlabel("Episode")

        ax.set_ylabel("Number of requests")

        # Show x-axis ticks every X steps.
        if data["episodes"] <= 10:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

        ax.legend()

    # Save the plot.
    path = plots_dir / "eval_summary_processed_requests_abs.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


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

    make_percentual(data, plots_dir, eval_env, train_env)

    make_absolute(data, plots_dir, eval_env, train_env)


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("evaluation_dir", nargs="+", type=Path,
                        help="Evaluation directory (for the default evaluation, give the experiment directory")

    args = parser.parse_args()

    for eval_dir in args.evaluation_dir:
        make(eval_dir.resolve())
