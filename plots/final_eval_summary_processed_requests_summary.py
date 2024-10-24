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


def _processable_requests_episode(data, epi_idx):
    eval = data["hist_stats"]
    input_reqs_epi = processable_reqs_epi = 0

    # Only environments with the same queue capacity are supported for this
    # script.
    #queue_capacity = np.unique(list(env.queue_capacity_max.values()))
    #assert queue_capacity.size == 1, "Environments with different queue capacity size for the agents are not supported"
    #total_capacity = queue_capacity.item() * env.agents
    total_capacity = 100 * 2

    for step in range(288):
        input_reqs_step = 0  # Input requests for this step.
        for agent in ["node_0", "node_1"]:
            input_reqs_step += eval["observation_input_requests"][epi_idx][agent][step]

        # Processable requests for this step.
        if input_reqs_step <= total_capacity:
            processable_reqs_step = input_reqs_step
        else:
            processable_reqs_step = total_capacity

        input_reqs_epi += input_reqs_step
        processable_reqs_epi += processable_reqs_step

    assert processable_reqs_epi <= input_reqs_epi
    return processable_reqs_epi


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
                       "processable_reqs": np.empty(episodes, dtype=np.int32)}

    data["total"] = {"local_reqs_ratio": np.empty(episodes),
                     "forward_reqs_ratio": np.empty(episodes),
                     "processed_reqs_ratio": np.empty(episodes),
                     "local_reqs": np.empty(episodes, dtype=np.int32),
                     "forward_reqs": np.empty(episodes, dtype=np.int32),
                     "processable_reqs": np.empty(episodes, dtype=np.int32)}

    data["steps"] = iter["episode_len_mean"]
    data["episodes"] = episodes

    for epi_idx in range(episodes):
        total_processable_reqs = total_local_reqs = total_forward_reqs = 0

        processable_reqs = _processable_requests_episode(iter, epi_idx)
        data["total"]["processable_reqs"][epi_idx] = processable_reqs

        for agent in env.agent_ids:
            action_local = np.sum(iter["hist_stats"]["action_local"][epi_idx][agent], dtype=np.int32)
            action_forward = np.sum(iter["hist_stats"]["action_forward"][epi_idx][agent], dtype=np.int32)

            excess_local = np.sum(iter["hist_stats"]["excess_local"][epi_idx][agent], dtype=np.int32)
            forward_reject = np.sum(iter["hist_stats"]["excess_forward_reject"][epi_idx][agent], dtype=np.int32)

            local_reqs = action_local - excess_local
            data[agent]["local_reqs"][epi_idx] = local_reqs
            total_local_reqs += local_reqs

            forward_reqs = action_forward - forward_reject
            data[agent]["forward_reqs"][epi_idx] = forward_reqs
            total_forward_reqs += forward_reqs

        data["total"]["local_reqs_ratio"][epi_idx] = total_local_reqs / processable_reqs
        data["total"]["forward_reqs_ratio"][epi_idx] = total_forward_reqs / processable_reqs
        data["total"]["processed_reqs_ratio"][epi_idx] = (total_local_reqs + total_forward_reqs) / processable_reqs

        data["total"]["local_reqs"][epi_idx] = total_local_reqs
        data["total"]["forward_reqs"][epi_idx] = total_forward_reqs

    data["total"]["local_reqs_ratio_avg"] = np.average(data["total"]["local_reqs_ratio"])
    data["total"]["forward_reqs_ratio_avg"] = np.average(data["total"]["forward_reqs_ratio"])
    data["total"]["processed_reqs_ratio_avg"] = np.average(data["total"]["processed_reqs_ratio"])

    return data


def make_percentual(data, plots_dir, eval_env, train_env):
    fig = plt.figure(figsize=(7, 5), dpi=600, layout="constrained")
    ax = fig.subplots()

    # Coordinates of the x-bar for the bar plots. Must be explicitly calculated
    # from the number of episodes.
    steps_x = np.arange(stop=data["episodes"])

    # Total (sum of the two nodes) plot.
    ax.bar(x=steps_x,
           height=data["total"]["local_reqs_ratio"] * 100,
           color="g",
           label="Locali")
    ax.bar(x=steps_x,
           height=data["total"]["forward_reqs_ratio"] * 100,
           bottom=data["total"]["local_reqs_ratio"] * 100,
           color="b",
           label="Inoltrate")

    avg_line = np.full(data["episodes"], data["total"]["processed_reqs_ratio_avg"] * 100)
    ax.plot(avg_line, linewidth=2, color="r", label="Media")

    ax.annotate(f'{data["total"]["processed_reqs_ratio_avg"]:.2%}',
                (0, data["total"]["processed_reqs_ratio_avg"] * 100 + 1),
                bbox={"boxstyle": "square", "alpha": .7, "facecolor": "white"})

    ax.set_xlabel("Episodio", fontsize="large")

    ax.set_ylabel("Richieste processate", fontsize="large")
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
    path = plots_dir / "eval_summary_processed_requests_final.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


def make_absolute(data, plots_dir, eval_env, train_env):
    fig = plt.figure(figsize=(7, 5), dpi=600, layout="constrained")
    ax = fig.subplots()

    # Coordinates of the x-bar for the bar plots. Must be explicitly calculated
    # from the number of episodes.
    steps_x = np.arange(stop=data["episodes"])

    # Total (sum of the two nodes) plot.
    base = np.zeros(data["episodes"], dtype=np.int32)
    ax.bar(x=steps_x,
           height=data["node_0"]["local_reqs"],
           bottom=base,
           color="g",
           label=f"Locali (node_0)")
    base = data["node_0"]["local_reqs"]

    ax.bar(x=steps_x,
           height=data["node_1"]["local_reqs"],
           bottom=base,
           color="g",
           edgecolor='black', linewidth=0, hatch="...",
           label=f"Locali (node_1)")
    base += data["node_1"]["local_reqs"]

    ax.bar(x=steps_x,
           height=data["node_0"]["forward_reqs"],
           color="b",
           bottom=base,
           label=f"Inoltrate (node_0)")
    base += data["node_0"]["forward_reqs"]

    ax.bar(x=steps_x,
           height=data["node_1"]["forward_reqs"],
           color="b",
           edgecolor='black', linewidth=0, hatch="...",
           bottom=base,
           label=f"Inoltrate (node_1)")
    base += data["node_1"]["forward_reqs"]

    ax.plot(data["total"]["processable_reqs"], linewidth=2, color="r", label="Richieste processabili totali")

    ax.set_xlabel("Episodio", fontsize="large")

    ax.set_ylabel("Richieste processate", fontsize="large")

    # Show x-axis ticks every X steps.
    if data["episodes"] <= 10:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.

    ax.legend()

    # Save the plot.
    path = plots_dir / "eval_summary_processed_requests_final_abs.pdf"
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
