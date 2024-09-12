# This Python script generates a plot showing the metrics related to the excess
# of requests (local, forward and reject processing) of the training phase.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

if __name__ == "__main__":
    import sys
    import os
    import argparse
    import matplotlib

    # Add the current directory (where Python is called) to sys.path. This
    # assumes this script is called in the project root directory, not inside
    # the directory where the script is.
    #
    # Required when calling this module directly as main.
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

    percent_reject_reqs_excess = {}
    percent_local_reqs_excess = {}
    percent_forward_reqs_excess = {}
    percent_forward_reqs_reject = {}

    # The data is grouped in agents.
    for agent in agents:
        percent_reject_reqs_excess[agent] = np.empty(len(iters), dtype=np.float32)
        percent_local_reqs_excess[agent] = np.empty(len(iters), dtype=np.float32)
        percent_forward_reqs_excess[agent] = np.empty(len(iters), dtype=np.float32)
        percent_forward_reqs_reject[agent] = np.empty(len(iters), dtype=np.float32)

    for iter in range(len(iters)):
        for agent in agents:
            percent_reject_reqs_excess[agent][iter] = metrics["reject_reqs_percent_excess_per_iteration"][iter][agent]
            percent_local_reqs_excess[agent][iter] = metrics["local_reqs_percent_excess_per_iteration"][iter][agent]
            percent_forward_reqs_excess[agent][iter] = metrics["forward_reqs_percent_excess_per_iteration"][iter][agent]
            percent_forward_reqs_reject[agent][iter] = metrics["forward_reqs_percent_reject_per_iteration"][iter][agent]

    data["agents"] = agents
    data["iterations"] = len(iters)
    data["percent_reject_reqs_excess"] = percent_reject_reqs_excess
    data["percent_local_reqs_excess"] = percent_local_reqs_excess
    data["percent_forward_reqs_excess"] = percent_forward_reqs_excess
    data["percent_forward_reqs_reject"] = percent_forward_reqs_reject

    return data


def make(exp_dir):
    exp_dir = dfaas_utils.to_pathlib(exp_dir)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _get_data(exp_dir)

    fig = plt.figure(figsize=(17, 8), dpi=600, layout="constrained")
    axs = fig.subplots(nrows=2, ncols=2)

    # For the ylim, the total reward for one episode cannot exceed the possible
    # max and min of one episode. The limits ensure a bit of space for both
    # bottom and top.
    #
    # TODO: extract dynamically the values from the env.
    bottom = 0.0 - 1
    top = 100.0 + 1

    for agent in data["agents"]:
        axs[0, 0].plot(data["percent_reject_reqs_excess"][agent], label=agent)
    axs[0, 0].set_ylim(bottom=bottom, top=top)
    axs[0, 0].set_title("Excessive rejected requests per step (average percent over rejected requests)")
    axs[0, 0].set_ylabel("Percentage")
    axs[0, 0].set_ylim(bottom=0, top=100)  # Set Y axis range from 0 to 100 (percent).
    axs[0, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    axs[0, 0].set_yticks(np.arange(0, 100+1, 10))

    for agent in data["agents"]:
        axs[0, 1].plot(data["percent_local_reqs_excess"][agent], label=agent)
    axs[0, 1].set_title("Excessive local requests per step (average percent over queue size)")
    axs[0, 1].set_ylabel("Percentage")
    axs[0, 1].set_ylim(bottom=0, top=100)
    axs[0, 1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    axs[0, 1].set_yticks(np.arange(0, 100+1, 10))

    for agent in data["agents"]:
        axs[1, 0].plot(data["percent_forward_reqs_excess"][agent], label=agent)
    axs[1, 0].set_title("Excessive forwarded requests per step (average percent over forwarded requests)")
    axs[1, 0].set_ylabel("Percentage")
    axs[1, 0].set_ylim(bottom=0, top=100)
    axs[1, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    axs[1, 0].set_yticks(np.arange(0, 100+1, 10))

    for agent in data["agents"]:
        axs[1, 1].plot(data["percent_forward_reqs_reject"][agent], label=agent)
    axs[1, 1].set_title("Forwarded requests rejected per step (average percent over forwarded requests)")
    axs[1, 1].set_ylabel("Percentage")
    axs[1, 1].set_ylim(bottom=0, top=100)
    axs[1, 1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    axs[1, 1].set_yticks(np.arange(0, 100+1, 10))

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

        ax.legend()

    # Save the plot.
    path = plots_dir / "train_summary_excess.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="dfaas_env")

    parser.add_argument(dest="exp_dir",
                        help="DFaaS experiment directory")

    args = parser.parse_args()

    make(args.exp_dir)
