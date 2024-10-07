# This Python script generates a plot of the DFaaS environment showing the
# curves of input requests, forward capacity, and queue capacity.
#
# The environment config is retrieved from the specified experiment directory.
from pathlib import Path
import sys
import os
import argparse

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add the current directory (where Python is called) to sys.path. This is
# required to load modules in the project root directory (like dfaas_utils.py).
sys.path.append(os.getcwd())

import plot_utils


def _get_data(exp_dir, seed):
    env = plot_utils.get_env(exp_dir)

    # Note that the seed argument is ignored because of "override_seed".
    env.reset(seed=seed, options={"override_seed": seed})

    data = {}
    data["env"] = env
    data["input_reqs"] = env.input_requests

    low, high = [], []
    for agent in env.agent_ids:
        low.append(env.observation_space[agent]["input_requests"].low[0])
        high.append(env.observation_space[agent]["input_requests"].high[0])

    data["input_reqs_min"] = min(low)
    data["input_reqs_max"] = max(high)

    if (hashes := getattr(env, "input_requests_hashes", None)) is not None:
        # Real traces.
        data["hashes"] = hashes
    else:
        # Syntethic traces.
        data["seed"] = env.seed

    return data


def make(exp_dir, seed):
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _get_data(exp_dir, seed)
    env = data["env"]

    if "hashes" in data:
        hashes = [data["hashes"][agent][-10:] for agent in env.agent_ids]
        hash_str = ", ".join(hashes)
        title = f"Input requests (function hashes {hash_str})"
    else:
        title = f"Input requests (reqs type {env.input_requests_type!r}, seed {data['seed']})"

    # Make the plot.
    fig = plt.figure(figsize=(25, 10), dpi=600, layout="constrained")
    fig.suptitle(title)
    axs = fig.subplots(nrows=2)

    idx = 0
    for agent in env.agent_ids:
        axs[idx].plot(data["input_reqs"][agent], label=agent)
        idx += 1

    # Common settings for the plots.
    for ax in axs.flat:
        ax.set_title("Input requests")

        ax.set_xlabel("Step")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

        ax.set_ylabel("Requests")
        ax.set_ylim(bottom=data["input_reqs_min"]-1, top=data["input_reqs_max"]+1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = plots_dir / f"dfaas_env_requests_{seed}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="dfaas_env_requests")

    parser.add_argument(dest="experiment_directory",
                        help="DFaaS experiment directory",
                        type=Path)
    parser.add_argument(dest="seed",
                        help="Seed of the environment",
                        type=int)

    args = parser.parse_args()

    make(args.experiment_directory.resolve(), args.seed)
