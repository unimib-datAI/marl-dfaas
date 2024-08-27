# This Python script generates a plot of each agent's input request curve in a
# random DFaaS environment.
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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

from dfaas_env import DFaaS
import dfaas_utils


def make(plots_dir):
    plots_dir = dfaas_utils.to_pathlib(plots_dir)

    env = DFaaS()

    env.reset()

    # Make the plot.
    fig = plt.figure(figsize=(15, 10), dpi=300, layout="constrained")
    ax = fig.subplots(ncols=1, nrows=1)

    # For the ylim, see the environment and observation contraints. The limits
    # ensure a bit of space for both bottom and top.
    #
    # Is assumed that each agent has the same observation space.
    bottom = env.observation_space["node_0"]["input_requests"].low[0] - 10
    top = env.observation_space["node_0"]["input_requests"].high[0] + 10

    # Input requests plot.
    for agent in env._agent_ids:
        ax.plot(env.input_requests[agent], label=agent)
    ax.set_ylim(bottom=bottom, top=top)
    ax.set_yticks(np.arange(start=bottom, stop=top+1, step=10))
    ax.set_xticks(np.arange(stop=env.node_max_steps+1, step=10))
    ax.set_title("Input requests")

    ax.set_xlabel("Step")
    ax.set_ylabel("Requests")

    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.
    ax.legend()

    # Save the plot.
    path = Path(plots_dir, "dfaas_env_requests.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="dfaas_env")

    parser.add_argument(dest="experiment_directory",
                        help="DFaaS experiment directory")

    args = parser.parse_args()

    # Create the folders.
    plots_dir = Path(args.experiment_directory, "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    make(plots_dir)
