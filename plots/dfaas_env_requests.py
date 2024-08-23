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

    # Create the arrays to store the input requests for each agent.
    input_requests = {"node_0": np.empty(shape=env.node_max_steps, dtype=np.int32),
                      "node_1": np.empty(shape=env.node_max_steps, dtype=np.int32)
                      }

    # The first observation is always the input requests for the node_0 agent.
    obs, _ = env.reset()
    input_requests["node_0"][0] = obs["node_0"][0]  # A single element.

    # The other observation cycles between the two agents.
    while True:
        action = env.action_space_sample()
        obs, _, terminated, _, _ = env.step(action)

        if terminated["__all__"]:
            break

        # Get the current agent for the action. Is expected that there is only
        # one agent for each step in the observation.
        assert len(list(obs)) == 1, "Only one agent should be in obs for each step!"
        agent = list(obs)[0]
        step = env.current_step[agent]
        input_requests[agent][step] = obs[agent][0]

    assert env.current_step == {"node_0": env.node_max_steps, "node_1": env.node_max_steps}

    # Make the plot.
    fig = plt.figure(figsize=(15, 10), dpi=300, layout="constrained")
    ax = fig.subplots(ncols=1, nrows=1)

    # For the ylim, see the environment and observation contraints. The limits
    # ensure a bit of space for both bottom and top.
    #
    # Is assumed that each agent has the same observation space.
    bottom = env.observation_space["node_0"].low[0] - 10
    top = env.observation_space["node_0"].high[0] + 10

    # Input requests plot.
    ax.plot(input_requests["node_0"], label="node_0")
    ax.plot(input_requests["node_1"], label="node_1")
    ax.set_ylim(bottom=bottom, top=top)
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
