# This Python script generates a plot of the DFaaS environment showing the
# curves of input requests, forward capacity, and queue capacity.
#
# The environment config is retrieved from the specified experiment directory.
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


def _get_data(env_config):
    env = DFaaS(config=env_config)

    assert env.agents == 2, "The scripts supports only DFaaS env with two agents"

    input_requests, queue_capacity, forward_capacity = {}, {}, {}
    for agent in env.agent_ids:
        input_requests[agent] = np.zeros(env.max_steps, dtype=np.int32)
        queue_capacity[agent] = np.zeros(env.max_steps, dtype=np.int32)
        forward_capacity[agent] = np.zeros(env.max_steps, dtype=np.int32)

    def push_data(obs, step):
        for agent in env.agent_ids:
            input_requests[agent][step] = obs[agent]["input_requests"].item()
            queue_capacity[agent][step] = obs[agent]["queue_capacity"].item()
            if agent == "node_0":
                forward_capacity[agent][step] = obs[agent]["forward_capacity"].item()

    # Run through all the steps in the environment and save the observations.
    obs, _ = env.reset()
    terminated = {"__all__": False}
    while not terminated["__all__"]:
        push_data(obs, env.current_step)

        # We can track the observation regardless of the chosen action because
        # input requests, forward capacity, and queue capacity do not depend on
        # the action.
        obs, _, terminated, _, _ = env.step(env.action_space_sample())

    data = {}
    data["env"] = env
    data["input_requests"] = input_requests
    data["queue_capacity"] = queue_capacity
    data["forward_capacity"] = forward_capacity

    return data


def make(plots_dir, env_config):
    data = _get_data(env_config)

    # Make the plot.
    fig = plt.figure(figsize=(13, 13), dpi=300, layout="constrained")
    axs = fig.subplots(nrows=3)

    env = data["env"]

    # 1. Input requests.

    # Get the upper and lower limits for the y-axis.
    bottom, top = +np.inf, -np.inf
    for agent in env.agent_ids:
        tmp_bottom = env.observation_space[agent]["input_requests"].low[0]
        if tmp_bottom < bottom:
            bottom = tmp_bottom
        tmp_top = env.observation_space[agent]["input_requests"].high[0]
        if tmp_top > top:
            top = tmp_top
    bottom -= 10  # Add a bit of space.
    top += 10

    for agent in env._agent_ids:
        axs[0].plot(env.input_requests[agent], label=agent)
    axs[0].set_ylim(bottom=bottom, top=top)
    axs[0].set_yticks(np.arange(start=bottom, stop=top+1, step=10))
    axs[0].set_title("Input requests")

    # 2. Queue capacity.

    # Get the upper and lower limits for the y-axis.
    bottom, top = +np.inf, -np.inf
    for agent in env.agent_ids:
        tmp_bottom = env.observation_space[agent]["queue_capacity"].low[0]
        if tmp_bottom < bottom:
            bottom = tmp_bottom
        tmp_top = env.observation_space[agent]["queue_capacity"].high[0]
        if tmp_top > top:
            top = tmp_top
    top += 10  # Add a bit of space.

    for agent in env._agent_ids:
        axs[1].plot(data["queue_capacity"][agent], label=agent)
    axs[1].set_ylim(bottom=bottom, top=top)
    axs[1].set_yticks(np.arange(start=bottom, stop=top+1, step=10))
    axs[1].set_title("Queue capacity")

    # 2. Forward capacity.

    # Get the upper and lower limits for the y-axis.
    bottom = env.observation_space["node_0"]["forward_capacity"].low[0] + 10
    top = env.observation_space["node_0"]["forward_capacity"].high[0] - 10

    for agent in env._agent_ids:
        axs[2].plot(data["forward_capacity"][agent], label=agent)
    axs[2].set_ylim(bottom=bottom, top=top)
    axs[2].set_yticks(np.arange(start=bottom, stop=top+1, step=10))
    axs[2].set_title("Forward capacity")

    # Common settings for the plots.
    for ax in axs.flat:
        ax.set_xlabel("Step")
        ax.set_xticks(np.arange(stop=env.max_steps+1, step=10))

        ax.set_ylabel("Requests")

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

        ax.legend()

    # Save the plot.
    path = Path(plots_dir, "dfaas_env_requests.pdf").absolute()
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

    # Get the environment config.
    config_path = Path(args.experiment_directory, "env_config.json")
    config = dfaas_utils.json_to_dict(config_path)

    make(plots_dir, config)
