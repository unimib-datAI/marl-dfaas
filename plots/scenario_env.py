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

from traffic_env import TrafficManagementEnv

from RL4CC.utilities.logger import Logger

logger = Logger(name="DFAAS-PLOTS", verbose=2)


def make(result_dir, scenario):
    logger.log(f"Making plots for scenario {scenario!r}")

    # Create the environment with the given scenario.
    env = TrafficManagementEnv({"scenario": scenario})
    max_steps = env.max_steps

    # Create the arrays to store the partial observations.
    input_requests = np.empty(shape=env.max_steps, dtype=np.int64)
    forward_capacity = np.empty(shape=env.max_steps, dtype=np.int64)

    # Store the first partial observation.
    obs, info = env.reset()
    seed = info["seed"]
    input_requests[0] = obs[0]
    forward_capacity[0] = obs[1]

    # Store the other partial observations.
    for step in np.arange(1, max_steps):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

        input_requests[step] = obs[0]
        forward_capacity[step] = obs[1]

    assert step == max_steps - 1

    # Make the plot.
    fig = plt.figure(figsize=(19.2, 14.3), dpi=300, layout="constrained")
    fig.suptitle(f"Scenario {scenario!r} width seed {seed}")
    axs = fig.subplots(ncols=1, nrows=2)

    # For the ylim, see the environment and observation contraints. The limits
    # ensure a bit of space for both bottom and top.

    # Input requests plot.
    axs[0].plot(input_requests)
    axs[0].set_ylim(bottom=40, top=160)
    axs[0].set_title("Input requests")

    # Forward capacity plot.
    axs[1].plot(forward_capacity)
    axs[1].set_ylim(bottom=-10, top=110)
    axs[1].set_title("Forwarding capacity")

    # Common settings for plots.
    for ax in axs.flat:
        ax.set_xlabel("Step")
        ax.set_ylabel("Requests")

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = Path(result_dir, f"{scenario}.pdf")
    fig.savefig(path)
    plt.close(fig)
    logger.log(f"{scenario}: {path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser(prog="scenario_env")

    parser.add_argument(dest="experiment_directory",
                        help="DFaaS experiment directory")

    args = parser.parse_args()

    # Create the correct folders.
    plots_dir = Path(args.experiment_directory, "plots", "environment")
    plots_dir.mkdir(parents=True, exist_ok=True)

    for scenario in TrafficManagementEnv.get_scenarios():
        make(plots_dir, scenario)
