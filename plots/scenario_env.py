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


def make(plots_dir, scenario):
    logger.log(f"Making plots for scenario {scenario!r}")

    env = TrafficManagementEnv({"scenario": scenario})
    env.reset()

    step = 0
    max_steps = env.max_steps

    input_requests = np.empty(shape=env.max_steps, dtype=np.int64)
    forward_capacity = np.empty(shape=env.max_steps, dtype=np.int64)

    while step < max_steps:
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

        input_requests[step] = obs[0]
        forward_capacity[step] = obs[1]

        step += 1

    # Make the plot.
    fig = plt.figure(figsize=(19.2, 14.3), dpi=300, layout="constrained")
    fig.suptitle(f"Scenario {scenario!r}")
    axs = fig.subplots(ncols=1, nrows=2)

    axs[0].plot(input_requests)
    axs[0].set_title("Input requests")

    axs[1].plot(forward_capacity)
    axs[1].set_title("Forwarding capacity")

    for ax in axs.flat:
        ax.set_xlabel("Step")

    # Save the plot.
    path = Path(plots_dir, f"{scenario}.pdf")
    fig.savefig(path)
    plt.close(fig)
    logger.log(f"{scenario}: {path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser(prog="scenario_env")

    parser.add_argument(dest="experiment_directory",
                        help="DFaaS experiment directory")

    args = parser.parse_args()

    for scenario in TrafficManagementEnv.get_scenarios():
        make(args.experiment_directory, scenario)
