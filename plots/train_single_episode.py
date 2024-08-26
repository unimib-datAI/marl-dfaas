# This script generates a plot showing the details of one episode of one
# iteration from the training process.
#
# The user can specify which episode and iteration should be plotted.
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
from dfaas_env import DFaaS


def _get_agents(iters):
    """Returns the agent IDs from the given list of iterations. Assumes at least
    one iteration and one episode for iteration."""
    return list(iters[0]["hist_stats"]["input_requests"][0].keys())


def _get_data(exp_dir, iteration_idx, episode_idx):
    data = {}

    # Read data from experiment directory.
    iters = dfaas_utils.parse_result_file(exp_dir / "result.json")
    metrics = dfaas_utils.json_to_dict(exp_dir / "metrics.json")
    agents = _get_agents(iters)

    # Get only a specific iteration.
    iter = iters[iteration_idx]

    # Get the data from the specified episode index.
    seed = iter["hist_stats"]["seed"][episode_idx]
    input_requests = iter["hist_stats"]["input_requests"][episode_idx]
    action = iter["hist_stats"]["action"][episode_idx]
    reward = iter["hist_stats"]["reward"][episode_idx]
    steps = len(input_requests[agents[0]])

    exceed = {"local": metrics["iterations"][iteration_idx]["local_reqs_exceed_per_step"][episode_idx],
              "reject": metrics["iterations"][iteration_idx]["reject_reqs_exceed_per_step"][episode_idx]}

    # Get the reward range from the environment.
    reward_range = DFaaS().reward_range

    data["agents"] = agents
    data["steps"] = steps
    data["seed"] = seed
    data["input_requests"] = input_requests
    data["action"] = action
    data["reward"] = reward
    data["reward_range"] = reward_range
    data["exceed"] = exceed

    return data


def make(exp_dir, iteration_idx, episode_idx):
    exp_dir = dfaas_utils.to_pathlib(exp_dir)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _get_data(exp_dir, iteration_idx, episode_idx)

    # Coordinates of the x-bar for the bar plots. Must be explicitly calculated
    # from the number of steps for each agent.
    steps_x = np.arange(stop=data["steps"])

    fig = plt.figure(figsize=(20, 15), dpi=600, layout="constrained")
    fig.suptitle(f"Episode {episode_idx} of iteration {iteration_idx} (env seed {data['seed']})")
    axs = fig.subplots(nrows=3, ncols=2)

    assert len(data["agents"]) == 2, "This plot supports only two-agent environment"

    # Calculate the lower and upper limits for the exceed requests for both the
    # local exceed and the reject exceed. These limits will later be used for
    # the third subplot for each agent for the y-axis.
    exceed_bottom, exceed_top = np.nan, np.nan
    for agent in data["agents"]:
        for action in data["exceed"]:
            agent_bottom = np.min(data["exceed"][action][agent])
            agent_top = np.max(data["exceed"][action][agent])

            exceed_bottom = np.fmin(agent_bottom, exceed_bottom)
            exceed_top = np.fmax(agent_top, exceed_top)

    # Make the same three plots for each agents.
    idx = 0
    for agent in data["agents"]:
        local = data["action"]["local"][agent]
        reject = data["action"]["reject"][agent]
        input_requests = data["input_requests"][agent]

        axs[0, idx].bar(x=steps_x, height=local, label="Local")
        axs[0, idx].bar(x=steps_x, height=reject, label="Reject", bottom=local)
        axs[0, idx].plot(input_requests, linewidth=3, color="r", label="Input requests")
        axs[0, idx].set_title(f"Action ({agent})")
        axs[0, idx].set_ylabel("Requests")

        axs[1, idx].plot(data["reward"][agent], label="Reward")
        axs[1, idx].set_title(f"Reward ({agent})")
        axs[1, idx].set_ylabel("Reward")

        # Set the y-ticks for the y-axis. Enforce a little space around the
        # borders.
        bottom, top = data["reward_range"]
        axs[1, idx].set_ylim(bottom=bottom-0.1, top=top+.1)
        axs[1, idx].set_yticks(np.arange(start=bottom, stop=top+.1, step=.1))

        axs[2, idx].bar(x=steps_x, height=data["exceed"]["local"][agent], label="Local exceed")
        axs[2, idx].bar(x=steps_x, height=data["exceed"]["reject"][agent], label="Reject exceed")
        axs[2, idx].set_title(f"Exceeded requests ({agent})")
        axs[2, idx].set_ylabel("Requests")
        axs[2, idx].set_ylim(bottom=exceed_bottom, top=exceed_top)

        idx += 1

    # Common settings for the plots.
    for ax in axs.flat:
        ax.set_xlabel("Step")

        # Show x-axis ticks every 10 steps.
        ax.set_xticks(np.arange(0, data["steps"]+1, 10))

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

        ax.legend()

    # Save the plot.
    path = plots_dir / f"train_iteration_{iteration_idx}_episode_{episode_idx}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="dfaas_env")

    parser.add_argument(dest="exp_dir",
                        help="DFaaS experiment directory")
    parser.add_argument(dest="episode",
                        help="Plot the given episode (the index).",
                        type=int)
    parser.add_argument(dest="iteration",
                        help="Select the episode from the given iteration (the index).",
                        type=int)

    args = parser.parse_args()

    make(args.exp_dir, args.iteration, args.episode)
