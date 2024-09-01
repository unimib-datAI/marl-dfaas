# This script generates a plot showing the details of one optimal episode
# generated by "run-optimal.py".
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


def _get_agents():
    """Returns the agent IDs from the given list of iterations. Assumes at least
    one iteration and one episode for iteration."""
    return ["node_0", "node_1"]


def _get_data(dir):
    agents = _get_agents()

    data = {}

    # Read data from the given directory.
    result = dfaas_utils.parse_result_file(dir / "result.json")

    # There is just one iteration (of one episode)!
    result = result[0]

    episode_idx = 0  # Only one episode.

    # Get the data from the specified episode index.
    seed = result["hist_stats"]["seed"][episode_idx]
    input_requests = result["hist_stats"]["observation_input_requests"][episode_idx]
    queue_capacity = result["hist_stats"]["observation_queue_capacity"][episode_idx]
    forward_capacity = result["hist_stats"]["observation_forward_capacity"][episode_idx]
    action_local = result["hist_stats"]["action_local"][episode_idx]
    action_forward = result["hist_stats"]["action_forward"][episode_idx]
    action_reject = result["hist_stats"]["action_reject"][episode_idx]
    reward = result["hist_stats"]["reward"][episode_idx]
    steps = result["episode_len"]

    # Check if all rewards are maximised.
    for agent in agents:
        reward_agent = np.array(reward[agent])
        assert np.all(np.equal(reward_agent, np.ones(steps))), "There is an error in run-optimal.py"

    input_reqs_max = DFaaS().observation_space["node_0"]["input_requests"].high[0]

    data["agents"] = agents
    data["steps"] = steps
    data["seed"] = seed
    data["input_requests"] = input_requests
    data["input_requests_max"] = input_reqs_max
    data["queue_capacity"] = queue_capacity
    data["forward_capacity"] = forward_capacity

    # Convert to np.ndarray.
    data["action_local"], data["action_reject"], data["action_forward"] = {}, {}, {}
    for agent in agents:
        data["action_local"][agent] = np.array(action_local[agent], dtype=np.int32)
        data["action_reject"][agent] = np.array(action_reject[agent], dtype=np.int32)

        if agent == "node_0":
            data["action_forward"][agent] = np.array(action_forward[agent], dtype=np.int32)
        else:
            data["action_forward"][agent] = np.zeros(steps, dtype=np.int32)

    return data


def make(dir):
    dir = dfaas_utils.to_pathlib(dir)
    plots_dir = dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _get_data(dir)

    # Coordinates of the x-bar for the bar plots. Must be explicitly calculated
    # from the number of steps for each agent.
    steps_x = np.arange(stop=data["steps"])

    fig = plt.figure(figsize=(20, 22), dpi=600, layout="constrained")
    fig.suptitle(f"Perfect episode (env seed {data['seed']})")
    axs = fig.subplots(nrows=4, ncols=2)

    # Make the same three plots for each agents.
    idx = 0
    for agent in data["agents"]:
        local = data["action_local"][agent]
        reject = data["action_reject"][agent]
        forward = data["action_forward"][agent]

        axs[0, idx].bar(x=steps_x, height=local, label="Local")
        axs[0, idx].bar(x=steps_x, height=reject, label="Reject", bottom=local)
        if agent == "node_0":
            axs[0, idx].bar(x=steps_x, height=forward, label="Forward", bottom=local+reject)
        axs[0, idx].plot(data["input_requests"][agent], linewidth=3, color="r", label="Input requests")
        axs[0, idx].set_title(f"Action ({agent})")
        axs[0, idx].set_ylabel("Requests")

        axs[1, idx].bar(x=steps_x, height=local, color="g", label="Local")
        axs[1, idx].plot(data["queue_capacity"][agent], linewidth=3, color="b", label="Queue capacity")
        axs[1, idx].set_title(f"Local action ({agent})")
        axs[1, idx].set_ylabel("Requests")
        axs[1, idx].set_ylim(top=data["input_requests_max"])

        axs[2, idx].bar(x=steps_x, height=forward, color="g", label="Forward")
        axs[2, idx].plot(data["forward_capacity"][agent], linewidth=3, color="b", label="Forward capacity")
        axs[2, idx].set_title(f"Forward action ({agent})")
        axs[2, idx].set_ylabel("Requests")
        axs[2, idx].set_ylim(top=data["input_requests_max"])

        axs[3, idx].bar(x=steps_x, height=reject, color="g", label="Reject")
        axs[3, idx].set_title(f"Reject action ({agent})")
        axs[3, idx].set_ylabel("Requests")
        axs[3, idx].set_ylim(top=data["input_requests_max"])

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
    path = plots_dir / "episode.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(dest="dir",
                        help="Directory containing the 'result.json' file.")

    args = parser.parse_args()

    make(args.dir)
