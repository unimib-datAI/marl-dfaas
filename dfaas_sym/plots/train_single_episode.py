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


def _get_agents():
    """Returns the agent IDs from the given list of iterations. Assumes at least
    one iteration and one episode for iteration."""
    return ["node_0", "node_1"]


def _get_data(exp_dir, iteration_idx, episode_idx):
    data = {}

    # Read data from experiment directory.
    iters = dfaas_utils.parse_result_file(exp_dir / "result.json")
    metrics = dfaas_utils.json_to_dict(exp_dir / "metrics-result.json")
    agents = _get_agents()

    # Get only a specific iteration.
    iter = iters[iteration_idx]

    # Get the data from the specified episode index.
    seed = iter["hist_stats"]["seed"][episode_idx]
    input_requests = iter["hist_stats"]["observation_input_requests"][episode_idx]
    queue_capacity = iter["hist_stats"]["observation_queue_capacity"][episode_idx]
    forward_capacity = iter["hist_stats"]["observation_forward_capacity"][episode_idx]
    action_local = iter["hist_stats"]["action_local"][episode_idx]
    action_forward = iter["hist_stats"]["action_forward"][episode_idx]
    action_reject = iter["hist_stats"]["action_reject"][episode_idx]
    excess_local = iter["hist_stats"]["excess_local"][episode_idx]
    excess_forward = iter["hist_stats"]["excess_forward"][episode_idx]
    excess_reject = metrics["reject_excess_per_step"][iteration_idx][episode_idx]
    reward = iter["hist_stats"]["reward"][episode_idx]
    steps = iter["hist_stats"]["episode_lengths"][0]

    # Get the reward range from the environment.
    reward_range = DFaaS().reward_range

    input_reqs_max = DFaaS().observation_space["node_0"]["input_requests"].high[0]

    data["agents"] = agents
    data["steps"] = steps
    data["seed"] = seed
    data["input_requests"] = input_requests
    data["input_requests_max"] = input_reqs_max
    data["queue_capacity"] = queue_capacity
    data["forward_capacity"] = forward_capacity

    data["action_local"], data["action_reject"], data["action_forward"] = {}, {}, {}
    data["excess_local"], data["excess_forward"], data["excess_reject"] = {}, {}, {}
    for agent in agents:
        data["action_local"][agent] = np.array(action_local[agent], dtype=np.int32)
        data["action_reject"][agent] = np.array(action_reject[agent], dtype=np.int32)

        data["excess_local"][agent] = np.array(excess_local[agent], dtype=np.int32)
        data["excess_reject"][agent] = np.array(excess_reject[agent], dtype=np.int32)

        if agent == "node_0":
            data["action_forward"][agent] = np.array(action_forward[agent], dtype=np.int32)
            data["excess_forward"][agent] = np.array(excess_forward[agent], dtype=np.int32)
        else:
            data["action_forward"][agent] = np.zeros(steps, dtype=np.int32)
            data["excess_forward"][agent] = np.zeros(steps, dtype=np.int32)

    data["reward"] = reward
    data["reward_range"] = reward_range

    return data


def make(exp_dir, iteration_idx, episode_idx):
    exp_dir = dfaas_utils.to_pathlib(exp_dir)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _get_data(exp_dir, iteration_idx, episode_idx)

    # Coordinates of the x-bar for the bar plots. Must be explicitly calculated
    # from the number of steps for each agent.
    steps_x = np.arange(stop=data["steps"])

    fig = plt.figure(figsize=(20, 27), dpi=600, layout="constrained")
    fig.suptitle(f"Episode {episode_idx} of iteration {iteration_idx} (env seed {data['seed']})")
    axs = fig.subplots(nrows=5, ncols=2)

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

        axs[1, idx].plot(data["reward"][agent], label="Reward")
        axs[1, idx].set_title(f"Reward ({agent})")
        axs[1, idx].set_ylabel("Reward")
        # Set the y-ticks for the y-axis. Enforce a little space around the
        # borders.
        bottom, top = data["reward_range"]
        axs[1, idx].set_ylim(bottom=bottom-0.1, top=top+.1)
        axs[1, idx].set_yticks(np.arange(start=bottom, stop=top+.1, step=.1))

        excess_local = data["excess_local"][agent]
        real_local = local - excess_local
        axs[2, idx].bar(x=steps_x, height=real_local, color="g", label="Local")
        axs[2, idx].bar(x=steps_x, height=excess_local, color="r", label="Local excess", bottom=real_local)
        axs[2, idx].plot(data["queue_capacity"][agent], linewidth=3, color="b", label="Queue capacity")
        axs[2, idx].set_title(f"Local action ({agent})")
        axs[2, idx].set_ylabel("Requests")
        axs[2, idx].set_ylim(top=data["input_requests_max"])

        excess_forward = data["excess_forward"][agent]
        real_forward = forward - excess_forward
        axs[3, idx].bar(x=steps_x, height=real_forward, color="g", label="Forward")
        axs[3, idx].bar(x=steps_x, height=excess_forward, color="r", label="Forward excess", bottom=real_forward)
        axs[3, idx].plot(data["forward_capacity"][agent], linewidth=3, color="b", label="Forward capacity")
        axs[3, idx].set_title(f"Forward action ({agent})")
        axs[3, idx].set_ylabel("Requests")
        axs[3, idx].set_ylim(top=data["input_requests_max"])

        excess_reject = data["excess_reject"][agent]
        good_reject = reject - excess_reject
        axs[4, idx].bar(x=steps_x, height=good_reject, color="g", label="Reject")
        axs[4, idx].bar(x=steps_x, height=excess_reject, color="r", label="Reject excess", bottom=good_reject)
        axs[4, idx].set_title(f"Reject action ({agent})")
        axs[4, idx].set_ylabel("Requests")
        axs[4, idx].set_ylim(top=data["input_requests_max"])

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
