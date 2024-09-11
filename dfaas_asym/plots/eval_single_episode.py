# This script generates a plot showing the details of one episode of one
# iteration from the evaluation process.
#
# The user can specify which episode to plot.
import sys
import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

# Add the current directory (where Python is called) to sys.path. This is
# required to load modules in the project root directory (like dfaas_utils.py).
sys.path.append(os.getcwd())

import dfaas_utils
from dfaas_asym.env import DFaaS


def _get_agents():
    """Returns the agent IDs from the given list of iterations. Assumes at least
    one iteration and one episode for iteration."""
    return ["node_0", "node_1"]


def _get_forward_capacity(iter, epi_idx):
    """Returns the forward capacity for each step in the given episode id for
    both agents. Data is read from the given iter dictionary.

    TODO: This function should not exist."""
    steps = int(iter["episode_len_mean"])  # TODO: Get exact value.
    forward_capacity = {agent: np.zeros(steps, dtype=np.int32) for agent in _get_agents()}

    input_reqs = {
            agent: np.array(iter["hist_stats"]["observation_input_requests"][epi_idx][agent], dtype=np.int32) for agent in _get_agents()
            }
    queue_capacity = {
            agent: np.array(iter["hist_stats"]["observation_queue_capacity"][epi_idx][agent], dtype=np.int32) for agent in _get_agents()
            }

    # Forward capacity for node_0.
    raw_fw_cap = queue_capacity["node_1"] - input_reqs["node_1"]
    forward_capacity["node_0"] = np.clip(raw_fw_cap, 0, np.inf)

    # Forward capacity for node_1.
    raw_fw_cap = queue_capacity["node_0"] - input_reqs["node_0"]
    forward_capacity["node_1"] = np.clip(raw_fw_cap, 0, np.inf)

    return forward_capacity


def _get_data(exp_dir, episode_idx):
    data = {}

    # Read data from experiment directory.
    iters = dfaas_utils.parse_result_file(exp_dir / "final_evaluation.json")
    metrics = dfaas_utils.json_to_dict(exp_dir / "metrics-final_evaluation.json")
    agents = _get_agents()

    # Get only a specific iteration (the only one).
    iter = iters[0]["evaluation"]

    # Get the data from the specified episode index.
    seed = iter["hist_stats"]["seed"][episode_idx]
    input_requests = iter["hist_stats"]["observation_input_requests"][episode_idx]
    queue_capacity = iter["hist_stats"]["observation_queue_capacity"][episode_idx]
    forward_capacity = _get_forward_capacity(iter, episode_idx)
    action_local = iter["hist_stats"]["action_local"][episode_idx]
    action_forward = iter["hist_stats"]["action_forward"][episode_idx]
    action_reject = iter["hist_stats"]["action_reject"][episode_idx]
    excess_local = iter["hist_stats"]["excess_local"][episode_idx]
    forward_reject = iter["hist_stats"]["excess_forward_reject"][episode_idx]
    excess_reject = metrics["reject_excess_per_step"][episode_idx]
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
    data["excess_local"], data["forward_reject"], data["excess_reject"] = {}, {}, {}
    for agent in agents:
        data["action_local"][agent] = np.array(action_local[agent], dtype=np.int32)
        data["action_reject"][agent] = np.array(action_reject[agent], dtype=np.int32)

        data["excess_local"][agent] = np.array(excess_local[agent], dtype=np.int32)
        data["excess_reject"][agent] = np.array(excess_reject[agent], dtype=np.int32)

        if agent == "node_0":
            data["action_forward"][agent] = np.array(action_forward[agent], dtype=np.int32)
            data["forward_reject"][agent] = np.array(forward_reject[agent], dtype=np.int32)
        else:
            data["action_forward"][agent] = np.zeros(steps, dtype=np.int32)
            data["forward_reject"][agent] = np.zeros(steps, dtype=np.int32)

    data["reward"] = reward
    data["reward_range"] = reward_range

    return data


def make(exp_dir, episode_idx):
    exp_dir = dfaas_utils.to_pathlib(exp_dir)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _get_data(exp_dir, episode_idx)

    # Coordinates of the x-bar for the bar plots. Must be explicitly calculated
    # from the number of steps for each agent.
    steps_x = np.arange(stop=data["steps"])

    fig = plt.figure(figsize=(20, 27), dpi=600, layout="constrained")
    fig.suptitle(f"Episode {episode_idx} of evaluation (env seed {data['seed']})")
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

        forward_reject = data["forward_reject"][agent]
        real_forward = forward - forward_reject
        axs[3, idx].bar(x=steps_x, height=real_forward, color="g", label="Forward")
        axs[3, idx].bar(x=steps_x, height=forward_reject, color="r", label="Forward rejects", bottom=real_forward)
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
    path = plots_dir / f"eval_episode_{episode_idx}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(dest="exp_dir",
                        help="DFaaS experiment directory")
    parser.add_argument(dest="episode",
                        help="Plot the given episode (the index).",
                        type=int)

    args = parser.parse_args()

    make(args.exp_dir, args.episode)
