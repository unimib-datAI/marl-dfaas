# This script generates a plot showing the details of one episode of one
# iteration from the evaluation process.
#
# The user can specify which episode to plot.
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add the current directory (where Python is called) to sys.path. This is
# required to load modules in the project root directory (like dfaas_utils.py).
sys.path.append(os.getcwd())

import dfaas_utils
import plot_utils


def _calc_excess_reject_step(action, excess, queue_capacity):
    """Returns the number of excess rejected requests for a single step for a
    generic agent, using the given action, excess, and queue capacity. This is a
    helper function that is used in calc_excess_reject().

    The arguments are:

        - action: a tuple of three sizes with the requests to locally process,
          forward, and reject,

        - excess: a tuple of length two with the excess number of locally
          processed requests and forwarded rejected requests,

        - queue_capacity: the queue capacity of the agent.
    """
    local, forward, reject = action
    excess_local, forward_reject = excess

    free_slots = queue_capacity - local
    if free_slots < 0:
        # There is a over-local processing.
        assert excess_local == -free_slots, f"Expected equal, found {excess_local} != {-free_slots}"
        excess_reject = 0
    elif free_slots >= reject:
        # All rejected requests could have been processed locally.
        excess_reject = reject
    else:
        # Some rejected requests could have been processed locally.
        excess_reject = reject - (reject - free_slots)
    reject -= excess_reject

    if forward_reject == 0:
        # All forwarded requests were not rejected. This means that the policy
        # may have forwarded more requests instead of rejecting them
        # immediately.
        excess_reject += reject

    return excess_reject


def calc_excess_reject(episode_data, agent, epi_idx):
    """Returns the number of excess rejected requests (requests that could have
    been forwarded or processed locally) for each step.

    The number is calculated for the given agent in each step.
    """
    input_reqs = episode_data["observation_input_requests"][epi_idx][agent]
    queue_capacity = episode_data["observation_queue_capacity"][epi_idx][agent]
    local_reqs = episode_data["action_local"][epi_idx][agent]
    reject_reqs = episode_data["action_reject"][epi_idx][agent]
    excess_local = episode_data["excess_local"][epi_idx][agent]

    if agent == "node_0":
        forward_reqs = episode_data["action_forward"][epi_idx][agent]
        excess_forward_reject = episode_data["excess_forward_reject"][epi_idx][agent]
    else:
        steps = len(input_reqs)
        forward_reqs = np.zeros(steps, dtype=np.int32)
        excess_forward_reject = np.zeros(steps, dtype=np.int32)

    excess_reject = np.zeros(len(input_reqs), dtype=np.int32)
    for step in range(len(input_reqs)):
        action = (local_reqs[step], forward_reqs[step], reject_reqs[step])
        excess = (excess_local[step], excess_forward_reject[step])

        excess_reject[step] = _calc_excess_reject_step(action,
                                                       excess,
                                                       queue_capacity[step])

    return excess_reject


def _get_excess_reject(iter, epi_idx, env, forward_capacity):
    result = {}
    for agent in env.agent_ids:
        queue_capacity = iter["hist_stats"]["observation_queue_capacity"][epi_idx][agent]
        local_reqs = iter["hist_stats"]["action_local"][epi_idx][agent]
        reject_reqs = iter["hist_stats"]["action_reject"][epi_idx][agent]
        excess_local = iter["hist_stats"]["excess_local"][epi_idx][agent]

        if env.type == "ASYM" and agent == "node_1":
            forward_reqs = np.zeros(env.max_steps, dtype=np.int32)
        else:
            forward_reqs = iter["hist_stats"]["action_forward"][epi_idx][agent]

        excess_reject = np.zeros(env.max_steps, dtype=np.int32)
        for step in range(env.max_steps):
            free_slots = queue_capacity[step] - local_reqs[step]
            if free_slots < 0:
                # There is a over-local processing.
                assert excess_local[step] == -free_slots, f"Expected equal, found {excess_local[step]} != {-free_slots}"
                excess = 0
            elif free_slots >= reject_reqs[step]:
                # All rejected requests could have been processed locally.
                excess = reject_reqs[step]
            else:
                # Some rejected requests could have been processed locally.
                excess = reject_reqs[step] - (reject_reqs[step] - free_slots)

            excess_reject[step] = excess

            if env.type == "ASYM" and agent == "node_1":
                continue

            free_slots = forward_capacity[agent][step] - forward_reqs[step]
            reject_reqs_left = reject_reqs[step] - excess
            if free_slots < 0:
                # There is a over-forwarding.
                pass
            elif free_slots >= reject_reqs_left:
                # All rejected requests could have been forwarded.
                excess += reject_reqs_left
            else:
                # Some rejected requests could have been processed locally.
                excess += reject_reqs_left - (reject_reqs_left - free_slots)

            excess_reject[step] = excess

        result[agent] = excess_reject

    return result


def _get_forward_capacity(iter, episode_idx, env):
    """Returns the theoretical forward capacity for all agents from the given
    episode of the given iteration data.

    Note that this function is specialized for environments with only two
    agents."""
    assert env.agents == 2, "Only two agents are supported for this function"

    forward_capacity, input_reqs, queue_capacity = {}, {}, {}

    # node_0
    input_reqs = np.array(iter["hist_stats"]["observation_input_requests"][episode_idx]["node_1"], dtype=np.int32)
    queue_capacity = np.array(iter["hist_stats"]["observation_queue_capacity"][episode_idx]["node_1"], dtype=np.int32)
    forward_capacity["node_0"] = queue_capacity - input_reqs

    # node_1
    input_reqs = np.array(iter["hist_stats"]["observation_input_requests"][episode_idx]["node_0"], dtype=np.int32)
    queue_capacity = np.array(iter["hist_stats"]["observation_queue_capacity"][episode_idx]["node_0"], dtype=np.int32)
    forward_capacity["node_1"] = queue_capacity - input_reqs

    for agent in env.agent_ids:
        # The difference may be negative if there is not enough space in the
        # queue.
        np.clip(forward_capacity[agent], 0, None, out=forward_capacity[agent])

    return forward_capacity


def _get_data(eval_dir, episode_idx):
    # Read data from the evaluation directory.
    iters = dfaas_utils.parse_result_file(eval_dir / "evaluation.json")
    iter = iters[0]["evaluation"]  # There is only one iteration.
    env = plot_utils.get_env(eval_dir)

    assert env.agents == 2, "Only two agents are supported for this plot"

    # Get the data from the specified episode index.
    input_requests = iter["hist_stats"]["observation_input_requests"][episode_idx]
    queue_capacity = iter["hist_stats"]["observation_queue_capacity"][episode_idx]
    forward_capacity = _get_forward_capacity(iter, episode_idx, env)
    action_local = iter["hist_stats"]["action_local"][episode_idx]
    action_forward = iter["hist_stats"]["action_forward"][episode_idx]
    action_reject = iter["hist_stats"]["action_reject"][episode_idx]
    excess_local = iter["hist_stats"]["excess_local"][episode_idx]
    forward_reject = iter["hist_stats"]["excess_forward_reject"][episode_idx]
    excess_reject = _get_excess_reject(iter, episode_idx, env, forward_capacity)
    reward = iter["hist_stats"]["reward"][episode_idx]

    data = {}

    if "hashes" in iter["hist_stats"]:
        # This evaluation was done with real traces, so I need to save the hases
        # for reproducibility.
        data["hashes"] = iter["hist_stats"]["hashes"][episode_idx]
    else:
        data["seed"] = iter["hist_stats"]["seed"][episode_idx]

    data["input_requests"] = input_requests
    data["queue_capacity"] = queue_capacity
    data["forward_capacity"] = forward_capacity

    data["action_local"], data["action_reject"], data["action_forward"] = {}, {}, {}
    data["excess_local"], data["forward_reject"], data["excess_reject"] = {}, {}, {}
    for agent in env.agent_ids:
        data["action_local"][agent] = np.array(action_local[agent], dtype=np.int32)
        data["action_reject"][agent] = np.array(action_reject[agent], dtype=np.int32)
        data["action_forward"][agent] = np.array(action_forward[agent], dtype=np.int32)

        data["excess_local"][agent] = np.array(excess_local[agent], dtype=np.int32)
        data["forward_reject"][agent] = np.array(forward_reject[agent], dtype=np.int32)
        data["excess_reject"][agent] = np.array(excess_reject[agent], dtype=np.int32)
    data["reward"] = reward

    return data


def make(eval_dir, episode_idx):
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    data = _get_data(eval_dir, episode_idx)
    env = plot_utils.get_env(eval_dir)

    # Coordinates of the x-bar for the bar plots. Must be explicitly calculated
    # from the number of steps for each agent.
    steps_x = np.arange(stop=env.max_steps)

    if "hashes" in data:
        hashes = [data["hashes"][agent][-10:] for agent in env.agent_ids]
        hash_str = ", ".join(hashes)
        title = f"Episode {episode_idx} of evaluation (function hashes {hash_str})"
    else:
        title = f"Episode {episode_idx} of evaluation (env seed {data['seed']})"

    fig = plt.figure(figsize=(32, 38), dpi=600, layout="constrained")
    fig.suptitle(title)
    # Set the reward plot to be smaller than the other ones.
    axs = fig.subplots(nrows=6, ncols=2,
                       gridspec_kw={"height_ratios": [2, 2, 2, 2, 2, 1]})

    # Make the same three plots for each agents.
    idx = 0
    for agent in env.agent_ids:
        local = data["action_local"][agent]
        reject = data["action_reject"][agent]
        forward = data["action_forward"][agent]

        axs[0, idx].bar(x=steps_x, height=local, label="Local")
        axs[0, idx].bar(x=steps_x, height=reject, label="Reject", bottom=local)
        axs[0, idx].bar(x=steps_x, height=forward, label="Forward", bottom=local+reject)
        axs[0, idx].plot(data["input_requests"][agent], linewidth=2, color="r", label="Input requests")
        axs[0, idx].set_title(f"Action ({agent})")
        axs[0, idx].set_ylabel("Requests")
        axs[0, idx].legend()

        excess_local = data["excess_local"][agent]
        real_local = local - excess_local
        axs[1, idx].bar(x=steps_x, height=real_local, color="g", label="Local")
        axs[1, idx].bar(x=steps_x, height=excess_local, color="r", label="Local excess", bottom=real_local)
        axs[1, idx].plot(data["queue_capacity"][agent], linewidth=2, color="b", label="Queue capacity")
        axs[1, idx].set_title(f"Local action ({agent})")
        axs[1, idx].set_ylabel("Requests")
        axs[1, idx].legend()

        forward_reject = data["forward_reject"][agent]
        real_forward = forward - forward_reject
        axs[2, idx].bar(x=steps_x, height=real_forward, color="g", label="Forward")
        axs[2, idx].bar(x=steps_x, height=forward_reject, color="r", label="Forward rejects", bottom=real_forward)
        axs[2, idx].plot(data["forward_capacity"][agent], linewidth=2, color="b", label="Forward capacity")
        axs[2, idx].set_title(f"Forward action ({agent})")
        axs[2, idx].set_ylabel("Requests")
        axs[2, idx].legend()

        excess_reject = data["excess_reject"][agent]
        good_reject = reject - excess_reject
        axs[3, idx].bar(x=steps_x, height=good_reject, color="g", label="Reject")
        axs[3, idx].bar(x=steps_x, height=excess_reject, color="r", label="Reject excess", bottom=good_reject)
        axs[3, idx].set_title(f"Reject action ({agent})")
        axs[3, idx].set_ylabel("Requests")
        axs[3, idx].legend()

        axs[4, idx].bar(x=steps_x, height=excess_local, label="Excess local")
        axs[4, idx].bar(x=steps_x, height=forward_reject, label="Forward reject", bottom=excess_local)
        axs[4, idx].bar(x=steps_x, height=excess_reject, label="Excess reject", bottom=excess_local+forward_reject)
        axs[4, idx].set_title(f"Total excess ({agent})")
        axs[4, idx].set_ylabel("Requests")
        axs[4, idx].legend()

        axs[5, idx].plot(data["reward"][agent], linewidth=2, label="Reward")
        axs[5, idx].set_title(f"Reward ({agent}) (max, min) = {env.reward_range}")
        axs[5, idx].set_ylabel("Reward")
        bottom, top = env.reward_range
        # Enforce a bit of space up and down, because if the reward is the
        # maximum or minimum, the line won't show up in the plot.
        axs[5, idx].set_ylim(bottom=bottom-.1, top=top+.1)
        axs[5, idx].yaxis.set_major_locator(ticker.MultipleLocator(.1))

        idx += 1

    # Common settings for the plots.
    for ax in axs.flat:
        ax.set_xlabel("Step")

        # Show x-axis ticks every 10 steps.
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = plots_dir / f"eval_episode_{episode_idx}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(dest="evaluation_dir", nargs="+", type=Path,
                        help="Evaluation directory (for the default evaluation, give the experiment directory")
    parser.add_argument("--episode", default=0, type=int,
                        help="Plot the given episode (a non-negative integer index)")

    args = parser.parse_args()

    for eval_dir in args.evaluation_dir:
        make(eval_dir.resolve(), args.episode)
