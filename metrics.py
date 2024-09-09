# This Python script calculates additional metrics from the 'result.json' file
# in the experiment output directory and saves the data to the
# 'metrics-result.json' file in the same directory.
#
# Some scripts related to plotting require this script to be run before
# plotting.
import argparse
from pathlib import Path
import logging

import numpy as np

import dfaas_utils

# Initialize logger for this module.
logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s", level=logging.DEBUG)
logger = logging.getLogger(Path(__file__).name)


def get_agents():
    """Returns the agent IDs."""
    # TODO: Make dynamic.
    return ["node_0", "node_1"]


def num_episodes(iters):
    """Returns the number of episodes in each iteration.

    The value is extracted from the given iter data. It is assumed that each
    iteration has the same number of episodes and that there is at least one
    iteration in the given data."""
    return iters[0]["episodes_this_iter"]


def average_reward_per_step(iters, metrics):
    """Calculates average reward per step per episode."""
    agents = get_agents()

    # Each item is an iteration.
    metrics["reward_average_per_step"] = []
    for iter in iters:
        # Each item is an episode.
        reward_iter = []
        for episode in iter["hist_stats"]["reward"]:
            reward_episode = {}
            for agent in agents:
                reward_episode[agent] = np.average(episode[agent])

            reward_iter.append(reward_episode)

        metrics["reward_average_per_step"].append(reward_iter)


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

    TODO: This function should not exist, the environment should provide this
    value."""
    input_reqs = episode_data["observation_input_requests"][epi_idx][agent]
    queue_capacity = episode_data["observation_queue_capacity"][epi_idx][agent]
    local_reqs = episode_data["action_local"][epi_idx][agent]
    reject_reqs = episode_data["action_reject"][epi_idx][agent]
    excess_local = episode_data["excess_local"][epi_idx][agent]

    forward_reqs = episode_data["action_forward"][epi_idx][agent]
    excess_forward_reject = episode_data["excess_forward_reject"][epi_idx][agent]

    excess_reject = np.zeros(len(input_reqs), dtype=np.int32)
    for step in range(len(input_reqs)):
        action = (local_reqs[step], forward_reqs[step], reject_reqs[step])
        excess = (excess_local[step], excess_forward_reject[step])

        excess_reject[step] = _calc_excess_reject_step(action,
                                                       excess,
                                                       queue_capacity[step])

    return excess_reject


def reqs_exceed_per_step(iters, metrics):
    """Calculates the following metrics:

        1. Percentage of local requests that exceed queue capacity out of all
        local requests.

        2. Percentage of forwarded requests that are rejected out of all
        forwarded requests.

        3. Percentage of excess rejected requests out of all rejected requests.

    The metrics are calculated for each step."""
    agents = get_agents()

    metrics["local_reqs_percent_excess_per_step"] = []
    metrics["reject_reqs_percent_excess_per_step"] = []
    metrics["forward_reject_reqs_percent_per_step"] = []

    # If an action is zero, we get a NaN or other invalid values, but can safely
    # convert to zero percent because the excess reject is also zero.
    old = np.seterr(invalid="ignore")

    for iter in iters:
        local_iter, reject_iter, forward_reject_iter = [], [], []
        for epi_idx in range(num_episodes(iters)):
            local_epi, reject_epi, forward_reject_epi = {}, {}, {}
            for agent in agents:
                # Calculation of local excess percent.
                # Note these are NumPy arrays.
                local_reqs = np.array(iter["hist_stats"]["action_local"][epi_idx][agent], dtype=np.int32)
                excess_local = np.array(iter["hist_stats"]["excess_local"][epi_idx][agent], dtype=np.int32)
                local_epi[agent] = excess_local * 100 / local_reqs
                local_epi[agent] = np.nan_to_num(local_epi[agent], posinf=0.0, neginf=0.0)

                # Calculation of reject excess percent.
                excess_reject = calc_excess_reject(iter["hist_stats"], agent, epi_idx)
                action_reject = np.array(iter["hist_stats"]["action_reject"][epi_idx][agent], dtype=np.int32)
                reject_epi[agent] = excess_reject * 100 / action_reject
                reject_epi[agent] = np.nan_to_num(reject_epi[agent], posinf=0.0, neginf=0.0)

                # Forwarded requests.
                forward_reject = np.array(iter["hist_stats"]["excess_forward_reject"][epi_idx][agent], dtype=np.int32)
                action_forward = np.array(iter["hist_stats"]["action_forward"][epi_idx][agent], dtype=np.int32)
                forward_reject_epi[agent] = forward_reject * 100 / action_forward
                forward_reject_epi[agent] = np.nan_to_num(forward_reject_epi["node_0"], posinf=0.0, neginf=0.0)

            local_iter.append(local_epi)
            reject_iter.append(reject_epi)
            forward_reject_iter.append(forward_reject_epi)

        metrics["local_reqs_percent_excess_per_step"].append(local_iter)
        metrics["reject_reqs_percent_excess_per_step"].append(reject_iter)
        metrics["forward_reject_reqs_percent_per_step"].append(forward_reject_iter)

    # Reset the NumPy error handling.
    np.seterr(invalid=old["invalid"])


def average_reqs_percent_exceed_per_episode(iters, metrics):
    """For each episode, for each iteration, calculates the average percentage
    of rejected requests that could have been handled locally but were rejected
    by the agent, and the same percentage but for local requests that exceed the
    local buffer."""
    agents = get_agents()

    metrics["local_reqs_percent_excess_per_episode"] = []
    metrics["reject_reqs_percent_excess_per_episode"] = []
    metrics["forward_reject_reqs_percent_per_episode"] = []

    for iter_idx in range(len(iters)):
        local_iter, reject_iter, forward_reject_iter = [], [], []

        for epi_idx in range(num_episodes(iters)):
            local_epi, reject_epi, forward_reject_epi = {}, {}, {}

            for agent in agents:
                tmp = metrics["local_reqs_percent_excess_per_step"][iter_idx][epi_idx][agent]
                local_epi[agent] = np.average(tmp)

                tmp = metrics["reject_reqs_percent_excess_per_step"][iter_idx][epi_idx][agent]
                reject_epi[agent] = np.average(tmp)

                tmp = metrics["forward_reject_reqs_percent_per_step"][iter_idx][epi_idx][agent]
                forward_reject_epi[agent] = np.average(tmp)

            local_iter.append(local_epi)
            reject_iter.append(reject_epi)
            forward_reject_iter.append(forward_reject_epi)

        metrics["local_reqs_percent_excess_per_episode"].append(local_iter)
        metrics["reject_reqs_percent_excess_per_episode"].append(reject_iter)
        metrics["forward_reject_reqs_percent_per_episode"].append(forward_reject_iter)


def average_reqs_percent_exceed_per_iteration(iters, metrics):
    """For each iteration, calculates the average percentage of rejected
    requests that could have been handled locally but were rejected by the
    agent, and the same percentage but for local requests that exceed the local
    buffer."""
    agents = get_agents()
    episodes = num_episodes(iters)

    metrics["local_reqs_percent_excess_per_iteration"] = []
    metrics["reject_reqs_percent_excess_per_iteration"] = []
    metrics["forward_reject_reqs_percent_per_iteration"] = []

    for iter_idx in range(len(iters)):
        local_iter, reject_iter, forward_reject_iter = {}, {}, {}
        for agent in agents:
            local_iter[agent] = np.empty(episodes, np.float32)
            reject_iter[agent] = np.empty(episodes, np.float32)
            forward_reject_iter[agent] = np.empty(episodes, np.float32)

        for epi_idx in range(episodes):
            for agent in agents:
                tmp = metrics["local_reqs_percent_excess_per_episode"][iter_idx][epi_idx][agent]
                local_iter[agent][epi_idx] = tmp

                tmp = metrics["reject_reqs_percent_excess_per_episode"][iter_idx][epi_idx][agent]
                reject_iter[agent][epi_idx] = tmp

                tmp = metrics["forward_reject_reqs_percent_per_episode"][iter_idx][epi_idx][agent]
                forward_reject_iter[agent][epi_idx] = tmp

        local_avg, reject_avg, forward_reject_avg = {}, {}, {}
        for agent in agents:
            local_avg[agent] = np.average(local_iter[agent])
            reject_avg[agent] = np.average(reject_iter[agent])
            forward_reject_avg[agent] = np.average(forward_reject_iter[agent])

        metrics["local_reqs_percent_excess_per_iteration"].append(local_avg)
        metrics["reject_reqs_percent_excess_per_iteration"].append(reject_avg)
        metrics["forward_reject_reqs_percent_per_iteration"].append(forward_reject_avg)


def reject_reqs_excess_per_step(iters, metrics):
    """For each step, for each episode, for each iteration, calculates the
    absolute number of rejected requests that could have been handled locally
    but were rejected by the agent.

    TODO: This function should not exist, the environment should provide this
    value."""
    agents = get_agents()

    metrics["reject_excess_per_step"] = []

    for iter in iters:
        excess_iter = []

        for epi_idx in range(num_episodes(iters)):
            excess_epi = {}

            for agent in agents:
                tmp = calc_excess_reject(iter["hist_stats"], agent, epi_idx)
                excess_epi[agent] = tmp

            excess_iter.append(excess_epi)

        metrics["reject_excess_per_step"].append(excess_iter)


def main(exp_dir):
    exp_dir = dfaas_utils.to_pathlib(exp_dir)

    # Each element is one iteration.
    iters = dfaas_utils.parse_result_file(exp_dir / "result.json")

    # Each key in the metrics dictionary is a calculated metric. Note that the
    # value can be a subdictionary or a sublist.
    metrics = {}

    average_reward_per_step(iters, metrics)

    reqs_exceed_per_step(iters, metrics)

    average_reqs_percent_exceed_per_episode(iters, metrics)

    average_reqs_percent_exceed_per_iteration(iters, metrics)

    reject_reqs_excess_per_step(iters, metrics)

    # Save the metrics dictionary to disk as a JSON file.
    metrics_path = exp_dir / "metrics-result.json"
    dfaas_utils.dict_to_json(metrics, metrics_path)
    logger.info(f"Metrics data saved to: {metrics_path.as_posix()!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="metrics")

    parser.add_argument(dest="exp_directory",
                        help="Directory with the results.json file")

    args = parser.parse_args()

    main(args.exp_directory)
