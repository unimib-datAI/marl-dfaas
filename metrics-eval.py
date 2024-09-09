# This Python script calculates additional metrics from the
# 'final_evaluation.json' file in the experiment output directory and saves the
# data to the 'metrics-final_evaluation.json' file in the same directory.
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


def num_episodes(eval):
    """Returns the number of episodes in the evaluation.

    The value is extracted from the given evaluation data."""
    return eval["episodes_this_iter"]


def average_reward_per_step(eval, metrics):
    """Calculates average reward per step per episode."""
    agents = get_agents()

    # Each item is an iteration.
    metrics["reward_average_per_step"] = []
    for episode in eval["hist_stats"]["reward"]:
        reward_episode = {}
        for agent in agents:
            reward_episode[agent] = np.average(episode[agent])

        metrics["reward_average_per_step"].append(reward_episode)


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


def reqs_exceed_per_step(eval, metrics):
    """Calculates the following metrics:

        1. Percentage of local requests that exceed queue capacity out of all
        local requests.

        2. Percentage of forwarded requests that are rejected out of all
        forwarded requests.

        3. Percentage of excess rejected requests out of all rejected requests.

    The metrics are calculated for each step."""
    agents = get_agents()
    steps = 100  # TODO: Extract this value from data, not as fixed constant!

    # If an action is zero, we get a NaN or other invalid values, but can safely
    # convert to zero percent because the excess reject is also zero.
    old = np.seterr(invalid="ignore")

    local_iter, reject_iter, forward_reject_iter = [], [], []
    for epi_idx in range(num_episodes(eval)):
        local_epi, reject_epi, forward_reject_epi = {}, {}, {}
        for agent in agents:
            # Calculation of local excess percent.
            # Note these are NumPy arrays.
            local_reqs = np.array(eval["hist_stats"]["action_local"][epi_idx][agent], dtype=np.int32)
            excess_local = np.array(eval["hist_stats"]["excess_local"][epi_idx][agent], dtype=np.int32)
            local_epi[agent] = excess_local * 100 / local_reqs
            local_epi[agent] = np.nan_to_num(local_epi[agent], posinf=0.0, neginf=0.0)

            # Calculation of reject excess percent.
            excess_reject = calc_excess_reject(eval["hist_stats"], agent, epi_idx)
            action_reject = np.array(eval["hist_stats"]["action_reject"][epi_idx][agent], dtype=np.int32)
            reject_epi[agent] = excess_reject * 100 / action_reject
            reject_epi[agent] = np.nan_to_num(reject_epi[agent], posinf=0.0, neginf=0.0)

        # Forwarded requests only for "node_0".
        forward_reject_epi["node_1"] = np.zeros(steps, dtype=np.float32)
        forward_reject = np.array(eval["hist_stats"]["excess_forward_reject"][epi_idx]["node_0"], dtype=np.int32)
        action_forward = np.array(eval["hist_stats"]["action_forward"][epi_idx]["node_0"], dtype=np.int32)
        forward_reject_epi["node_0"] = forward_reject * 100 / action_forward
        forward_reject_epi["node_0"] = np.nan_to_num(forward_reject_epi["node_0"], posinf=0.0, neginf=0.0)

        local_iter.append(local_epi)
        reject_iter.append(reject_epi)
        forward_reject_iter.append(forward_reject_epi)

    metrics["local_reqs_percent_excess_per_step"] = local_iter
    metrics["reject_reqs_percent_excess_per_step"] = reject_iter
    metrics["forward_reject_reqs_percent_per_step"] = forward_reject_iter

    # Reset the NumPy error handling.
    np.seterr(invalid=old["invalid"])


def average_reqs_percent_exceed_per_episode(eval, metrics):
    """For each episode, for each iteration, calculates the average percentage
    of rejected requests that could have been handled locally but were rejected
    by the agent, and the same percentage but for local requests that exceed the
    local buffer."""
    agents = get_agents()

    local_iter, reject_iter, forward_reject_iter = [], [], []
    for epi_idx in range(num_episodes(eval)):
        local_epi, reject_epi, forward_reject_epi = {}, {}, {}

        for agent in agents:
            tmp = metrics["local_reqs_percent_excess_per_step"][epi_idx][agent]
            local_epi[agent] = np.average(tmp)

            tmp = metrics["reject_reqs_percent_excess_per_step"][epi_idx][agent]
            reject_epi[agent] = np.average(tmp)

            tmp = metrics["forward_reject_reqs_percent_per_step"][epi_idx][agent]
            forward_reject_epi[agent] = np.average(tmp)

        local_iter.append(local_epi)
        reject_iter.append(reject_epi)
        forward_reject_iter.append(forward_reject_epi)

    metrics["local_reqs_percent_excess_per_episode"] = local_iter
    metrics["reject_reqs_percent_excess_per_episode"] = reject_iter
    metrics["forward_reject_reqs_percent_per_episode"] = forward_reject_iter


def reject_reqs_excess_per_step(eval, metrics):
    """For each step, for each episode, calculates the absolute number of
    rejected requests that could have been handled locally but were rejected by
    the agent.

    TODO: This function should not exist, the environment should provide this
    value."""
    agents = get_agents()

    metrics["reject_excess_per_step"] = []

    for epi_idx in range(num_episodes(eval)):
        excess_epi = {}

        for agent in agents:
            tmp = calc_excess_reject(eval["hist_stats"], agent, epi_idx)
            excess_epi[agent] = tmp

        metrics["reject_excess_per_step"].append(excess_epi)


def main(exp_dir):
    exp_dir = dfaas_utils.to_pathlib(exp_dir)

    eval = dfaas_utils.parse_result_file(exp_dir / "final_evaluation.json")[0]["evaluation"]

    # Each key in the metrics dictionary is a calculated metric. Note that the
    # value can be a subdictionary or a sublist.
    metrics = {}

    average_reward_per_step(eval, metrics)

    reqs_exceed_per_step(eval, metrics)

    average_reqs_percent_exceed_per_episode(eval, metrics)

    reject_reqs_excess_per_step(eval, metrics)

    # Save the metrics dictionary to disk as a JSON file.
    metrics_path = exp_dir / "metrics-final_evaluation.json"
    dfaas_utils.dict_to_json(metrics, metrics_path)
    logger.info(f"Metrics data saved to: {metrics_path.as_posix()!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="metrics-eval")

    parser.add_argument(dest="exp_directory",
                        help="Directory with the final_evaluation.json file")

    args = parser.parse_args()

    main(args.exp_directory)
