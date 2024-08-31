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


def calc_excess_reject(episode_data, agent, epi_idx):
    """Returns the number of rejected requests in excess of the allowed rejected
    requests expected for each step in the episode.

    TODO: This function should not exist, the environment should provide this
    value."""
    input_reqs = episode_data["observation_input_requests"][epi_idx][agent]
    local_reqs = episode_data["action_local"][epi_idx][agent]
    reject_reqs = episode_data["action_reject"][epi_idx][agent]

    if agent == "node_0":
        forward_reqs = episode_data["action_forward"][epi_idx][agent]
        forward_capacity = episode_data["observation_forward_capacity"][epi_idx][agent]
    else:
        steps = len(input_reqs)
        forward_reqs = np.zeros(steps)
        forward_capacity = np.zeros(steps)

    excess_reject = np.empty(len(input_reqs), dtype=np.int32)
    for step in range(len(input_reqs)):
        # All rejected requests are excessive.
        if input_reqs[step] <= 100:  # Size of the queue.
            excess_reject[step] = reject_reqs[step]
            continue

        # We calculate the slots that are not used to process a request locally
        # or to route it. Note that the value may be negative if there is
        # over-forwarding or over-local processing: in this case it must be set
        # to zero.
        free_local = np.clip(100 - local_reqs[step], 0, 100)
        free_forward = np.clip(forward_capacity[step] - forward_reqs[step], 0, forward_capacity[step])

        # If the result is positive, it means that some requests were rejected
        # but should be processed.
        excess_reject_step = free_local + free_forward - reject_reqs[step]

        excess_reject[step] = np.clip(excess_reject_step, 0, reject_reqs[step])

    return excess_reject


def reqs_exceed_per_step(iters, metrics):
    """Calculates the following metrics:

    1. Percentage of local requests that exceed queue capacity.
    2. Percentage of forwarded requests that exceed forwarding capacity.
    3. Percentage of rejected requests that could have been forwarded or handled
    locally, but were rejected.

    Percentages are calculated for each step."""
    agents = get_agents()
    steps = 100  # TODO: Extract this value from data, not as fixed constant!

    metrics["local_reqs_percent_excess_per_step"] = []
    metrics["reject_reqs_percent_excess_per_step"] = []
    metrics["forward_reqs_percent_excess_per_step"] = []
    metrics["forward_reqs_percent_reject_per_step"] = []

    for iter in iters:
        local_iter, reject_iter, forward_iter, forward_reject_iter = [], [], [], []
        for epi_idx in range(num_episodes(iters)):
            local_epi, reject_epi, forward_epi, forward_reject_epi = {}, {}, {}, {}
            for agent in agents:
                # Calculation of local excess percent.
                excess_local = np.array(iter["hist_stats"]["excess_local"][epi_idx][agent], dtype=np.int32)

                # TODO: Extract this value from data, not as fixed constant!
                queue_capacity_max = 100

                excess_local_perc = excess_local * 100 / queue_capacity_max
                local_epi[agent] = excess_local_perc

                # Calculation of reject excess percent.
                excess_reject = calc_excess_reject(iter["hist_stats"], agent, epi_idx)
                action_reject = np.array(iter["hist_stats"]["action_reject"][epi_idx][agent], dtype=np.int32)

                # If the action is zero, we get a NaN or other invalid values,
                # but can safely convert to zero percent because the excess
                # reject is also zero.
                old = np.seterr(invalid="ignore")
                excess_reject_perc = excess_reject * 100 / action_reject
                np.seterr(invalid=old["invalid"])
                reject_epi[agent] = np.nan_to_num(excess_reject_perc, posinf=0.0, neginf=0.0)

            # Forwarded requests only for "node_0".
            excess_forward = np.array(iter["hist_stats"]["excess_forward"][epi_idx]["node_0"], dtype=np.int32)
            action_forward = np.array(iter["hist_stats"]["action_forward"][epi_idx]["node_0"], dtype=np.int32)

            old = np.seterr(invalid="ignore")
            excess_forward_perc = excess_forward * 100 / action_forward
            np.seterr(invalid=old["invalid"])
            forward_epi = {"node_0": np.nan_to_num(excess_forward_perc, posinf=0.0, neginf=0.0),
                           "node_1": np.zeros(steps, dtype=np.float32)}

            # Rejected forwarded requests only for "node_0".
            reject_reqs = np.array(iter["hist_stats"]["excess_forward_reject"][epi_idx]["node_0"], dtype=np.int32)

            # First, we need to get the actual forwarded requests.
            actual_forward = action_forward - excess_forward

            old = np.seterr(invalid="ignore")
            reject_reqs_perc = reject_reqs * 100 / actual_forward
            np.seterr(invalid=old["invalid"])
            forward_reject_epi = {"node_0": np.nan_to_num(reject_reqs_perc, posinf=0.0, neginf=0.0),
                                  "node_1": np.zeros(steps, dtype=np.float32)}

            local_iter.append(local_epi)
            reject_iter.append(reject_epi)
            forward_iter.append(forward_epi)
            forward_reject_iter.append(forward_reject_epi)

        metrics["local_reqs_percent_excess_per_step"].append(local_iter)
        metrics["reject_reqs_percent_excess_per_step"].append(reject_iter)
        metrics["forward_reqs_percent_excess_per_step"].append(forward_iter)
        metrics["forward_reqs_percent_reject_per_step"].append(forward_reject_iter)


def average_reqs_percent_exceed_per_episode(iters, metrics):
    """For each episode, for each iteration, calculates the average percentage
    of rejected requests that could have been handled locally but were rejected
    by the agent, and the same percentage but for local requests that exceed the
    local buffer."""
    agents = get_agents()

    metrics["local_reqs_percent_excess_per_episode"] = []
    metrics["reject_reqs_percent_excess_per_episode"] = []
    metrics["forward_reqs_percent_excess_per_episode"] = []
    metrics["forward_reqs_percent_reject_per_episode"] = []

    for iter_idx in range(len(iters)):
        local_iter, reject_iter, forward_iter, forward_reject_iter = [], [], [], []

        for epi_idx in range(num_episodes(iters)):
            local_epi, reject_epi, forward_epi, forward_reject_epi = {}, {}, {}, {}

            for agent in agents:
                tmp = metrics["local_reqs_percent_excess_per_step"][iter_idx][epi_idx][agent]
                local_epi[agent] = np.average(tmp)

                tmp = metrics["reject_reqs_percent_excess_per_step"][iter_idx][epi_idx][agent]
                reject_epi[agent] = np.average(tmp)

                tmp = metrics["forward_reqs_percent_excess_per_step"][iter_idx][epi_idx][agent]
                forward_epi[agent] = np.average(tmp)

                tmp = metrics["forward_reqs_percent_reject_per_step"][iter_idx][epi_idx][agent]
                forward_reject_epi[agent] = np.average(tmp)

            local_iter.append(local_epi)
            reject_iter.append(reject_epi)
            forward_iter.append(forward_epi)
            forward_reject_iter.append(forward_reject_epi)

        metrics["local_reqs_percent_excess_per_episode"].append(local_iter)
        metrics["reject_reqs_percent_excess_per_episode"].append(reject_iter)
        metrics["forward_reqs_percent_excess_per_episode"].append(forward_iter)
        metrics["forward_reqs_percent_reject_per_episode"].append(forward_reject_iter)


def average_reqs_percent_exceed_per_iteration(iters, metrics):
    """For each iteration, calculates the average percentage of rejected
    requests that could have been handled locally but were rejected by the
    agent, and the same percentage but for local requests that exceed the local
    buffer."""
    agents = get_agents()
    episodes = num_episodes(iters)

    metrics["local_reqs_percent_excess_per_iteration"] = []
    metrics["reject_reqs_percent_excess_per_iteration"] = []
    metrics["forward_reqs_percent_excess_per_iteration"] = []
    metrics["forward_reqs_percent_reject_per_iteration"] = []

    for iter_idx in range(len(iters)):
        local_iter, reject_iter, forward_iter, forward_reject_iter = {}, {}, {}, {}
        for agent in agents:
            local_iter[agent] = np.empty(episodes, np.float32)
            reject_iter[agent] = np.empty(episodes, np.float32)
            forward_iter[agent] = np.empty(episodes, np.float32)
            forward_reject_iter[agent] = np.empty(episodes, np.float32)

        for epi_idx in range(episodes):
            for agent in agents:
                tmp = metrics["local_reqs_percent_excess_per_episode"][iter_idx][epi_idx][agent]
                local_iter[agent][epi_idx] = tmp

                tmp = metrics["reject_reqs_percent_excess_per_episode"][iter_idx][epi_idx][agent]
                reject_iter[agent][epi_idx] = tmp

                tmp = metrics["forward_reqs_percent_excess_per_episode"][iter_idx][epi_idx][agent]
                forward_iter[agent][epi_idx] = tmp

                tmp = metrics["forward_reqs_percent_reject_per_episode"][iter_idx][epi_idx][agent]
                forward_reject_iter[agent][epi_idx] = tmp

        local_avg, reject_avg, forward_avg, forward_reject_avg = {}, {}, {}, {}
        for agent in agents:
            local_avg[agent] = np.average(local_iter[agent])
            reject_avg[agent] = np.average(reject_iter[agent])
            forward_avg[agent] = np.average(forward_iter[agent])
            forward_reject_avg[agent] = np.average(forward_reject_iter[agent])

        metrics["local_reqs_percent_excess_per_iteration"].append(local_avg)
        metrics["reject_reqs_percent_excess_per_iteration"].append(reject_avg)
        metrics["forward_reqs_percent_excess_per_iteration"].append(forward_avg)
        metrics["forward_reqs_percent_reject_per_iteration"].append(forward_reject_avg)


def abs_reqs_exceed_per_step(iters, metrics):
    """For each step, for each episode, for each iteration, calculates the
    absolute number of rejected requests that could have been handled locally
    but were rejected by the agent, and the same percentage but for local
    requests that exceed the local buffer."""
    agents = get_agents(iters)

    for iter in range(len(metrics["iterations"])):
        reject_iter_data = []
        local_iter_data = []
        for episode in range(len(iters[iter]["hist_stats"]["seed"])):
            reject_episode_data = {}
            local_episode_data = {}
            for agent in agents:
                steps = len(iters[iter]["hist_stats"]["input_requests"][episode][agent])

                input_reqs = np.asarray(iters[iter]["hist_stats"]["input_requests"][episode][agent],
                                        dtype=np.int32)
                reject_reqs = np.asarray(iters[iter]["hist_stats"]["action"][episode]["reject"][agent],
                                         dtype=np.int32)

                reject_reqs_exc = np.empty(steps, dtype=np.int32)
                local_reqs_exc = np.empty(steps, dtype=np.int32)

                for step in range(steps):
                    local_reqs_step = input_reqs[step] - reject_reqs[step]
                    # TODO: get the max requests to process locally dinamically
                    # from the environment or some config.
                    if local_reqs_step > 100:
                        local_reqs_exc[step] = local_reqs_step - 100
                        reject_reqs_exc[step] = 0
                        continue

                    if input_reqs[step] > 100:
                        # The result may be negative if the policy attempts to
                        # process more requests locally than possible. This is
                        # why there is an np.clip() call at the end of the
                        # cycle.
                        reject_reqs_exc[step] = reject_reqs[step] - (input_reqs[step] - 100)
                        local_reqs_exc[step] = 0
                        continue

                    # All rejected requests are excessive.
                    reject_reqs_exc[step] = reject_reqs[step]
                    local_reqs_exc[step] = 0

                reject_episode_data[agent] = reject_reqs_exc
                local_episode_data[agent] = local_reqs_exc

            reject_iter_data.append(reject_episode_data)
            local_iter_data.append(local_episode_data)

        metrics["iterations"][iter]["local_reqs_exceed_per_step"] = local_iter_data
        metrics["iterations"][iter]["reject_reqs_exceed_per_step"] = reject_iter_data


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

    # abs_reqs_exceed_per_step(iters, metrics)

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
