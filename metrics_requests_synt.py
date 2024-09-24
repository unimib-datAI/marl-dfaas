# This Python script shows the sum, mean, and standard deviation from a pool of
# generated syntetic requests with the given seed and number of episodes.
#
# It is used to compare these traces with the real ones.
import argparse

import dfaas_env

import numpy as np


def main(seed, episodes, requests_type):
    # Master seed used to generate the seeds for each episode.
    master_rng = np.random.default_rng(seed=seed)

    # Create a dummy env from which some info are extracted.
    env_config = {"input_requests_type": requests_type}
    dummy_env = dfaas_env.DFaaS(config=env_config)
    limits = {}  # Used by _synthetic_input_requests().
    for agent in dummy_env.agent_ids:
        limits[agent] = {
                "min": dummy_env.observation_space[agent]["input_requests"].low.item(),
                "max": dummy_env.observation_space[agent]["input_requests"].high.item()
                }

    # Each episode generates two input requests (if the env has two agents), so
    # the total number of input requests is greater.
    length = episodes * dummy_env.agents

    sum = np.empty(length, dtype=np.int32)
    mean = np.empty(length)
    std = np.empty(length)
    for episode in range(episodes):
        # Seed used only for this episode.
        seed = master_rng.integers(0, high=np.iinfo(np.uint32).max)

        rng = np.random.default_rng(seed=seed)

        if requests_type == "synthetic-sinusoidal":
            input_reqs = dfaas_env._synthetic_sinusoidal_input_requests(dummy_env.max_steps,
                                                                        dummy_env.agent_ids,
                                                                        limits,
                                                                        rng)
        else:
            input_reqs = dfaas_env._synthetic_normal_input_requests(dummy_env.max_steps,
                                                                    dummy_env.agent_ids,
                                                                    limits,
                                                                    rng)

        # Save each input request to the right global slot in the arrays.
        agent_offset = 0
        for agent in dummy_env.agent_ids:
            idx = episode * dummy_env.agents + agent_offset

            sum[idx] = np.sum(input_reqs[agent])
            mean[idx] = np.average(input_reqs[agent])
            std[idx] = np.std(input_reqs[agent])

            agent_offset += 1

    for (metric, data) in {"sum": sum, "mean": mean, "std": std}.items():
        # Show the result on the standard output. Maybe in the future can be
        # saved to a file.
        print(f"Metrics for {metric!r} column")
        print(f"  sum  = {np.sum(data)}")
        print(f"  mean = {np.average(data)}")
        print(f"  std  = {np.std(data)}")


if __name__ == "__main__":
    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="metrics_requests")

    parser.add_argument(dest="seed",
                        help="The seed to use when generating the syntetic requests",
                        type=int)
    parser.add_argument(dest="episodes",
                        help="How many episodes to generate",
                        type=int)
    parser.add_argument(dest="type",
                        help="Type of the input requests to generate")

    args = parser.parse_args()

    main(args.seed, args.episodes, args.type)
