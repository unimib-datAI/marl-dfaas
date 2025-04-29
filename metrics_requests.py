# This Python script shows the sum, mean, and standard deviation from a pool of
# generated syntetic requests with the given seed and number of episodes.
#
# It is used to compare these traces with the real ones.
from pathlib import Path
import argparse

import dfaas_env
import dfaas_utils

import numpy as np


def main(env_config, seed, num_episodes):
    # Create a dummy env to generate the requests.
    env = dfaas_env.DFaaS(config=env_config)
    env.reset(seed=seed)  # Set the initial seed.

    # Each episode generates k series of input requests, where k is the number
    # of agents.
    length = num_episodes * len(env.agents)

    sum = np.empty(length, dtype=np.int32)
    mean = np.empty(length)
    std = np.empty(length)
    for episode in range(num_episodes):
        # Generate new requests.
        env.reset()

        # Save each input request to the right global slot in the arrays.
        agent_offset = 0
        for agent in env.agents:
            idx = episode * len(env.agents) + agent_offset

            sum[idx] = np.sum(env.input_requests[agent])
            mean[idx] = np.average(env.input_requests[agent])
            std[idx] = np.std(env.input_requests[agent])

            agent_offset += 1

    reqs_type = env.get_config()["input_requests_type"]
    print(f"Input requests of {reqs_type = } with {seed = } for {num_episodes = }")
    for metric, data in {"sum": sum, "mean": mean, "std": std}.items():
        # Show the result on the standard output. Maybe in the future can be
        # saved to a CSV file.
        print(f"Metrics for {metric!r} column")
        print(f"  sum  = {np.sum(data)}")
        print(f"  mean = {np.average(data)}")
        print(f"  std  = {np.std(data)}")


if __name__ == "__main__":
    # Create parser and parse arguments from the command line.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--seeds",
        nargs="+",
        default=42,
        type=int,
        help="Seeds used to generated synthetic requests",
    )
    parser.add_argument("--episodes", default=10000, type=int, help="Number of episodes to generate")
    parser.add_argument(
        "--env-config",
        type=Path,
        help="Environment config file (if None uses default config)",
    )

    args = parser.parse_args()

    for seed in args.seeds:
        assert seed >= 0, f"Seed must be a non-negative integer, found {seed}"
    assert args.episodes > 0, "Episodes to generate must be a positive integer"

    if args.env_config is not None:
        env_config = dfaas_utils.json_to_dict(args.env_config)
    else:
        env_config = {}
        print("WARNING: No given env config, using default!")

    for seed in args.seeds:
        main(env_config, seed, args.episodes)
        print()
