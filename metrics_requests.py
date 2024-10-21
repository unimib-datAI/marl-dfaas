# This Python script shows the sum, mean, and standard deviation from a pool of
# generated syntetic requests with the given seed and number of episodes.
#
# It is used to compare these traces with the real ones.
import argparse

import dfaas_env

import numpy as np


def main(type, seed, episodes):
    # Create a dummy env to generate the requests.
    env = dfaas_env.DFaaS(config={"input_requests_type": type})
    env.reset(seed=seed)  # Set the initial seed.

    # Each episode generates two input requests (if the env has two agents), so
    # the total number of input requests is greater.
    length = episodes * env.agents

    sum = np.empty(length, dtype=np.int32)
    mean = np.empty(length)
    std = np.empty(length)
    for episode in range(episodes):
        # Generate new requests.
        env.reset()

        # Save each input request to the right global slot in the arrays.
        agent_offset = 0
        for agent in env.agent_ids:
            idx = episode * env.agents + agent_offset

            sum[idx] = np.sum(env.input_requests[agent])
            mean[idx] = np.average(env.input_requests[agent])
            std[idx] = np.std(env.input_requests[agent])

            agent_offset += 1

    print(f"Input requests of {type = } with {seed = } for {episodes = }")
    for (metric, data) in {"sum": sum, "mean": mean, "std": std}.items():
        # Show the result on the standard output. Maybe in the future can be
        # saved to a CSV file.
        print(f"Metrics for {metric!r} column")
        print(f"  sum  = {np.sum(data)}")
        print(f"  mean = {np.average(data)}")
        print(f"  std  = {np.std(data)}")


if __name__ == "__main__":
    # Create parser and parse arguments from the command line.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("type", choices=["synthetic-sinusoidal",
                                         "synthetic-normal",
                                         "real"],
                        help="Type of input requests to generate")
    parser.add_argument("--seed", default=0, type=int,
                        help="Seed used to generated synthetic requests")
    parser.add_argument("--episodes", default=10000, type=int,
                        help="Number of episodes to generate")

    args = parser.parse_args()

    assert args.seed >= 0, "Seed must be a non-negative integer"
    assert args.episodes > 0, "Episodes to generate must be a positive integer"

    main(args.type, args.seed, args.episodes)
