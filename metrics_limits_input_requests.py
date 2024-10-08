# TODO
import argparse

import dfaas_env

import numpy as np


def main(type, seed):
    # Set function arguments to fixed generic values.
    env = dfaas_env.DFaaS()
    limits = {}
    for agent in env.agent_ids:
        limits[agent] = {
                "min": env.observation_space[agent]["input_requests"].low.item(),
                "max": env.observation_space[agent]["input_requests"].high.item()
                }

    # Create the RNG used to generate input requests.
    rng = np.random.default_rng(seed=seed)

    match type:
        case "synthetic-sinusoidal":
            input_reqs = dfaas_env._synthetic_sinusoidal_input_requests(env.max_steps,
                                                                        env.agent_ids,
                                                                        limits,
                                                                        rng)
        case "synthetic-normal":
            input_reqs = dfaas_env._synthetic_normal_input_requests(env.max_steps,
                                                                    env.agent_ids,
                                                                    limits,
                                                                    rng)
        case "real":
            input_reqs = dfaas_env._real_input_requests(env.max_steps,
                                                        env.agent_ids,
                                                        limits,
                                                        rng,
                                                        True)
        case _:
            assert False, f"Unsupported synthetic {type = }"

    # Only environments with the same queue capacity are supported for this
    # script.
    queue_capacity = np.unique(list(env.queue_capacity_max.values()))
    assert queue_capacity.size == 1, "Environments with different queue capacity size for the agents are not supported"
    total_capacity = queue_capacity.item() * env.agents

    # The maximium processable input requests for an episode.
    p_max = 0

    # The simple sum of input requests for an episode.
    r_in = 0

    # Seed used only for this episode.
    env.reset(seed=seed)
    for step in range(env.max_steps):
        input_reqs_step = 0  # Input requests for this step.
        for agent in env.agent_ids:
            input_reqs_step += input_reqs[agent][step]

        r_in += input_reqs_step

        if input_reqs_step <= total_capacity:
            p_max += input_reqs_step
        else:
            p_max += queue_capacity * env.agents

    p_max = p_max.item()  # Unwrap np.array to a single value.

    print(f"Input requests of {type = } with {seed = }")
    print(f"  {p_max = }")
    print(f"  {r_in = }")
    print(f"  queue_capacity * num. agents = {total_capacity * env.max_steps}")


if __name__ == "__main__":
    # Create parser and parse arguments from the command line.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("type", choices=["synthetic-sinusoidal",
                                         "synthetic-normal",
                                         "real"],
                        help="Type of input requests to generate/select")
    parser.add_argument("--seed", default=0, type=int,
                        help="Seed used to generated/select input requests")

    args = parser.parse_args()

    assert args.seed >= 0, "Seed must be a non-negative integer"

    main(args.type, args.seed)
