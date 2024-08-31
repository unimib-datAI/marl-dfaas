import argparse

from dfaas_env import DFaaS
import dfaas_utils

def choose_action(obs, info):
    action_abs = {agent: {} for agent in obs.keys()}
    action_abs["node_0"]["local_reqs"] = 0
    action_abs["node_0"]["forward_reqs"] = 0
    action_abs["node_0"]["reject_reqs"] = 0
    action_abs["node_1"]["local_reqs"] = 0
    action_abs["node_1"]["reject_reqs"] = 0

    for agent in obs.keys():
        input_reqs = obs[agent]["input_requests"]
        queue_capacity = obs[agent]["queue_capacity"]
        forward_capacity = obs[agent]["forward_capacity"]

        # Try to process all input requests locally.
        if input_reqs <= queue_capacity:
            action[agent]["local_reqs"] = input_reqs
            continue

        action[agent]["local_reqs"] = queue_capacity
        input_reqs = input_reqs - queue_capacity

        if agent == "node_0":
            # Try to forward all remaining input requests.
            if input_reqs < forward_capacity:
                action[agent]["forward_reqs"] = input_reqs
                continue

            action[agent]["forward_reqs"] = forward_capacity
            input_reqs = input_reqs - forward_capacity

        # Reject the remaining input requests.
        action[agent]["reject_reqs"] = input_reqs

    # Convert to distribution.

    # TODO

    action = {agent: np.zeros(3, dtype=float32) for agent in obs.keys()}
    return action

def main(out_dir, seed):
    out_dir = dfaas_utils.to_pathlib(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = DFaaS()
    obs, info = env.reset(options={"override_seed": seed})
    terminated = {"__all__": False}

    while not terminated["__all__"]:
        action = choose_action(obs, info)

        obs, reward, terminated, _, info = env.step(action)

        assert reward["node_0"] == 1.0 and reward["node_1"] == 1.0

if __name__ == "__main__":
    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="run_optimal")

    parser.add_argument(dest="out_dir",
                        help="Output directory")
    parser.add_argument(dest="seed",
                        help="Seed of the environment",
                        type=int)

    args = parser.parse_args()

    main(args.out_dir, args.seed)
