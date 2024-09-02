import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2

from dfaas_env import DFaaS
from dfaas_callbacks import DFaaSCallbacks
import dfaas_utils


def choose_action(obs):
    """Returns the action distribution for the given observation. The action is
    the "perfect action" that always maximizes the reward for all agents."""
    action = {agent: {} for agent in obs.keys()}
    action["node_0"]["local"] = 0
    action["node_0"]["forward"] = 0
    action["node_0"]["reject"] = 0
    action["node_1"]["local"] = 0
    action["node_1"]["reject"] = 0

    for agent in obs.keys():
        input_reqs = obs[agent]["input_requests"]
        queue_capacity = obs[agent]["queue_capacity"]

        # Try to process all input requests locally.
        if input_reqs <= queue_capacity:
            action[agent]["local"] = input_reqs
            continue

        # Otherwise, fill the queue and get the remaining input requests.
        action[agent]["local"] = queue_capacity
        input_reqs = input_reqs - queue_capacity

        if agent == "node_0":
            forward_capacity = obs[agent]["forward_capacity"]

            # Try to forward all remaining input requests.
            if input_reqs < forward_capacity:
                action[agent]["forward"] = input_reqs
                continue

            # Otherwise, forward all possibile requests and get the remaining
            # input requests.
            action[agent]["forward"] = forward_capacity
            input_reqs = input_reqs - forward_capacity

        # Reject the remaining input requests.
        action[agent]["reject"] = input_reqs

    # The number of requests must be converted to a distribution, because this
    # is what the environment expects in each "step()" call.
    action_dist = {}
    input_reqs = obs["node_0"]["input_requests"]
    dist = (action["node_0"]["local"] / input_reqs,
            action["node_0"]["forward"] / input_reqs,
            action["node_0"]["reject"] / input_reqs)
    # flatten() is needed because some elements can be np.ndarray.
    action_dist["node_0"] = np.array(dist).flatten()
    input_reqs = obs["node_1"]["input_requests"]
    dist = (action["node_1"]["local"] / input_reqs,
            action["node_1"]["reject"] / input_reqs)
    action_dist["node_1"] = np.array(dist).flatten()

    return action_dist


def main(config):
    """Run an episode of the DFaaS environment with the optimal action at each step.

    The environment will be created with the given configuration. The seed is
    the only mandatory key.

    The result of the episode is stored in the results/optimal directory."""
    assert isinstance(config, dict), f"config must be a dictionary, it is {type(config)!r}"
    assert "seed" in config, "'seed' key is required in config"

    seed = config["seed"]

    # Write the result file to a specified directory.
    out_dir = Path.cwd() / "results" / "optimal"
    out_dir = out_dir / Path(f"DFAAS-MA_{seed}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # This dictionary contains all the data related to the run, which is then
    # saved as a JSON file in the results directory.
    result = {}

    env = DFaaS(config=config)

    # Save the env configuration to a JSON file.
    config = env.get_config()
    config_path = out_dir / "env_config.json"
    dfaas_utils.dict_to_json(config, config_path)
    print(f"Environment configuration saved to: {config_path.as_posix()!r}")

    # To store the data, the following code mimics Ray and reuses the same
    # callbacks used for training.
    episode = EpisodeV2(None, None, None)
    base_env = BaseEnv()
    # Mimics the "envs" attribute, is the only one used in the callbacks.
    base_env.envs = [env]

    result["episode_len"] = env.max_steps

    callbacks = DFaaSCallbacks()

    # Reset the environment and get the first observation.
    obs, info = env.reset(options={"override_seed": seed})
    episode.total_env_steps += 1  # Required by the callbacks.
    callbacks.on_episode_start(episode=episode, base_env=base_env)

    # Cycle all steps
    terminated = {"__all__": False}
    while not terminated["__all__"]:
        action = choose_action(obs)

        # The reward and info are stored internally in the environment.
        obs, _, terminated, _, _ = env.step(action)
        episode.total_env_steps += 1

        callbacks.on_episode_step(episode=episode, base_env=base_env)

    callbacks.on_episode_end(episode=episode)

    result["hist_stats"] = episode.hist_data
    # These attributes are used by the callbacks.
    result["sampler_results"] = {}
    result["episodes_this_iter"] = 1

    callbacks.on_train_result(algorithm=None, result=result)

    result_file = out_dir / "result.json"
    dfaas_utils.dict_to_json(result, result_file)
    print(f"Episode data saved to: {result_file.as_posix()!r}")


if __name__ == "__main__":
    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="run_optimal")

    parser.add_argument(dest="seed",
                        help="Seed of the environment",
                        type=int)

    args = parser.parse_args()

    config = {}
    config["seed"] = args.seed

    main(config)
