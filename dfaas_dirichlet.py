from pathlib import Path
import shutil
from datetime import datetime
import logging
import argparse

import numpy as np

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.connectors.agent.obs_preproc import ObsPreprocessorConnector
from ray.rllib.utils.typing import AgentConnectorDataType
from ray.rllib.models.catalog import MODEL_DEFAULTS

import dfaas_utils
import dfaas_env
import dfaas_train
import dfaas_apl

# Disable Ray's warnings.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logger for this module.
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(Path(__file__).name)


def main():
    parser = argparse.ArgumentParser(prog="dfaas_dirichlet")
    parser.add_argument("experiment_dir", type=Path, help="Existing experiment directory")

    args = parser.parse_args()

    exp_dir = args.experiment_dir.resolve()
    if not exp_dir.exists():
        logger.critical(f"Experiment directory not found: {exp_dir.as_posix()!r}")
        raise FileNotFoundError(exp_dir)

    # Read the existing experiment configuration. Note that is a JSON, not a
    # TOML file!
    exp_config = dfaas_utils.json_to_dict(exp_dir / "exp_config.json")
    logger.info(f"Experiment configuration")
    for key, value in exp_config.items():
        logger.info(f"{key:>25}: {value}")

    # Environment configuration.
    assert exp_config.get("env_config") is not None
    env_config = dfaas_utils.toml_to_dict(exp_config["env_config"])

    # Create a dummy environment, used as reference.
    dummy_env = dfaas_env.DFaaS(config=env_config)
    logger.info(f"Environment configuration")
    for key, value in dummy_env.get_config().items():
        logger.info(f"{key:>25}: {value}")

    ray_info = ray.init(include_dashboard=False)
    logger.info(f"Ray address: {ray_info['address']}")

    # See dfaas_train.py.
    # ----------------------------------
    policies = {}
    policies_to_train = []
    for agent in dummy_env.agents:
        policy_name = f"policy_{agent}"
        policies[policy_name] = PolicySpec(
            policy_class=None,
            observation_space=dummy_env.observation_space[agent],
            action_space=dummy_env.action_space[agent],
            config=None,
        )
        policies_to_train.append(policy_name)

    # Allow to overwrite the default policies with the TOML config file.
    if exp_config.get("policies") is not None:
        policies.clear()
        policies_to_train.clear()

        for policy_cfg in exp_config["policies"]:
            policy_name = policy_cfg["name"]

            # PolicySpec expects the Python class of the policy, not the raw
            # string!
            if policy_cfg.get("class"):
                assert policy_cfg["class"] == "APLPolicy", "Only APLPolicy is supported as custom policy"
                policy_class = dfaas_apl.APLPolicy
            else:
                # It uses PPO (default if None).
                policy_class = None
                policies_to_train.append(policy_name)

            policies[policy_name] = PolicySpec(
                policy_class=policy_class,
                observation_space=dummy_env.observation_space[agent],
                action_space=dummy_env.action_space[agent],
                config=None,
            )

        assert len(policies) == len(dummy_env.agents), "Each policy should be mapped to an agent (and viceversa)"

    # Log policies data.
    for policy_name, policy in policies.items():
        logger.info(f"Policy: name = {policy_name!r}, class = {policy.policy_class!r}")
    logger.info(f"Policies to train: {policies_to_train}")

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        """Called by RLlib at each step to map an agent to a policy (defined above).
        In this case, the map is static: every agent has the same policy, and a
        policy has the same single agent."""
        return f"policy_{agent_id}"

    # Model options.
    model = MODEL_DEFAULTS.copy()
    # Must be False to have a different network for the Critic.
    model["vf_share_layers"] = False

    if exp_config.get("model") is not None:
        # Update the model with the given options.
        given_model = dfaas_utils.json_to_dict(exp_config.get("model"))
        model = model | given_model

    assert dummy_env.max_steps == 288, "Only 288 steps supported for the environment"
    # ----------------------------------

    assert exp_config["algorithm"] == "PPO", "Only PPO is currently supported"

    # Get all the checkpoints.
    checkpoints = sorted(exp_dir.glob("checkpoint_*"))
    logger.info("Available checkpoints:")
    for checkpoint in checkpoints:
        logger.info(f"    {checkpoint.name!r}")

    for checkpoint in checkpoints:
        # We reuse build_ppo() from the dfaas_train module, but we modify the
        # exp_config since we only use the agents to run evaluations.
        experiment = dfaas_train.build_ppo(
            runners=0,  # We just need a local worker to run episodes.
            dummy_env=dummy_env,
            env_config=env_config,
            model=model,
            env_eval_config=env_config,  # Dummy value, not used.
            seed=exp_config["seed"],
            no_gpu=False,
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
            evaluation_num_episodes=1,  # Dummy, do not evaluate!
            training_num_episodes=1,  # Dummy value, we won't train.
        )

        experiment.restore(checkpoint.as_posix())
        logger.info(f"Algorithm restored from {checkpoint.name!r}")

        policy = experiment.get_policy("policy_node_1")
        obs = {
            "input_rate": np.array([100], dtype=np.int32),
            "prev_input_rate": np.array([100], dtype=np.int32),
            "prev_forward_requests": np.array([0], dtype=np.int32),
            "prev_forward_rejects": np.array([0], dtype=np.int32),
        }
        obs_pp = preprocess(obs, policy)

        print()
        print("Obs:", obs)
        print("Obs pp:", obs_pp)
        action, _, extra_info = policy.compute_single_action(obs=obs_pp)
        print("Action", action)
        print("Extra info:", extra_info)
        print()

        # An Algorithm instance cannot be reused after a restore(), so I need
        # to build a new instance for every checkpoint.
        experiment.stop()


def preprocess(raw_obs, policy):
    """Preprocess and returns an observation using the policy's RLlib
    observation preprocessor."""
    # If I want to compute a policy for an observation, I need to preprocess the
    # observation because Ray RLLib automatically wraps the original observation
    # with an array.

    pp = policy.agent_connectors[ObsPreprocessorConnector]
    pp = pp[0]
    _input_dict = {"obs": raw_obs}
    acd = AgentConnectorDataType("0", "0", _input_dict)
    pp.reset(env_id="0")
    ac_o = pp([acd])[0]
    obs_pp = ac_o.data["obs"]
    return obs_pp


if __name__ == "__main__":
    main()
