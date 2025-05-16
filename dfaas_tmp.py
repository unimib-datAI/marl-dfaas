from pathlib import Path
import shutil
from datetime import datetime
import logging
import argparse

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

import dfaas_utils
import dfaas_env

# Disable Ray's warnings.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(prog="dfaas_evaluate_ppo")
parser.add_argument("experiment_dir", help="Existing experiment directory")
parser.add_argument("--env-config", help="Use this env config file to update the original env config")
parser.add_argument("--seed", help="Use given seed instead of experiment original one", type=int)
parser.add_argument(
    "--runners",
    help="Number of evaluation runners (non-negative integer)",
    type=int,
    default=5,
)
args = parser.parse_args()

# Initialize logger for this module.
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(Path(__file__).name)

exp_dir = Path(args.experiment_dir).resolve()
if not exp_dir.exists():
    logger.critical(f"Experiment directory not found: {exp_dir.as_posix()!r}")
    raise FileNotFoundError(exp_dir)

# Experiment configuration (read the existing one).
exp_config = dfaas_utils.json_to_dict(exp_dir / "exp_config.json")
if args.seed:
    # Allow to update the original seed with a provided one.
    exp_config["seed"] = args.seed
logger.info(f"Experiment configuration = {exp_config}")

# Try to load the environment class from the dfaas_env module.
try:
    DFaaS = getattr(dfaas_env, exp_config["env"])
    logger.info(f"Environment {DFaaS.__name__!r} loaded")
except AttributeError:
    logger.critical(f"Environment {exp_config['env']!r} not found in dfaas_env.py")
    exit(1)

# Environment configuration (read the existing one).
env_config = dfaas_utils.json_to_dict(exp_dir / "env_config.json")
if args.env_config is not None:
    # Allow to update the original env config with a provided one.
    new_env_config = dfaas_utils.json_to_dict(args.env_config)
    env_config.update(new_env_config)
logger.info(f"Environment configuration = {env_config}")

# Create a dummy environment, used to get observation and action spaces.
dummy_env = DFaaS(config=env_config)

# PolicySpec is required to specify the action/observation space for each
# policy. Because each agent in the env has different action and observation
# space, it is important to configure them.
#
# See this thread: https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/5
#
# Each agent has its own policy (policy_node_X in policy -> node_X in the env).
#
# Note that if no option is given to PolicySpec, it will inherit the
# configuration/algorithm from the main configuration.
policies = {
    "policy_node_0": PolicySpec(
        policy_class=None,
        observation_space=dummy_env.observation_space["node_0"],
        action_space=dummy_env.action_space["node_0"],
        config=None,
    ),
    "policy_node_1": PolicySpec(
        policy_class=None,
        observation_space=dummy_env.observation_space["node_1"],
        action_space=dummy_env.action_space["node_1"],
        config=None,
    ),
}


# This function is called by Ray to determine which policy to use for an agent
# returned in the observation dictionary by step() or reset().
def policy_mapping_fn(agent_id, episode, runner, **kwargs):
    """This function is called at each step to assign the agent to a policy. In
    this case, each agent has a fixed corresponding policy."""
    return f"policy_{agent_id}"


assert dummy_env.max_steps == 288, "Only 288 steps supported for the environment"

# Algorithm config.
ppo_config = (
    PPOConfig()
    # By default RLlib uses the new API stack, but I use the old one.
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .environment(env=DFaaS.__name__, env_config=env_config)
    .framework("torch")
    .env_runners(num_env_runners=0)  # Only evaluate.
    .evaluation(
        evaluation_interval=None,
        evaluation_duration=50,
        evaluation_num_env_runners=args.runners,
        evaluation_config={"env_config": env_config},
    )
    .debugging(seed=exp_config["seed"])
    .resources(num_gpus=1)
    .callbacks(dfaas_env.DFaaSCallbacks)
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
)

# Build the experiment.
ppo_algo = ppo_config.build()

# Load by default the latest checkpoint.
checkpoint_path = sorted(exp_dir.glob("checkpoint_*"))[-1]
exp_config["from_checkpoint"] = checkpoint_path.name
ppo_algo.restore(checkpoint_path.as_posix())
logger.info(f"Algorithm restored from {checkpoint_path.name!r}")

obs = dummy_env.observation_space.sample()
policy = ppo_algo.get_policy("policy_node_1")
obs = obs["node_1"]


# Pre-process obs
def preprocess(obs, policy):
    from ray.rllib.connectors.agent.obs_preproc import ObsPreprocessorConnector
    from ray.rllib.utils.typing import AgentConnectorDataType

    pp = policy.agent_connectors[ObsPreprocessorConnector]
    pp = pp[0]
    _input_dict = {"obs": obs}
    acd = AgentConnectorDataType("0", "0", _input_dict)
    pp.reset(env_id="0")
    ac_o = pp([acd])[0]
    obs_pp = ac_o.data["obs"]
    return obs_pp


import numpy as np

obs = {
    "input_requests": np.array([100], dtype=np.int32),
    "prev_forward_rejects": np.array([0.0], dtype=np.float32),
    "prev_forward_requests": np.array([30.9], dtype=np.float32),
    "prev_local_rejects": np.array([0.0], dtype=np.float32),
    "prev_local_requests": np.array([31.0], dtype=np.float32),
}

obs_pp = preprocess(obs, policy)

action, _, extra_info = policy.compute_single_action(obs=obs_pp)
print(action)
print(extra_info)
