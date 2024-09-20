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
parser.add_argument("suffix",
                    help="A string to append to the evaluation directory")
parser.add_argument("--env-config",
                    help="Use this env config file to update the original env config")
parser.add_argument("--seed",
                    help="Use given seed instead of experiment original one",
                    type=int)
args = parser.parse_args()

# Initialize logger for this module.
logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s", level=logging.DEBUG)
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
policies = {"policy_node_0": PolicySpec(policy_class=None,
                                        observation_space=dummy_env.observation_space["node_0"],
                                        action_space=dummy_env.action_space["node_0"],
                                        config=None),
            "policy_node_1": PolicySpec(policy_class=None,
                                        observation_space=dummy_env.observation_space["node_1"],
                                        action_space=dummy_env.action_space["node_1"],
                                        config=None)
            }


# This function is called by Ray to determine which policy to use for an agent
# returned in the observation dictionary by step() or reset().
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''This function is called at each step to assign the agent to a policy. In
    this case, each agent has a fixed corresponding policy.'''
    return f"policy_{agent_id}"


assert dummy_env.max_steps == 288, "Only 288 steps supported for the environment"

# Algorithm config.
ppo_config = (PPOConfig()
              .environment(env=DFaaS.__name__, env_config=env_config)
              .framework("torch")
              .rollouts(num_rollout_workers=0)  # Only evaluate.
              .evaluation(evaluation_interval=None,
                          evaluation_duration=50,
                          evaluation_num_workers=5)
              .debugging(seed=exp_config["seed"])
              .resources(num_gpus=1)
              .callbacks(dfaas_env.DFaaSCallbacks)
              .multi_agent(policies=policies,
                           policy_mapping_fn=policy_mapping_fn)
              )

# Build the experiment.
ppo_algo = ppo_config.build()

# Load by default the latest checkpoint.
checkpoint_path = sorted(exp_dir.glob("checkpoint_*"))[-1]
exp_config["from_checkpoint"] = checkpoint_path.name
ppo_algo.restore(checkpoint_path.as_posix())
logger.info(f"Algorithm restored from {checkpoint_path.name!r}")

# Get the timestamp and create evaluation sub-directory
start = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
eval_dir = Path(ppo_algo.logdir).resolve() / f"evaluation_{start}_{args.suffix}"
eval_dir.mkdir()

# Save exp and env configs.
eval_config_path = eval_dir / "eval_config.json"
dfaas_utils.dict_to_json(exp_config, eval_config_path)
logger.info(f"Evaluation configuration saved to: {eval_config_path.as_posix()!r}")
env_config_path = eval_dir / "env_config.json"
dfaas_utils.dict_to_json(env_config, env_config_path)
logger.info(f"Environment configuration saved to: {env_config_path.as_posix()!r}")

logger.info("Evaluation start")
evaluation = ppo_algo.evaluate()
eval_file = eval_dir / "evaluation.json"
dfaas_utils.dict_to_json(evaluation, eval_file)
logger.info(f"Evaluation result saved to: {eval_file.as_posix()!r}")

# Move the evaluation directory under the original experiment directory.
result_dir = exp_dir / eval_dir.name
shutil.move(eval_dir, result_dir)
logger.info(f"Evaluation data moved to {result_dir.as_posix()!r}")

# When running an evaluation, Ray automatically creates an experiment
# directory in "~/ray_results". Delete this directory because we saved the
# results of the evaluation in the original experiment directory.
shutil.rmtree(ppo_algo.logdir)
