# This Python script runs a training experiment using the SAC algorithm.
#
# WARNING: SAC currently does not work with DFAAS-MA envs because of an unknown
# problem during training, I think it's due to unoptimized hyperparameters that
# cause the gradients to explode.
from pathlib import Path
import shutil
from datetime import datetime
import logging
import argparse

from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.policy import PolicySpec

import dfaas_utils
import dfaas_env

# Disable Ray's warnings.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(prog="dfaas_run_sac")
parser.add_argument(dest="env", help="DFaaS environment to train")
parser.add_argument(dest="suffix", help="A string to append to experiment directory")
parser.add_argument("--no-gpu",
                    help="Disable GPU usage",
                    default=True, dest="gpu", action="store_false")
parser.add_argument("--env-config", help="Environment config file")
parser.add_argument("--workers", type=int, default=5,
                    help="Number of rollout workers")
args = parser.parse_args()

# Initialize logger for this module.
logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s", level=logging.DEBUG)
logger = logging.getLogger(Path(__file__).name)

# Try to load the given environment class from the dfaas_env module.
try:
    DFaaS = getattr(dfaas_env, args.env)
    logger.info(f"Environment {DFaaS.__name__!r} loaded")
except AttributeError:
    logger.error(f"Environment {args.env!r} not found in dfaas_env.py")
    exit(1)

# The number of rollout workers, by default 5.
rollout_workers = args.workers

# Experiment configuration.
# TODO: make this configurable!
exp_config = {"seed": 42,  # Seed of the experiment.
              "max_iterations": 200,  # Number of iterations.
              "env": DFaaS.__name__,  # Environment.
              "gpu": args.gpu,
              "workers": rollout_workers
              }
logger.info(f"Experiment configuration = {exp_config}")

# Env configuration.
if args.env_config is not None:
    env_config = dfaas_utils.json_to_dict(args.env_config)
else:
    env_config = {}
logger.info(f"Environment configuration = {env_config}")

# For the evaluation phase at the end, the env_config is different than the
# training one.
env_eval_config = env_config.copy()
env_eval_config["evaluation"] = True

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


# Algorithm config.
sac_config = (SACConfig()
              .environment(env=DFaaS.__name__, env_config=env_config)
              .framework("torch")
              .training(grad_clip=50)
              .rollouts(batch_mode="complete_episodes", num_rollout_workers=rollout_workers)
              .evaluation(evaluation_interval=None, evaluation_duration=50,
                          evaluation_num_workers=1,
                          evaluation_config={"env_config": env_eval_config})
              .debugging(seed=exp_config["seed"])
              .resources(num_gpus=1 if args.gpu else 0)
              .callbacks(dfaas_env.DFaaSCallbacks)
              .multi_agent(policies=policies,
                           policy_mapping_fn=policy_mapping_fn)
              )

# Build the experiment.
sac_algo = sac_config.build()

# Get the experiment directory to save other files.
logdir = Path(sac_algo.logdir).resolve()
logger.info(f"DFAAS experiment directory created at {logdir.as_posix()!r}")
# This will be used after the evaluation.
start = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

exp_file = logdir / "exp_config.json"
dfaas_utils.dict_to_json(exp_config, exp_file)
logger.info(f"Experiment configuration saved to: {exp_file.as_posix()!r}")

dummy_config = dummy_env.get_config()
env_config_path = logdir / "env_config.json"
dfaas_utils.dict_to_json(dummy_config, env_config_path)
logger.info(f"Environment configuration saved to: {env_config_path.as_posix()!r}")

# Copy the environment source file into the experiment directory. This ensures
# that the original environment used for the experiment is preserved.
dfaas_env_dst = logdir / Path(dfaas_env.__file__).name
shutil.copy2(dfaas_env.__file__, dfaas_env_dst)
logger.info(f"Environment source file saved to: {dfaas_env_dst.as_posix()!r}")

# Run the training phase for n iterations.
logger.info("Training start")
for iteration in range(exp_config["max_iterations"]):
    logger.info(f"Iteration {iteration}")
    sac_algo.train()

    # Save a checkpoint every 50 iterations.
    if ((iteration + 1) % 50) == 0:
        checkpoint_path = (logdir / f"checkpoint_{iteration:03d}").as_posix()
        sac_algo.save(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path!r}")

# Save always the latest training iteration.
latest_iteration = exp_config["max_iterations"] - 1
checkpoint_path = logdir / f"checkpoint_{latest_iteration:03d}"
if not checkpoint_path.exists():  # May exist if max_iter is a multiple of 50.
    checkpoint_path = checkpoint_path.as_posix()
    sac_algo.save(checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path!r}")
logger.info(f"Iterations data saved to: {sac_algo.logdir}/result.json")

# Do a final evaluation.
'''
logger.info("Final evaluation start")
evaluation = sac_algo.evaluate()
eval_file = logdir / "evaluation.json"
dfaas_utils.dict_to_json(evaluation, eval_file)
logger.info(f"Final evaluation saved to: {sac_algo.logdir}/final_evaluation.json")
'''

# Remove this file as it is redundant, "result.json" already contains the same
# data.
Path(logdir / "progress.csv").unlink()

# Move the original experiment directory to a custom directory.
exp_name = f"DFAAS-MA_{start}_{DFaaS.type}_{args.suffix}"
result_dir = Path.cwd() / "results" / exp_name
shutil.move(logdir, result_dir.resolve())
logger.info(f"DFAAS experiment results moved to {result_dir.as_posix()!r}")
