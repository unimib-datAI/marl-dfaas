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

parser = argparse.ArgumentParser(prog="dfaas_run_ppo")
parser.add_argument(dest="env", help="DFaaS environment to train")
parser.add_argument(dest="suffix", help="A string to append to experiment directory")
parser.add_argument("--rollout-workers",
                    help="Number of rollout worker to create for parallel sampling",
                    default=0, dest="workers", type=int)
parser.add_argument("--use-gpu",
                    help="Use GPU instead of CPU",
                    default=False, dest="gpu", action="store_true")
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

# Experiment configuration.
# TODO: make this configurable!
exp_config = {"seed": 42,  # Seed of the experiment.
              "max_iterations": 200,  # Number of iterations.
              "env": DFaaS.__name__,  # Environment.
              "gpu": args.gpu,
              "workers": args.workers
              }
logger.info(f"Experiment configuration = {exp_config}")

# Env configuration.
env_config = {}
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


# Algorithm config.
ppo_config = (PPOConfig()
              .environment(env=DFaaS.__name__, env_config=env_config)
              .framework("torch")
              .rollouts(num_rollout_workers=args.workers, create_env_on_local_worker=True)
              .evaluation(evaluation_interval=None)
              .debugging(seed=exp_config["seed"])
              .resources(num_gpus=1 if args.gpu else 0)
              .callbacks(dfaas_env.DFaaSCallbacks)
              .multi_agent(policies=policies,
                           policy_mapping_fn=policy_mapping_fn)
              )

# Build the experiment.
ppo_algo = ppo_config.build()

# Get the experiment directory to save other files.
logdir = Path(ppo_algo.logdir).resolve()
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
    ppo_algo.train()

    # Save a checkpoint every 50 iterations.
    if ((iteration + 1) % 50) == 0:
        checkpoint_path = (logdir / f"checkpoint_{iteration:03d}").as_posix()
        ppo_algo.save(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path!r}")

# Save always the latest training iteration.
latest_iteration = exp_config["max_iterations"] - 1
checkpoint_path = logdir / f"checkpoint_{latest_iteration:03d}"
if not checkpoint_path.exists():  # May exist if max_iter is a multiple of 50.
    checkpoint_path = checkpoint_path.as_posix()
    ppo_algo.save(checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path!r}")
logger.info(f"Iterations data saved to: {ppo_algo.logdir}/result.json")

# Do a final evaluation.
logger.info("Final evaluation start")
evaluation = ppo_algo.evaluate()
eval_file = logdir / "final_evaluation.json"
dfaas_utils.dict_to_json(evaluation, eval_file)
logger.info(f"Final evaluation saved to: {ppo_algo.logdir}/final_evaluation.json")

# Remove this file as it is redundant, "result.json" already contains the same
# data.
Path(logdir / "progress.csv").unlink()

# Move the original experiment directory to a custom directory.
exp_name = f"DFAAS-MA_{start}_{DFaaS.type}_{args.suffix}"
result_dir = Path.cwd() / "results" / exp_name
shutil.move(logdir, result_dir.resolve())
logger.info(f"DFAAS experiment results moved to {result_dir.as_posix()!r}")
