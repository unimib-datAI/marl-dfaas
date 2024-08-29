from pathlib import Path
from datetime import datetime
import logging

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import UnifiedLogger

from dfaas_env import DFaaS  # noqa: F401
from dfaas_callbacks import DFaaSCallbacks
import dfaas_utils

# Disable Ray's warnings.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logger for this module.
logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s", level=logging.DEBUG)
logger = logging.getLogger(Path(__file__).name)

tmp_env = DFaaS()

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
                                        observation_space=tmp_env.observation_space["node_0"],
                                        action_space=tmp_env.action_space["node_0"],
                                        config=None),
            "policy_node_1": PolicySpec(policy_class=None,
                                        observation_space=tmp_env.observation_space["node_1"],
                                        action_space=tmp_env.action_space["node_1"],
                                        config=None)
            }


# This function is called by Ray to determine which policy to use for an agent
# returned in the observation dictionary by step() or reset().
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''This function is called at each step to assign the agent to a policy. In
    this case, each agent has a fixed corresponding policy.'''
    return f"policy_{agent_id}"


# Create the results directory to override Ray's default.
logdir = Path.cwd() / "results"
logdir = logdir / Path(f"DFAAS-MA_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
logdir.mkdir(parents=True, exist_ok=True)
logger.info(f"DFAAS experiment directory created at {logdir.as_posix()!r}")

# Experiment configuration.
# TODO: make this configurable!
exp_config = {"seed": 42,  # Seed of the experiment.
              "max_iterations": 100  # Number of iterations.
              }
exp_file = logdir / "exp_config.json"
dfaas_utils.dict_to_json(exp_config, exp_file)

# Env configuration.
env_config = {'seed': exp_config["seed"]}


# This function is called by Ray when creating the logger for the experiment.
def logger_creator(config):
    return UnifiedLogger(config, logdir.as_posix())


# Algorithm config.
ppo_config = (PPOConfig()
              .environment(env="DFaaS", env_config=env_config)
              .framework("torch")
              .rollouts(num_rollout_workers=0)  # Only a local worker.
              .evaluation(evaluation_interval=None)
              .debugging(logger_creator=logger_creator, seed=exp_config["seed"])
              .resources(num_gpus=1)
              .callbacks(DFaaSCallbacks)
              .multi_agent(policies=policies,
                           policy_mapping_fn=policy_mapping_fn)
              )

# Build the experiment.
ppo_algo = ppo_config.build()

# Run the training phase for n iterations.
for iteration in range(exp_config["max_iterations"]):
    logger.info(f"Iteration {iteration}")
    result = ppo_algo.train()
logger.info(f"Iterations data saved to: {ppo_algo.logdir}/result.json")

# Do a final evaluation.
logger.info("Final evaluation")
evaluation = ppo_algo.evaluate()
eval_file = logdir / "final_evaluation.json"
dfaas_utils.dict_to_json(evaluation, eval_file)
logger.info(f"Final evaluation saved to: {ppo_algo.logdir}/final_evaluation.json")

# Remove this file as it is redundant, "result.json" already contains the same
# data.
Path(logdir / "progress.csv").unlink()
