from pathlib import Path
from datetime import datetime
import json

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import UnifiedLogger

from dfaas_env import DFaaS  # noqa: F401

tmp_env = DFaaS()

# Each agent in the DFaaS network has the same action and observation space, but
# this can be changed in the future. A PolicySpec is required to specify this
# information for each policy used in this experiment.
#
# See this thread: https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/5
policy_spec = PolicySpec(policy_class=None,
                         observation_space=tmp_env.observation_space["node_0"],
                         action_space=tmp_env.action_space["node_0"],
                         config=None)

# Each agent has its own policy (policy_node_X in policy -> node_X in the env).
#
# Note that if no option is given to PolicySpec, it will inherit the
# configuration/algorithm from the main configuration.
policies = {"policy_node_0": policy_spec, "policy_node_1": policy_spec}


# This function is called by Ray to determine which policy to use for an agent
# returned in the observation dictionary by step() or reset().
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''This function is called at each step to assign the agent to a policy. In
    this case, each agent has a fixed corresponding policy.'''
    return f"policy_{agent_id}"


# Fixed seed.
seed = 42

env_config = {'seed': 42}

# Create the results directory to override Ray's default.
logdir = Path.cwd() / "results"
logdir = logdir / Path(f"DFAAS-MA_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
logdir.mkdir(parents=True, exist_ok=True)
print(f"DFAAS experiment directory created at {logdir.as_posix()!r}")


# This function is called by Ray when creating the logger for the experiment.
def logger_creator(config):
    return UnifiedLogger(config, logdir.as_posix())


# Experiment config.
ppo_config = (PPOConfig()
              .environment(env="DFaaS", env_config=env_config)
              .framework("torch")
              .rollouts(num_rollout_workers=0)  # Only a local worker.
              .evaluation(evaluation_interval=None)
              .debugging(logger_creator=logger_creator, seed=seed)
              .resources(num_gpus=1)
              .multi_agent(policies=policies,
                           policy_mapping_fn=policy_mapping_fn)
              )

# Build the experiment.
ppo_algo = ppo_config.build()

# Run the training phase for n iterations.
for iteration in range(1000):
    print(f"Iteration {iteration}")
    result = ppo_algo.train()

# Do a final evaluation.
print("Final evaluation")
evaluation = ppo_algo.evaluate()
eval_file = logdir / "final_evaluation.json"
eval_file.write_text(json.dumps(evaluation), encoding="utf8")
