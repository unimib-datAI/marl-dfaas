from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print

from dfaas_env import DFaaS  # noqa: F401

# Each agent has its own policy (policy_node_X in policy -> node_X in the env).
#
# Note that if no option is given to PolicySpec, it will inherit the
# configuration/algorithm from the main configuration.
policies = {"policy_node_0": PolicySpec(), "policy_node_1": PolicySpec()}


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''This function is called at each step to assign the agent to a policy. In
    this case, each agent has a fixed corresponding policy.'''
    return f"policy_{agent_id}"


# Fixed seed.
seed = 42

env_config = {'seed': 42}

# Experiment config.
ppo_config = (PPOConfig()
              .environment(env="DFaaS", env_config=env_config, disable_env_checking=True)
              .framework("torch")
              .rollouts(num_rollout_workers=0)  # Local worker.
              .evaluation(evaluation_interval=None)
              .debugging(seed=seed)
              .resources(num_gpus=1)
              .multi_agent(policies=policies,
                           policy_mapping_fn=policy_mapping_fn)
              )

# Build the experiment.
ppo_algo = ppo_config.build()

# Run the training phase for 10 iterations.
for iteration in range(1):
    print(f"Iteration nr. {iteration}")
    result = ppo_algo.train()

# Do a final evaluation.
print("Evaluation")
print(pretty_print(ppo_algo.evaluate()))
