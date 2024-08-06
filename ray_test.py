# This is a test file to run Ray in a multi-agent environment with two different
# algorithms. If it runs, Ray is installed correctly.
from ray.rllib.examples.env.multi_agent import GuessTheNumberGame
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

# GuessTheNumberGame is not registered, I have to do it manually.
register_env("GuessTheNumberGame", lambda env_config: GuessTheNumberGame(config=env_config))

# The first agent can choose a number between 0 and 100. The second agent should
# find this number in less than 20 steps (13/14 steps using binary search).
env_config = {"max_number": 100, "max_steps": 20}

# Fixed seed.
seed = 42

# This experiment has two policies, one for each player. The first player
# (player_0) must choose the number to be guessed by the second player
# (player_1).
#
# Note that if no option is given to PolicySpec, it will inherit the
# configuration/algorithm from the main configuration.
policies = {"policy_0": PolicySpec(policy_class=RandomPolicy),
            "policy_1": PolicySpec()
            }


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''This function is called at each step to assign the agent to a policy. In
    this case, each agent has a fixed corresponding policy.'''
    return f"policy_{agent_id}"


# Experiment config.
ppo_config = (PPOConfig()
              .environment(env="GuessTheNumberGame", env_config=env_config, disable_env_checking=True)
              .framework("torch")
              .rollouts(num_rollout_workers=0)
              .evaluation(evaluation_interval=None)
              .debugging(seed=seed)
              .resources(num_gpus=1)
              .multi_agent(policies=policies,
                           policy_mapping_fn=policy_mapping_fn,
                           policies_to_train=["policy_1"])
              )

# Build the experiment.
ppo_algo = ppo_config.build()

# Run the training phase for 10 iterations.
for iteration in range(10):
    print(f"Iteration nr. {iteration}")
    result = ppo_algo.train()

# Do a final evaluation.
print("Evaluation")
print(pretty_print(ppo_algo.evaluate()))