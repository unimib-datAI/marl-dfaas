# This is a just a test file to check if Ray is installed correctly.
#
# Taken from: https://docs.ray.io/en/releases-2.8.1/rllib/index.html#rllib-in-60-seconds
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("Taxi-v3")
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    .resources(num_gpus=1)
    .evaluation(evaluation_num_workers=1)
)

algo = config.build()  # 2. build the algorithm,

for _ in range(5):
    result = algo.train()  # 3. train it,
    print(pretty_print(result))

print(pretty_print(algo.evaluate()))  # 4. and evaluate it.
