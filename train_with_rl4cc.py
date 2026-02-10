from dfaas_env_metrics import DFaaSMetricsEnvironment
from ray.tune.registry import register_env

def _register_env_dfaasmetrics(env_config):
  """Automatically called by Ray RLLib when creating the DFaaS environment.
  Returns a new DFaaS environment, created with the given config."""
  return DFaaSMetricsEnvironment(env_config)

# Register the environments with Ray so that they can be used automatically 
# when creating experiments.
register_env("DFaaSMetricsEnvironment", _register_env_dfaasmetrics)

from RL4CC.experiments.train import TrainingExperiment
from RL4CC.utilities.common import load_config_file

import yaml
import sys


def main(exp_config_file: str):
  # load and adjust configuration
  exp_config = load_config_file(exp_config_file)
  with open(exp_config.pop("env_config_file"), "r") as istream:
    env_config = yaml.safe_load(istream)
  env_config["joined_metrics"] = "dataset/joined_metrics.csv"
  env_config["joined_metrics_avg"] = "dataset/joined_metrics_avg.csv"
  exp_config["env_config"] = env_config
  # define experiment
  exp = TrainingExperiment(exp_config = exp_config)
  # run
  algo = exp.run()


if __name__ == "__main__":
  exp_config_file = sys.argv[1]
  main(exp_config_file)
