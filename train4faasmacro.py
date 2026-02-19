from dfaasmacro_env import DFaaSMacroEnvironment
from ray.tune.registry import register_env

def _register_env_dfaasmacro(env_config):
  """Automatically called by Ray RLLib when creating the DFaaS environment.
  Returns a new DFaaS environment, created with the given config."""
  return DFaaSMacroEnvironment(env_config)

# Register the environments with Ray so that they can be used automatically 
# when creating experiments.
register_env("DFaaSMacroEnvironment", _register_env_dfaasmacro)

from RL4CC.experiments.train import TrainingExperiment
from RL4CC.utilities.common import load_config_file

from typing import Tuple
import json
import yaml
import ast
import sys
import os


def load_base_instance(folder: str) -> Tuple[dict, dict]:
  # base instance
  base_instance = {}
  with open(
    os.path.join(folder, "base_instance_data.json"), "r"
  ) as istream:
    data = json.load(istream)
    base_instance = restore_types(data)
  # load limits
  load_limits = {}
  with open(
    os.path.join(folder, "load_limits.json"), "r"
  ) as istream:
    data = json.load(istream)
    load_limits = restore_types(data)
  return base_instance, load_limits


def restore_types(serialized_dict: dict):
  """Restore the original types"""
  _dict = {}
  for key, value in serialized_dict.items():
    new_key = key
    try:
      new_key = ast.literal_eval(key)
    except ValueError:
      pass
    if isinstance(value, dict):
      _dict[new_key] = restore_types(value)
    else:
      _dict[new_key] = value
  return _dict


def main(exp_config_file: str):
  # load and adjust configuration
  exp_config = load_config_file(exp_config_file)
  with open(exp_config.pop("env_config_file"), "r") as istream:
    env_config = yaml.safe_load(istream)
  exp_config["env_config"] = env_config
  # -- instance data
  base_instance, load_limits = load_base_instance("/Users/federicafilippini/Documents/GitHub/DFaaSOptimizer/solutions/newmadea/2026-02-03_17-04-56.370884")
  exp_config["env_config"]["instance"] = base_instance
  # define experiment
  exp = TrainingExperiment(exp_config = exp_config)
  # run
  algo = exp.run()


if __name__ == "__main__":
  exp_config_file = "configs/rl4cc/faasmacro.json"#sys.argv[1]
  main(exp_config_file)
