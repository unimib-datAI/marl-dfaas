from dfaas_utils import yaml_to_dict, json_to_dict
from dfaas_train import run_experiment

from datetime import datetime
from parse import parse
import numpy as np
import argparse
import json
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Run experiments", 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "--base_env_config",
    help = "Base environment configuration file (YAML format)",
    type = str,
    required = True
  )
  parser.add_argument(
    "--exp_config",
    help = "Experiment configuration file (YAML format)",
    type = str,
    required = True
  )
  parser.add_argument(
    "--n_experiments",
    help = "Number of experiments (per network) to run",
    type = int,
    default = 3
  )
  parser.add_argument(
    "--seed",
    help = "Seed for random number generation",
    type = int,
    default = 4850
  )
  # Parse the arguments
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def main(exp_config: str, base_env_config: str, n_experiments: int, seed: int):
  bname = os.path.basename(base_env_config)
  experiments = {
    "n": [],
    "k": [],
    "network_idx": [],
    "exp_seed": [],
    "exp_suffix": [],
    "start_time": [],
    "finish_time": [],
    "elapsed_time": []
  }
  rng = np.random.default_rng(seed = seed)
  # load experiment configuration
  exp_config = yaml_to_dict(exp_config)
  # -- define algorithm suffix
  algo_suffix = ""
  if exp_config["algorithm"]["name"] == "MAPPO":
    mode = "concat"
    if exp_config.get("model") is not None:
      model = json_to_dict(exp_config.get("model"))
      mode = model.get("custom_model_config", {}).get("mode", "concat")
    algo_suffix = f"{mode}_"
  # loop over scenarios
  for dirname in os.listdir(base_env_config):
    n,k = parse("n_{}-k_{}", dirname)
    for filename in os.listdir(os.path.join(base_env_config, dirname)):
      network_idx = parse("_{}.yaml", filename.replace(bname, ""))[0]
      # loop over experiments
      for exp_seed in rng.integers(
          low = 0, high = 4850 * 4850, size = n_experiments
        ):
        # read environment configuration
        env_config = yaml_to_dict(
          os.path.join(base_env_config, dirname, filename)
        )
        # -- save info
        experiments["n"].append(int(n))
        experiments["k"].append(int(k))
        experiments["network_idx"].append(int(network_idx))
        experiments["exp_seed"].append(int(exp_seed))
        experiments["exp_suffix"].append(
          f"{algo_suffix}n{n}_k{k}_i{network_idx}_s{exp_seed}"
        )
        experiments["start_time"].append(
          datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S.%f")
        )
        with open("experiments.json", "w") as ostream:
          ostream.write(json.dumps(experiments, indent = 2))
        # -- start experiment
        run_experiment(
          suffix = experiments["exp_suffix"][-1],
          exp_config = exp_config,
          env_config = env_config,
          runners = None,
          seed = int(exp_seed),
          dry_run = True,
        )
        # -- record end
        e = datetime.now()
        experiments["finish_time"].append(
          datetime.strftime(e, "%Y-%m-%d_%H-%M-%S.%f")
        )
        experiments["elapsed_time"].append(
          (
            e - datetime.strptime(
              experiments["start_time"][-1], "%Y-%m-%d_%H-%M-%S.%f"
            )
          ).total_seconds()
        )
        with open("experiments.json", "w") as ostream:
          ostream.write(json.dumps(experiments, indent = 2))


if __name__ == "__main__":
  args = parse_arguments()
  base_env_config = args.base_env_config
  exp_config = args.exp_config
  n_experiments = args.n_experiments
  seed = args.seed
  # check if the experiments file exists
  if os.path.exists("experiments.json"):
    answer = input("Experiments file exists; do you really want to continue?")
    if answer.lower() in ["y","yes"]:
      main(exp_config, base_env_config, n_experiments, seed)
    elif answer.lower() in ["n","no"]:
      print("Aborted")
    else:
      raise ValueError("Provide y/n answer!")
  else:
    main(exp_config, base_env_config, n_experiments, seed)
