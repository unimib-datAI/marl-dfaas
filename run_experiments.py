from RL4CC.experiments.federated_train import FederatedTrainingExperiment

from dfaas_utils import yaml_to_dict, json_to_dict

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
    help = "Folder containing environment configuration files (YAML format)",
    type = str,
    required = True
  )
  parser.add_argument(
    "--exp_config",
    help = "Experiment configuration file (JSON format)",
    type = str,
    required = True
  )
  parser.add_argument(
    "--n_agents_list",
    help = "Number of agents to consider (must be among those in the data)",
    type = int,
    nargs = "+",
    required = True
  )
  parser.add_argument(
    "--instance_idxs",
    help = "Index of instance(s) to run",
    type = int,
    nargs = "+",
    default = [0]
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
  parser.add_argument(
    "--restart_from",
    help = "Experiments JSON file with partially-completed experiments",
    type = str,
    default = None
  )
  # Parse the arguments
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def find_completed_experiment(
    n: int, k: int, network_idx: int, exp_seed: int, experiments: dict
  ) -> int:
  num_experiments = len(experiments.get("n", []))
  # loop over experiments
  for i in range(num_experiments):
    # -- check if the description is the same
    if (
        experiments["n"][i] == n and
        experiments["k"][i] == k and
        experiments["network_idx"][i] == network_idx and
        experiments["exp_seed"][i] == exp_seed
      ):
      # -- check if the experiments was completed
      ftlist = experiments.get("finish_time", [])
      if len(ftlist) > i and ftlist[i] is not None:
        return i
  return -1


def main(
    exp_config: str, 
    base_env_config: str, 
    n_agents_list: list,
    instance_idxs: list,
    n_experiments: int, 
    seed: int,
    restart_from: str = None
  ):
  bname = os.path.basename(base_env_config)
  experiments = {
    "n": [],
    "k": [],
    "network_idx": [],
    "exp_seed": [],
    "exp_suffix": [],
    "exp_logdir": [],
    "start_time": [],
    "finish_time": [],
    "elapsed_time": []
  }
  # load previous experiments (if provided)
  exp_list_file = "experiments.json"
  if restart_from is not None:
    exp_list_file = restart_from
    with open(restart_from, "r") as istream:
      experiments = json.load(istream)
  rng = np.random.default_rng(seed = seed)
  # load experiment configuration
  exp_config = json_to_dict(exp_config)
  # -- define algorithm suffix
  algo_suffix = ""
  if exp_config["algorithm"] == "MAPPO":
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
        if int(n) in n_agents_list and int(network_idx) in instance_idxs:
          # check if the experiment was already run
          exp_idx = find_completed_experiment(
            int(n), int(k), int(network_idx), int(exp_seed), experiments
          )
          if exp_idx < 0:
            # read environment configuration
            env_config = yaml_to_dict(
              os.path.join(base_env_config, dirname, filename)
            )
            exp_config["env_config"] = env_config
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
            with open(exp_list_file, "w") as ostream:
              ostream.write(json.dumps(experiments, indent = 2))
            # -- start experiment
            exp = FederatedTrainingExperiment(exp_config = exp_config)
            exp.run()
            # -- record end
            e = datetime.now()
            experiments["exp_logdir"].append(exp.logdir)
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
            with open(exp_list_file, "w") as ostream:
              ostream.write(json.dumps(experiments, indent = 2))


if __name__ == "__main__":
  args = parse_arguments()
  base_env_config = args.base_env_config
  exp_config = args.exp_config
  n_agents_list = args.n_agents_list
  instance_idxs = args.instance_idxs
  n_experiments = args.n_experiments
  seed = args.seed
  restart_from = args.restart_from
  # check if partially-run experiments are provided
  if restart_from is None:
    # check if the experiments file exists
    if os.path.exists("experiments.json"):
      answer = input(
        "Experiments file exists; do you really want to continue? "
      )
      if answer.lower() in ["y","yes"]:
        main(
          exp_config, 
          base_env_config, 
          n_agents_list, 
          instance_idxs, 
          n_experiments, 
          seed
        )
      elif answer.lower() in ["n","no"]:
        print("Aborted")
      else:
        raise ValueError("Provide y/n answer!")
    else:
      main(
        exp_config, 
        base_env_config, 
        n_agents_list, 
        instance_idxs, 
        n_experiments, 
        seed
      )
  else:
    main(
      exp_config, 
      base_env_config, 
      n_agents_list, 
      instance_idxs, 
      n_experiments, 
      seed, 
      restart_from
    )
