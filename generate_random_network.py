from dfaas_utils import yaml_to_dict, generate_random_float, generate_random_int

from networkx import random_regular_graph
from copy import deepcopy
import numpy as np
import argparse
import yaml
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Results postprocessing", 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "--base_env_config",
    help = "Base environment configuration file (YAML format)",
    type = str,
    required = True
  )
  parser.add_argument(
    "--n_agents",
    help = "Number of agents",
    type = int,
    required = True
  )
  parser.add_argument(
    "--degree",
    help = "Node degree",
    type = int,
    required = True
  )
  parser.add_argument(
    "--seed",
    help = "Seed for random number generation",
    type = int,
    default = 4850
  )
  parser.add_argument(
    "--n_networks",
    help = "Number of networks to generate",
    type = int,
    default = 3
  )
  # Parse the arguments
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def generate_networks(
    base_env_config: str, n_agents: int, k: int, n_networks: int, seed: int
  ) -> str:
  # create folder to store the generated networks
  bfolder, bname = os.path.split(base_env_config)
  bname = bname.replace(".yaml", "")
  networks_folder = os.path.join(bfolder, bname, f"n_{n_agents}-k_{k}")
  os.makedirs(networks_folder, exist_ok = True)
  # load base configuration
  base_env_config = yaml_to_dict(base_env_config)
  limits = base_env_config.pop("limits")
  # random number generator
  rng = np.random.default_rng(seed = seed)
  for network_idx in range(n_networks):
    print(f"Generating network {network_idx}")
    # generate network
    network_seed = rng.integers(low = 0, high = 4850 * 4850 * 4850)
    graph = random_regular_graph(
      d = k, n = n_agents, seed = int(network_seed)
    )
    # update the environment definition
    updated_env = deepcopy(base_env_config)
    updated_env["network"] = []
    updated_env["network_links"] = {}
    updated_env["node_ram_gb"] = {}
    updated_env["perfmodel_params"] = {}
    for u in graph.nodes:
      # -- ram capacity
      updated_env["node_ram_gb"][f"node_{u}"] = generate_random_int(
        rng, limits["node_ram_gb"]
      )
      # -- performance parameters
      updated_env["perfmodel_params"][f"node_{u}"] = {
        "cold_service_time": None,
        "idle_time_before_kill": generate_random_int(
          rng, limits["idle_time_before_kill"]
        ),
        "maximum_concurrency": None,
        "warm_service_time": None,
      }
      # -- links
      updated_env["network_links"][f"node_{u}"] = {}
      for v in graph.neighbors(u):
        if f"node_{v} node_{u}" not in updated_env["network"]:
          # -- list of (undirected) edges
          updated_env["network"].append(f"node_{u} node_{v}")
          # -- links (with attributes)
          updated_env["network_links"][f"node_{u}"][f"node_{v}"] = {
            "access_delay_ms": generate_random_float(
              rng, limits["access_delay_ms"]
            ),
            "bandwidth_mbps": None,
            "bandwidth_mbps_mean": generate_random_float(
              rng, limits["bandwidth_mbps_mean"]
            ),
            "bandwidth_mbps_method": "generated",
            "bandwidth_mbps_random_noise": generate_random_float(
              rng, limits["bandwidth_mbps_random_noise"]
            )
          }
    # -- save
    with open(
        os.path.join(networks_folder, f"{bname}_{network_idx}.yaml"), "w"
      ) as ostream:
      ostream.write(yaml.dump(updated_env, sort_keys = True, indent = 4))


if __name__ == "__main__":
  # parse arguments
  args = parse_arguments()
  base_env_config = args.base_env_config
  n_agents = args.n_agents
  k = args.degree
  n_networks = args.n_networks
  seed = args.seed
  # run
  generate_networks(base_env_config, n_agents, k, n_networks, seed)
  
