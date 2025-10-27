from copy import deepcopy
from tqdm import trange
from pathlib import Path
import sys
import os
import logging

# By default Ray uses DEBUG level, but I prefer the ERROR level and this must be
# set manually!
logging.basicConfig(level=logging.ERROR)

# We are inside the "model" directory, but we need to imports modules oustide
# this directory (like perfmodel or dfaas_env), so we add the parent directory
# to PATH.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perfmodel import get_sls_warm_count_dist
from dfaas_env import DFaaS as DFaaSEnv
from dfaas_utils import toml_to_dict

from model_utilities import (
    decode_solution,
    extract_solution,
    init_complete_solution,
    join_complete_solution,
    plot_history,
    save_solution,
)
from model import SimpleCentralizedLMM


def compute_rejections(max_load_dict: dict, instance_data: dict) -> dict:
    rejection_rate = {}
    # for n, max_load in max_load_dict.items():
    #   print(f"Compute rejections for node {n}")
    #   for ll in tqdm(range(max_load + 1)):
    #     rej = float(get_sls_warm_count_dist(ll, 15, 30, 60)[0]["rejection_rate"])
    #     rejection_rate[(n+1, 1, ll)] = rej
    warm_service_time = instance_data[None]["warm_service_time"][None]
    cold_service_time = instance_data[None]["cold_service_time"][None]
    idle_time_before_kill = instance_data[None]["idle_time_before_kill"][None]
    maximum_concurrency = instance_data[None]["maximum_concurrency"][None]
    for ll in range(1, max(max_load_dict.values()) + 1):
        rejection_rate[ll] = float(
            max(
                0,
                get_sls_warm_count_dist(
                    ll, warm_service_time, cold_service_time, idle_time_before_kill, maximum_concurrency
                )[0]["rejection_rate"],
            )
        )
    return rejection_rate


def convert_to_pb_instance(env: DFaaSEnv) -> dict:
    Nn = len(env.agents)
    instance_data = {
        None: {"Nn": {None: Nn}, "Nf": {None: 1}, "neighborhood": define_neighborhood(Nn, env.agent_neighbors)}
    }
    return instance_data


def define_neighborhood(Nn: int, neighbors_dict: dict) -> dict:
    converted_neighbors_dict = {}
    for n1 in range(Nn):
        for n2 in range(Nn):
            if f"node_{n1}" in neighbors_dict and f"node_{n2}" in neighbors_dict[f"node_{n1}"]:
                converted_neighbors_dict[(n1 + 1, n2 + 1)] = 1
            else:
                converted_neighbors_dict[(n1 + 1, n2 + 1)] = 0
    return converted_neighbors_dict


def initialize_environment(env_config_file: str, base_seed: int | list[int], n_experiments: int):
    """
    Initializes and returns a DFaaS environment using the given configuration
    file, random seed, and number of experiments.

    The returned DFaaS environment is ready for experimental runs, it is
    necessary to call reset() without any seed to start the first episode.
    """
    # Load env. configuration.
    env_config = toml_to_dict(env_config_file)

    if isinstance(base_seed, int):
        # Manually overwrite the evaluation mode to have reproducible seeds.
        env_config["evaluation"] = True

    print(f"Environment config.: {env_config}\n")

    # Initialize environment.
    env = DFaaSEnv(config=env_config)

    # Pre-calculate seeds only if in evaluation mode, otherwise each reset()
    # call requires a seed.
    if isinstance(base_seed, int):
        env.set_master_seed(base_seed, n_experiments)
    else:
        # Dummy seed, we need to call reset at least once to set up the env.
        env.reset(seed=seed)

    return env


def update_instance(base_instance_data: dict, input_rate: dict, t: int) -> dict:
    instance_data = deepcopy(base_instance_data)
    Nn = base_instance_data[None]["Nn"][None]
    neighborhood = base_instance_data[None]["neighborhood"]
    # extract the incoming load in the current time step
    instance_data[None]["incoming_load"] = {}
    max_load_dict = {}
    for n1 in range(Nn):
        agent = f"node_{n1}"
        instance_data[None]["incoming_load"][(n1 + 1, 1)] = int(input_rate[agent][t])
        # compute the maximum per-agent load (worst-case: every neighbor sends it
        # all its requests and the agents decides to process locally all its own)
        max_load = int(input_rate[agent][t])
        for n2 in range(Nn):
            if neighborhood[(n2 + 1, n1 + 1)]:
                max_load += int(input_rate[f"node_{n2}"][t])
        max_load_dict[n1] = max_load
    # compute reject rate
    instance_data[None]["Ml"] = {None: max(max_load_dict.values())}
    instance_data[None]["rejection_rate"] = compute_rejections(max_load_dict, instance_data)
    return instance_data


def run_episode(
    env: DFaaSEnv,
    exp_iter: int,
    warm_service_time: float,
    cold_service_time: float,
    idle_time_before_kill: int,
    maximum_concurrency: int,
    base_results_folder: str,
    seed: int | None,
):
    # Reset environment to get new episode with input workload.
    if seed is None:
        # The env. will use an internal generated seed.
        _ = env.reset()
    else:
        # By default, seed=seed sets the master seed, which is then used to
        # generate a sequence of seeds, one for each episode. However, in this
        # case, we want to directly control the seed for a single episode, so we
        # override it individually for each one.
        _ = env.reset(options={"override_seed": seed})

    # use environment to build the base problem instance
    base_instance_data = convert_to_pb_instance(env)
    base_instance_data[None]["warm_service_time"] = {None: warm_service_time}
    base_instance_data[None]["cold_service_time"] = {None: cold_service_time}
    base_instance_data[None]["idle_time_before_kill"] = {None: idle_time_before_kill}
    base_instance_data[None]["maximum_concurrency"] = {None: maximum_concurrency}
    # loop over steps
    model = SimpleCentralizedLMM()
    complete_solution = init_complete_solution()
    obj_values = []
    for t in trange(env.max_steps, desc=f"Episode {exp_iter} seed {env.seed}"):
        # add the current input workload to the problem instance
        instance_data = update_instance(base_instance_data, env.input_rate, t)
        # generate problem instance and solve it
        instance = model.generate_instance(instance_data)
        sol = model.solve(instance, {}, solver_name="glpk", initial_solution=None)
        x, y, z, obj = extract_solution(instance_data, sol)
        # print(f"Number of rejections at time {t}: {obj}")
        obj_values.append(obj)
        complete_solution = decode_solution(x, y, z, complete_solution)
    # merge
    solution, offloaded, detailed_fwd_solution = join_complete_solution(complete_solution)
    # prepare folder to save results
    results_folder = os.path.join(base_results_folder, f"{exp_iter}_seed_{env.seed}")
    os.makedirs(results_folder, exist_ok=True)
    # plot
    plot_history(env.input_rate, solution, offloaded, obj_values, os.path.join(results_folder, "history.png"))
    # compute rejection percentage
    total_reject_rate = [
        obj_values[t] / sum([env.input_rate[a][t] for a in env.input_rate]) * 100 for t in range(env.max_steps)
    ]
    # save
    save_solution(solution, offloaded, detailed_fwd_solution, obj_values, total_reject_rate, model.name, results_folder)


def run(
    env_config_file: str,
    n_experiments: int,
    base_seed: int | list[int],
    warm_service_time: float,
    cold_service_time: float,
    idle_time_before_kill: int,
    maximum_concurrency: int,
    base_results_folder: str,
):
    if isinstance(base_seed, int):
        base_dir = Path(base_results_folder).absolute() / f"seed_{base_seed}"

        try:
            base_dir.mkdir(parents=True)
        except FileExistsError:
            print(f"Directory {base_dir.as_posix()!r} already exist!")
            return
    else:
        assert len(base_seed) == n_experiments

    env = initialize_environment(env_config_file, base_seed, n_experiments)

    for exp in trange(n_experiments, desc="Running episodes"):
        if isinstance(base_seed, int):
            # Seed automatically generated by the env.
            seed = None
        else:
            seed = base_seed[exp]

        run_episode(
            env,
            exp,
            warm_service_time,
            cold_service_time,
            idle_time_before_kill,
            maximum_concurrency,
            base_results_folder,
            seed,
        )


if __name__ == "__main__":
    env_config_file = "configs/env/five_agents.toml"

    n_experiments = 10
    # I need to specify manually all seeds because I do not know the base seed.
    base_seed = [
        1114399290,
        586248983,
        1296339178,
        980462265,
        2807237418,
        3153669498,
        1573623524,
        1657272726,
        409898216,
        2730495449,
    ]

    warm_service_time = 15
    cold_service_time = 30
    idle_time_before_kill = 600
    maximum_concurrency = 1000
    base_results_folder = "model_results"
    run(
        env_config_file,
        n_experiments,
        base_seed,
        warm_service_time,
        cold_service_time,
        idle_time_before_kill,
        maximum_concurrency,
        base_results_folder,
    )
