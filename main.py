from pathlib import Path
import sys
import argparse
from datetime import datetime
from copy import deepcopy

import utils

import numpy as np

from RL4CC.experiments.train import TrainingExperiment
from RL4CC.algorithms.algorithm import Algorithm
from RL4CC.utilities.postprocessing import evaluate_policy
from RL4CC.utilities.logger import Logger
from RL4CC.utilities.common import not_defined

from traffic_env import TrafficManagementEnv

# Logger used in this file.
logger = Logger(name="DFAAS", verbose=2)


def main(dfaas_config_path, experiments_prefix):
    dfaas_config, experiments = prepare_experiments(dfaas_config_path)

    # The absolute path to the DFAAS output directory.
    dfaas_dir = Path(dfaas_config["results_directory"]).absolute()

    # Where to save the experiments dictionary.
    experiments_path = Path(dfaas_dir, "experiments.json")

    # Save a first version of the experiments to disk.
    utils.dict_to_json(experiments, experiments_path)

    # Run all experiments.
    for (algo, algo_values) in experiments.items():
        for (params, params_values) in algo_values.items():
            for (scenario, scenario_value) in params_values.items():
                for exp in scenario_value.values():
                    if (experiments_prefix is not None
                       and len(experiments_prefix) >= 1
                       and not exp["id"].startswith(tuple(experiments_prefix))):
                        logger.log(f"Skipping experiment ID {exp['id']}")
                        continue
                    logger.log(f"Running experiment ID {exp['id']}")

                    # Based on "base_exp_config", make a copy and adjust some
                    # properties for this experiment.
                    exp_config = deepcopy(dfaas_config["base_exp_config"])
                    exp_config["algorithm"] = algo
                    exp_config["logdir"] = dfaas_config['results_directory']
                    exp_config["id"] = exp["id"]
                    exp_config["env_config"]["seed"] = exp["seed"]
                    exp_config["env_config"]["scenario"] = scenario
                    # The "debugging" key may be not present.
                    exp_config["ray_config"].setdefault("debugging", {})
                    exp_config["ray_config"]["debugging"]["seed"] = exp["seed"]

                    # Get the algorithm's parameters ("standard" or "tuned") and
                    # then update the configuration.
                    parameters = dfaas_config["parameters"][algo][params]
                    exp_config["ray_config"]["training"].update(parameters)

                    # Run the experiment.
                    train_exp, policy = train(exp_config)

                    # Evaluate the experiment.
                    evaluate(dfaas_config["eval_config"], train_exp, policy)

                    # exp.logdir is the absolute path to the experiment results
                    # directory.  We want a path relative to dfaas_dir because
                    # we already know the absolute directory (is dfaas_dir).
                    # This is necessary to avois any absolute paths in the
                    # output files.
                    out_dir = Path(train_exp.logdir).relative_to(dfaas_dir).as_posix()

                    # Update experiment data.
                    exp["done"] = True
                    exp["directory"] = out_dir

                    # Update the experiments data to disk.
                    utils.dict_to_json(experiments, experiments_path)

    logger.log(f"Experiments data written to {experiments_path.as_posix()!r}")


def prepare_experiments(dfaas_config_path):
    # Read the DFAAS configuration file.
    if dfaas_config_path is None:
        dfaas_config_path = Path(Path.cwd(), "dfaas_config.json")
        logger.warn(f"No DFAAS config JSON file given, using default {dfaas_config_path.as_posix()!r}")
    elif isinstance(dfaas_config_path, str):
        # Make sure dfaas_config_path is a Path object
        dfaas_config_path = Path(dfaas_config_path).absolute()
    dfaas_config = utils.json_to_dict(dfaas_config_path)

    # Check the "results_directory" property in the configuration. This property
    # sets the base directory where the main global experiment is run.
    if not_defined("results_directory", dfaas_config):
        prefix_dir = Path.home() / "ray_results"
    else:
        prefix_dir = Path(dfaas_config["results_directory"])

    # The results directory is a unique root directory under the specified
    # directory where all experiments are stored.
    results_dir = Path(f"DFAAS_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    results_dir = prefix_dir / results_dir
    if results_dir.exists():
        logger.warn(f"Results directory {results_dir.as_posix()!r} alreasy exists")
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"DFAAS experiment directory created at {results_dir.as_posix()!r}")

    # Overwrite the original property, it will be used by subsequent
    # experiments.
    dfaas_config["results_directory"] = results_dir.as_posix()

    # Number of seeds to try for each experiment.
    default_seeds = 5
    if not_defined("seeds", dfaas_config):
        logger.warn(f"No 'seeds' property found, using default ({default_seeds})")
        dfaas_config["seeds"] = default_seeds

    # The user can give the values of the seeds, but it is optional: if
    # 'seeds_values' is not given, we generate new seeds.
    if not_defined("seeds_values", dfaas_config):
        logger.warn("No 'seeds_values' property found, generating new seeds")

        # Generate the seeds used by each class of experiments. Note that the
        # seed must be a non-negative 32 bit integer.
        seed_gen = np.random.default_rng()
        iinfo = np.iinfo(np.uint32)
        seeds = seed_gen.integers(0, high=iinfo.max, size=dfaas_config["seeds"])
    elif dfaas_config["seeds"] != len(dfaas_config["seeds_values"]):
        # If the user give a list of seeds, must equals the given number of
        # seeds.
        logger.err(f"'seeds_values' list length ({len(dfaas_config['seed_values'])}) must be equal to 'seeds' ({dfaas_config['seeds']})")
        sys.exit(1)
    else:
        seeds = np.array(dfaas_config["seeds_values"], dtype=np.uint32)

    # Save the updated dfaas config to the results directory.
    mod_dfaas_config_path = Path(results_dir, "dfaas_config.json")
    utils.dict_to_json(dfaas_config, mod_dfaas_config_path)
    logger.log(f"DFAAS config written to {mod_dfaas_config_path.as_posix()!r}")

    # Experiments is a dictionary that contains all the experiments done, in
    # progress and to do. It matches an experiment with a specific subfolder
    # created by RL4CC TrainingExperiment.
    experiments = {}

    # Generate all experiments ID.
    algorithms = ["PPO", "SAC"]
    scenarios = ["scenario1", "scenario2", "scenario3"]
    parameters = ["standard", "tuned"]
    for algo in algorithms:
        experiments[algo] = {}

        for params in parameters:
            experiments[algo][params] = {}

            for scenario in scenarios:
                experiments[algo][params][scenario] = {}

                for seed_idx in range(len(seeds)):
                    exp_id = ":".join([algo, params, scenario, str(seed_idx)])

                    experiments[algo][params][scenario][exp_id] = {
                        "id": exp_id,
                        "seed_idx": seed_idx,
                        # item() because Python JSON module doesn't support np.int64
                        "seed": seeds[seed_idx].item(),
                        "done": False,
                        }

    return dfaas_config, experiments


def train(exp_config):
    logger.log("START of training experiment")
    logger.log(f"  Algorithm: {exp_config['algorithm']}")
    logger.log(f"  Environment scenario: {exp_config['env_config']['scenario']}")
    logger.log(f"  Seed: {exp_config['env_config']['seed']}")

    exp = TrainingExperiment(exp_config=exp_config)
    policy = exp.run()

    return exp, policy


def evaluate(eval_config, train_experiment, policy):
    """Evaluation evaluates the given policy and the trained experiment with all
    scenarios and writes the output as "evaluation_scenarios.json" in the
    experiment directory.

    The evaluation can be configured using the eval_config dictionary. It has
    the following keys:

    - num_episodes_for_scenario: how many episodes to run for a single scenario
      (positive integer),
    - allow_exploration: whether the policy can select new exploration actions
      instead of just the best rewarded action (boolean),
    - starting_seed: the seed used to create the RNG that generates the seeds
      used for each episode.
    """
    episodes = eval_config["num_episodes_for_scenario"]
    allow_exploration = eval_config["allow_exploration"]
    starting_seed = eval_config["starting_seed"]

    # Results dictionary. "evaluations" contains the results for each scenario
    # evaluation.
    results = {
        "train_scenario": train_experiment.env_config["scenario"],
        "num_episodes_for_scenario": episodes,
        "allow_exploration": allow_exploration,
        "starting_seed": starting_seed,
        "scenarios": {}
    }

    # Scenarios to test.
    for scenario in TrafficManagementEnv.get_scenarios():
        # Copy and adjust the original env_config.
        env_config = deepcopy(train_experiment.exp_config["env_config"])
        env_config["scenario"] = scenario

        env = TrafficManagementEnv(env_config)

        # Evaluate the policy with the given environment.
        result = evaluate_policy(policy=policy,
                                 env=env,
                                 num_eval_episodes=episodes,
                                 explore=allow_exploration,
                                 seed=starting_seed)

        results["scenarios"][scenario] = result

    eval_results_path = Path(train_experiment.logdir, "evaluations_scenarios.json")
    utils.dict_to_json(results, eval_results_path)
    logger.log(f"Evaluation of {train_experiment.exp_config['id']!r} saved in {eval_results_path.as_posix()!r}")


def from_checkpoint():
    print("--- From checkpoint!")

    logdir = "/home/emanuele/ray_results/PPO_TrafficManagementEnv_2024-05-29_11-54-17rq9ys4jm"

    algo = Algorithm(algo_name="PPO", checkpoint_path=logdir + "/checkpoints/1")
    print(algo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="dfaas-rl",
                                     description="Training and evaluation of DFAAS RL")

    parser.add_argument("--dfaas-config",
                        help="DFAAS JSON configuration file.",
                        dest="dfaas_config_path")
    parser.add_argument("--experiments", "-e",
                        help="Run only experiments with the given prefix ID (a list)",
                        action="append",
                        default=[],
                        dest="experiments_prefix")

    args = parser.parse_args()

    main(args.dfaas_config_path, args.experiments_prefix)
