import sys
import json
from pathlib import Path
import argparse
from datetime import datetime
from copy import deepcopy

# I need to log stdout and stderr to a log file. For this reason, I need to
# replace sys.stdout and sys.stderr with a custom wrapper as quickly as
# possible. This wrapper will be used transparently by Ray, RL4CC and my code.
#
# The only difference is that after the experiment is run, I can set the output
# path for the log, inside the "run()" function.
#
# This is a dirty hacky.
import utils
#sys.stdout = utils.OutputDuplication()
#sys.stderr = sys.stdout

import numpy as np

from RL4CC.experiments.train import TrainingExperiment
from RL4CC.algorithms.algorithm import Algorithm
from RL4CC.utilities.postprocessing import evaluate_policy
from RL4CC.utilities.logger import Logger
from RL4CC.utilities.common import load_config_file, write_config_file, not_defined

import traffic_env

# Logger used in this file.
logger = Logger(name="DFAAS", verbose=2)

def main(dfaas_config_path, experiments_prefix):
    dfaas_config, experiments = prepare_experiments(dfaas_config_path)

    # Default exp_config with env_config and ray_config.
    default_exp_config = {
            "algorithm": "PPO",
            "env_config": {
                "env_name": "TrafficManagementEnv",
                "min_time": 0,
                "max_time": 100,
                "time_step": 1,
                "scenario": "scenario1"
                },
            "ray_config": {
                "framework": "torch",
                "callbacks": {
                    "callbacks_class": "RL4CC.callbacks.base_callbacks.BaseCallbacks"
                    },
                "rollouts": {
                    "duration_unit": "episodes",
                    "duration_per_worker": 1,
                    "num_rollout_workers": 1
                    },
                "training": {
                    "batch_size": 64,
                    "num_train_batches": 5
                    },
                "debugging": {
                    "log_level": "INFO"
                    }
                },
            "stopping_criteria": {
                "max_iterations": 1
                },
            "logger": {
                "verbosity": 2
                },
            "evaluation": {
                "num_episodes_for_scenario": 50
                },
            }

    # The absolute path to the DFAAS output directory.
    dfaas_dir = Path(dfaas_config["results_directory"]).absolute()

    # Run all experiments.
    for algo in experiments.values():
        for params in algo.values():
            for scenario in params.values():
                for exp in scenario.values():
                    if (experiments_prefix is not None
                        and len(experiments_prefix) >= 1
                        and not exp["id"].startswith(tuple(experiments_prefix))):
                        logger.log(f"Skipping experiment ID {exp['id']}")
                        continue
                    logger.log(f"Running experiment ID {exp['id']}")

                    # Base on the global exp_config, copy the config and set the
                    # experiment specific settings.
                    exp_config = deepcopy(default_exp_config)
                    exp_config["logdir"] = dfaas_config['results_directory']
                    exp_config["seed"] = exp_config["env_config"]["seed"] = exp["seed"]
                    exp_config["id"] = exp["id"]

                    # Run the experiment.
                    train_exp, policy = train(exp_config)

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
                    dump = json.dumps(experiments)
                    write_config_file(dump, dfaas_config["results_directory"], "experiments.json")

    path = Path(dfaas_config["results_directory"], "experiments.json").absolute()
    logger.log(f"Experiments data written to {path.as_posix()!r}")


def prepare_experiments(dfaas_config_path):
    default_dfaas_config_path = "dfaas_config.json"
    if dfaas_config_path is None:
        logger.warn(f"No DFAAS config JSON file given, using default ({default_dfaas_config_path!r})")
        dfaas_config_path = default_dfaas_config_path

    # Read the DFAAS configuration file.
    dfaas_config = load_config_file(dfaas_config_path)
    if dfaas_config is None:
        logger.err(f"Failed to read DFAAS config file {dfaas_config_path!r}")
        sys.exit(1)

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

    default_seeds = 5
    if not_defined("seeds", dfaas_config):
        logger.warn(f"No 'seeds' property found, using default ({default_seeds})")
        dfaas_config["seeds"] = default_seeds

    # Save the modified dfaas_config to disk as a JSON file.
    dump = json.dumps(dfaas_config)
    path = write_config_file(dump, results_dir.as_posix(), default_dfaas_config_path)
    logger.log(f"DFAAS config written to {path!r}")

    # Generate the seeds used by each class of experiments.
    seed_gen = np.random.default_rng()
    iinfo = np.iinfo(np.int64)
    seeds = seed_gen.integers(iinfo.min, high=iinfo.max, size=dfaas_config["seeds"])

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
    logger.log(f"START of training experiment")
    logger.log(f"  Algorithm: {exp_config['algorithm']}")
    logger.log(f"  Environment scenario: {exp_config['env_config']['scenario']}")
    logger.log(f"  Seed: {exp_config['seed']}")

    exp = TrainingExperiment(exp_config=exp_config)
    policy = exp.run()

    evaluation(exp, policy)

    return exp, policy

def get_eval_config(exp_config):
    """Returns a dictionary representing the evaluation configuration.

    The evaluation config is extracted and built from the given experiment
    config.

    The returned dictionary has the following keys:

        * "num_episodes_for_scenario" of type int (default is 1),
        * "allow_exploration" of type bool (default is False).
    """
    # The evaluation config dictionary is written directly to the given
    # experiment config. RL4CC ignores any unknown keys in the config. This step
    # is also done after RL4CC has trained the model.
    if "evaluation" not in exp_config.keys():
        exp_config["evaluation"] = {}

    eval_config = exp_config["evaluation"]

    num_eps_key = "num_episodes_for_scenario"
    if num_eps_key not in eval_config.keys():
        eval_config[num_eps_key] = 1
    elif (t := type(eval_config[num_eps_key])) is not int:
        logger.err(f"{num_eps_key!r} must be of type int, is {t.__name__}")
        sys.exit(1)

    explore_key = "allow_exploration"
    if explore_key not in eval_config.keys():
        eval_config[explore_key] = False
    elif (t := type(eval_config[explore_key])) is not bool:
        logger.err(f"{explore_key!r} must be of type bool, is {t.__name__}")
        sys.exit(1)

    return eval_config


def evaluation(experiment, policy):
    """Evaluations evaluates a single experiment against all three scenarios and
    saves the result to a JSON file in the experiment results directory."""
    config = get_eval_config(experiment.exp_config)

    episodes = config["num_episodes_for_scenario"]
    allow_exploration = config["allow_exploration"]

    # Results dictionary. "evaluations" contains the results for each scenario
    # evaluation.
    results = {
        "train_scenario": experiment.env_config["scenario"],
        "num_episodes_for_scenario": episodes,
        "allow_exploration": allow_exploration,
        "evaluations": {}
    }

    # Scenarios to test.
    for scenario in ["scenario" + str(i+1) for i in range(3)]:
        # Get original env_config from the experiment.
        env_config = deepcopy(experiment.exp_config["env_config"])

        # Set the scenario the policy will be evaluated with.
        env_config["scenario"] = scenario
        env = traffic_env.TrafficManagementEnv(env_config)

        # Evaluate the model with the specified scenario.
        result = evaluate_policy(policy=policy,
                                 env=env,
                                 num_eval_episodes=episodes,
                                 explore=allow_exploration)

        results["evaluations"][scenario] = result

    # Save results as JSON file to disk.
    dump = json.dumps(results, cls=utils.NumpyEncoder)
    path = write_config_file(dump, experiment.logdir, "evaluations_scenarios.json")

    logger.log(f"Evaluation of {experiment.exp_config['id']!r} saved in {path!r}")


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
