"""This script runs a grid search over entropy coefficient start values and
entropy coefficient decay percentual over total iterations.

Saves each experiment with a different name.

The base config is taken from a real experiment, with synthetic workload.
"""

import multiprocessing
from dfaas_train import run_experiment

# Base config, taken from a real experiment.
base_exp_config = {
    "algorithm": {
        "name": "PPO",
        "gamma": 0.7,
        "lambda": 0.3,
        "entropy_coeff": None,  # Will be set in the loop
        "entropy_coeff_decay_enable": True,
        "entropy_coeff_decay_iterations": None,  # Will be set in the loop
    },
    "disable_gpu": False,
    "iterations": 2000,
    "training_num_episodes": 1,
    "runners": 0,
    "model": "configs/models/softplus.json",
    "seed": 42,
    "checkpoint_interval": 200,
    "evaluation_interval": 25,
    "evaluation_num_episodes": 10,
    "final_evaluation": True,
}

env_config = {
    "network": ["node_0 node_1 node_2", "node_3 node_2 node_0", "node_1 node_4"],
    "input_rate_same_method": True,
    "input_rate_method": "synthetic-sinusoidal",
}

# Will use default values.
runners = None
seed = None
dry_run = False

# Grid search over these values.
entropy_coeff_values = [0.005, 0.01, 0.02, 0.05]
entropy_decay_iter_values = [0.6, 0.7, 0.8]


def experiment_job(args):
    """Runs a single experiment."""
    entropy_coeff, decay_iter = args
    exp_config = base_exp_config.copy()
    exp_config["algorithm"] = exp_config["algorithm"].copy()
    exp_config["algorithm"]["entropy_coeff"] = entropy_coeff
    exp_config["algorithm"]["entropy_coeff_decay_iterations"] = decay_iter

    local_suffix = f"5_decay_ec{entropy_coeff}_di{decay_iter}"
    print(f"-- Running with entropy_coeff={entropy_coeff}, decay_iter={decay_iter}")
    run_experiment(
        suffix=local_suffix, exp_config=exp_config, env_config=env_config, runners=runners, seed=seed, dry_run=dry_run
    )


def main():
    """Main entry point for running entropy coefficient experiments in parallel.

    Generates all combinations of entropy_coeff and decay_iter values, then runs
    each experiment in parallel using multiprocessing.
    """
    jobs = [
        (entropy_coeff, decay_iter)
        for entropy_coeff in entropy_coeff_values
        for decay_iter in entropy_decay_iter_values
    ]

    with multiprocessing.Pool(processes=4) as pool:
        # Run jobs in parallel
        for _ in pool.imap_unordered(experiment_job, jobs):
            pass


if __name__ == "__main__":
    main()
