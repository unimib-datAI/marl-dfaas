from pathlib import Path
import argparse
import concurrent.futures

import matplotlib

import utils
from traffic_env import TrafficManagementEnv

import plots.single_exp
import plots.multiple_exp
import plots.scenario_env
import plots.algo_eval_params

from RL4CC.utilities.logger import Logger


logger = Logger(name="DFAAS-PLOTS", verbose=2)


def make_experiments_plots(exp_dir, exp_prefix):
    """Create plots for the DFaaS RL experiments."""
    # Read and parse the experiments.json file.
    experiments_path = Path(exp_dir, "experiments.json")
    experiments = utils.json_to_dict(experiments_path)

    # Directory that will contains plots related to macro-experiments (not the
    # single experiment).
    Path(exp_dir, "plots", "training").mkdir(parents=True, exist_ok=True)

    # Make plots concurrently.
    executor = concurrent.futures.ThreadPoolExecutor()
    tasks = []

    # Make plots for all experiments.
    for (algo, algo_values) in experiments.items():
        for (params, params_values) in algo_values.items():
            for (scenario, scenario_value) in params_values.items():
                exp_dirs = []
                all_done = False
                for exp in scenario_value.values():
                    # Make plots only for experiments selected by the user
                    # (only if the user give the argument, otherwise
                    # consider all experiments).
                    if (exp_prefix is not None and len(exp_prefix) >= 1
                       and not exp["id"].startswith(tuple(exp_prefix))):
                        continue

                    # Make plots only for finished experiments.
                    if not exp["done"]:
                        continue

                    # Make plots for a single experiment. We only give the
                    # experiment's directory and its ID.
                    exp_directory = Path(exp_dir, exp["directory"])
                    exp_dirs.append(exp_directory)

                    task = executor.submit(plots.single_exp.make, exp_directory, exp["id"])
                    tasks.append(task)

                    all_done = True

                # When making aggregate plots, all sub-experiments must be
                # run.
                exp_id = f"{algo}:{params}:{scenario}"
                if not all_done:
                    continue
                task = executor.submit(plots.multiple_exp.make, exp_dirs, exp_id, exp_dir)
                tasks.append(task)

    executor.shutdown()

    for task in tasks:
        # If there is an exception, it will be thrown.
        task.result()


def make_environment_plots(exp_dir):
    """Makes some basic plots to show how the scenarios generate input requests
    and forwarding capacity for each scenario."""
    plots_dir = Path(exp_dir, "plots", "environment")
    plots_dir.mkdir(parents=True, exist_ok=True)

    executor = concurrent.futures.ThreadPoolExecutor()
    tasks = []

    for scenario in TrafficManagementEnv.get_scenarios():
        task = executor.submit(plots.scenario_env.make, plots_dir, scenario)
        tasks.append(task)

    executor.shutdown()

    for task in tasks:
        task.result()


def make_algo_plots(exp_dir):
    """Makes summary plots for each algorithm in terms of scenarios and
    parameters."""
    # Read and parse the experiments.json file.
    experiments_path = Path(exp_dir, "experiments.json")
    experiments = utils.json_to_dict(experiments_path)

    for (algo, algo_values) in experiments.items():
        # All sub-experiments must be done, otherwise can't make the general
        # plot for the algorithm.
        all_done = True
        for (params, params_values) in algo_values.items():
            for (scenario, scenario_value) in params_values.items():
                for exp in scenario_value.values():
                    if not exp["done"]:
                        all_done = False

        if all_done:
            plots.algo_eval_params.make(exp_dir, algo)


def main(exp_dir, exp_prefix):
    plots_path = Path(exp_dir, "plots")
    plots_path.mkdir(parents=True, exist_ok=True)

    # Use only the PDF (non-interactive) backend.
    matplotlib.use("pdf", force=True)

    make_experiments_plots(exp_dir, exp_prefix)

    make_environment_plots(exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="make_plots",
                                     description="Make plots from training phase")

    parser.add_argument(dest="experiments_directory",
                        help="Directory with the experiments.json file")
    parser.add_argument("--experiments", "-e",
                        help="Make plots only for experiments with the given prefix ID (a list)",
                        action="append",
                        default=[],
                        dest="experiments_prefix")

    args = parser.parse_args()

    main(args.experiments_directory, args.experiments_prefix)
