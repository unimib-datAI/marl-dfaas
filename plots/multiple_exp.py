from pathlib import Path
import traceback

if __name__ == "__main__":
    import sys
    import os
    import argparse
    import matplotlib

    # Add the current directory (where Python is called) to sys.path. This
    # assumes this script is called in the project root directory, not inside
    # the directory where the script is.
    #
    # Required when calling this module directly as main.
    sys.path.append(os.getcwd())

    import multiple_exp_train
else:
    from . import multiple_exp_train, multiple_exp_eval

import utils

from RL4CC.utilities.logger import Logger

logger = Logger(name="DFAAS-PLOTS", verbose=2)


def make(exp_dirs, exp_id, res_dir):
    logger.log(f"Making aggregate plots for {exp_id!r}")

    # Make sure the directory is created.
    Path(res_dir, "plots").mkdir(parents=True, exist_ok=True)

    try:
        multiple_exp_train.make(exp_dirs, exp_id, res_dir)
    except Exception:
        logger.err(f"Failed to make plots for training summary for aggregate experiment {exp_id!r}: {traceback.format_exc()}")

    try:
        multiple_exp_eval.make(exp_dirs, exp_id, res_dir)
    except Exception:
        logger.err(f"Failed to make plots for evaluation summary for aggregate experiment {exp_id!r}: {traceback.format_exc()}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser(prog="multiple_exp")

    parser.add_argument("experiment_directory",
                        help="DFaaS Experiment directory")
    parser.add_argument("experiment_id",
                        help="Which main experiment make plots (example 'PPO:standard:scenario1')")

    args = parser.parse_args()

    # Read and parse the experiments.json file.
    experiments_path = Path(args.experiment_directory, "experiments.json")
    experiments = utils.json_to_dict(experiments_path)

    # Directory that will contains plots related to macro-experiments (not the
    # single experiment).
    Path(args.experiment_directory, "plots", "training").mkdir(parents=True, exist_ok=True)

    # Make plots for all experiments.
    for (algo, algo_values) in experiments.items():
        for (params, params_values) in algo_values.items():
            for (scenario, scenario_value) in params_values.items():
                exp_id = f"{algo}:{params}:{scenario}"
                if args.experiment_id is not None and exp_id != args.experiment_id:
                    continue

                exp_dirs = []
                all_done = True
                for exp in scenario_value.values():
                    if not exp["done"]:
                        all_done = False
                        continue
                    exp_dirs.append(Path(args.experiment_directory, exp["directory"]))

                if all_done:
                    make(exp_dirs, exp_id, args.experiment_directory)
