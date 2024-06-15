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

    import single_exp_train_summary
    import single_exp_iter
    import single_eval
else:  # When this module is imported.
    from . import single_exp_train_summary
    from . import single_exp_iter
    from . import single_eval

from RL4CC.utilities.logger import Logger

logger = Logger(name="DFAAS-PLOTS", verbose=2)


def make(exp_dir, exp_id):
    """Makes plots related to a single experiment. The plots are stored in the
    'plots' directory within the experiment directory."""
    logger.log(f"Making plots for experiment ID {exp_id}")

    # Make sure the directory is created.
    Path(exp_dir, "plots").mkdir(parents=True, exist_ok=True)

    try:
        single_exp_train_summary.make(exp_dir, exp_id)
    except Exception:
        logger.err(f"Failed to make plots for summary of training of experiment {exp_id!r}: {traceback.format_exc()}")

    try:
        single_exp_iter.make(exp_dir, exp_id)
    except Exception:
        logger.err(f"Failed to make plots for single iterations during training of experiment {exp_id!r}: {traceback.format_exc()}")

    try:
        single_eval.make(exp_dir, exp_id)
    except Exception:
        logger.err(f"Failed to make plots for evaluation of experiment {exp_id!r}: {traceback.format_exc()}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser(prog="single_exp")

    parser.add_argument(dest="experiment_directory",
                        help="Experiment directory")

    args = parser.parse_args()

    exp_id = Path(args.experiment_directory).name

    make(args.experiment_directory, exp_id)
