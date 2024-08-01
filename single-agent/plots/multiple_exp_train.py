from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

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

import utils

from RL4CC.utilities.logger import Logger

logger = Logger(name="DFAAS-PLOTS", verbose=2)


def _get_metrics_training_exp(exp_dir):
    # Each item is one experiment training iteration object.
    iters = []

    # Fill the iters list parsing the "result.json" file.
    idx = 0
    result_path = Path(exp_dir, "result.json")
    with result_path.open() as result:
        # The "result.json" file is not a valid JSON file. Each row is an
        # isolated JSON object, the result of one training iteration.
        while (raw_iter := result.readline()) != "":
            iters.append(json.loads(raw_iter))

    # Fill the arrays with data, they will be plotted.
    reward_mean = np.empty(shape=len(iters), dtype=np.float64)
    reward_total = np.empty(shape=len(iters), dtype=np.float64)
    congested_steps = np.empty(shape=len(iters), dtype=np.int64)
    rejected_reqs_total = np.empty(shape=len(iters), dtype=np.int64)
    idx = 0
    for iter in iters:
        # Ray post-processes the custo metrics calculatin for each value the
        # mean, min and max. So we have "reward_mean_mean", but we have only one
        # episode for iteration so it is the same as "reward_mean".
        #
        # Same applies for congested_steps and rejected_reqs_total.
        reward_mean[idx] = iter["custom_metrics"]["reward_mean_mean"]
        congested_steps[idx] = iter["custom_metrics"]["congested_steps_mean"]
        rejected_reqs_total[idx] = iter["custom_metrics"]["rejected_reqs_total_mean"]

        # Each iteration is one episode, so there is only one episode total
        # reward.
        reward_total[idx] = iter["hist_stats"]["episode_reward"][0]

        idx += 1

    assert idx == len(iters), f"Some elements of ndarrays were not accessed (idx = {idx}, tot. iters = {len(iters)}"

    result = {
            "reward_mean": reward_mean,
            "reward_total": reward_total,
            "congested_steps": congested_steps,
            "rejected_reqs_total": rejected_reqs_total,
            }

    return result


def make(exp_dirs, exp_id, res_dir):
    logger.log(f"Making summary training aggregate plots for {exp_id!r}")

    # Each value is a ndarray for each experiment.
    reward_mean = []
    reward_total = []
    congested_steps = []
    rejected_reqs_total = []
    for exp_dir in exp_dirs:
        result = _get_metrics_training_exp(exp_dir)
        reward_mean.append(result["reward_mean"])
        reward_total.append(result["reward_total"])
        congested_steps.append(result["congested_steps"])
        rejected_reqs_total.append(result["rejected_reqs_total"])

    num_exps = len(exp_dirs)  # Number of experiments.

    # Now make the aggregated plot. A figure with four plots: one for each
    # metrics.
    fig = plt.figure(figsize=(19.2, 14.3), dpi=300, layout="constrained")
    fig.suptitle(exp_id)
    axs = fig.subplots(ncols=2, nrows=2)

    # Reward mean plot.
    axs[0, 0].set_title("Reward mean")
    for exp in range(num_exps):
        axs[0, 0].plot(reward_mean[exp], label=f"Seed nr. {exp}", alpha=.4)
    # Print the mean of each reward mean (column by column, this is why axis=0).
    axs[0, 0].plot(np.mean(reward_mean, axis=0), color="r", label="Mean")
    axs[0, 0].legend()

    # Reward total plot.
    axs[0, 1].set_title("Reward total")
    for exp in range(num_exps):
        axs[0, 1].plot(reward_total[exp], label=f"Seed nr. {exp}", alpha=.4)
    axs[0, 1].plot(np.mean(reward_total, axis=0), color="r", label="Mean")
    axs[0, 1].legend()

    # Congested steps plot.
    axs[1, 0].set_title("Congested steps")
    for exp in range(num_exps):
        axs[1, 0].plot(congested_steps[exp], label=f"Seed nr. {exp}", alpha=.4)
    axs[1, 0].plot(np.mean(congested_steps, axis=0), color="r", label="Mean")
    axs[1, 0].legend()

    # Rejected requests total.
    axs[1, 1].plot(rejected_reqs_total)
    axs[1, 1].set_title("Rejected requests total")
    for exp in range(num_exps):
        axs[1, 1].plot(rejected_reqs_total[exp], label=f"Seed nr. {exp}", alpha=.4)
    axs[1, 1].plot(np.mean(rejected_reqs_total, axis=0), color="r", label="Mean")
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.set_xlabel("Iteration")

    path = Path(res_dir, "plots", "training", f"{exp_id}.pdf")
    fig.savefig(path)
    plt.close(fig)
    logger.log(f"{exp_id}: {path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser(prog="multiple_exp_train")

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
