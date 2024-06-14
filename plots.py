from pathlib import Path
import argparse
import json
import concurrent.futures

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import utils
from traffic_env import TrafficManagementEnv

from RL4CC.utilities.logger import Logger

logger = Logger(name="DFAAS-PLOTS", verbose=2)


def make_training_plot(exp_dir, exp_id):
    """Makes the plots related to the training phase for the given
    experiment. Data is extracted from the "result.json" file.

    The experiment ID is needed to annotate the plots."""
    result = get_metrics_training_exp(exp_dir)
    reward_mean = result["reward_mean"]
    reward_total = result["reward_total"]
    congested_steps = result["congested_steps"]
    rejected_reqs_total = result["rejected_reqs_total"]

    # Generate now the plots using the arrays for the x axis.
    fig = plt.figure(figsize=(19.2, 14.3), dpi=300, layout="constrained")
    fig.suptitle(exp_id)
    axs = fig.subplots(ncols=2, nrows=2)

    axs[0, 0].plot(reward_mean)
    axs[0, 0].set_title("Reward mean")

    axs[0, 1].plot(reward_total)
    axs[0, 1].set_title("Reward total")

    axs[1, 0].plot(congested_steps)
    axs[1, 0].set_title("Congested steps")

    axs[1, 1].plot(rejected_reqs_total)
    axs[1, 1].set_title("Rejected requests total")

    for ax in axs.flat:
        ax.set_xlabel("Iteration")

    # Save the plot.
    fig.savefig(Path(exp_dir, "plots", "training_plot.pdf"))
    plt.close(fig)


def make_evaluation_plot(exp_dir, exp_id):
    # Read raw evaluations data from the experiment.
    eval_path = Path(exp_dir, "evaluations_scenarios.json")
    data = utils.json_to_dict(eval_path)

    num_episodes = data["num_episodes_for_scenario"]
    # We assume that each episode has the same evaluation steps.
    eval_steps = len(data["scenarios"]["scenario1"][0]["evaluation_steps"])

    for scenario in data["scenarios"]:
        reward_mean = np.empty(shape=(num_episodes, eval_steps))
        reward_total = np.empty(shape=(num_episodes, eval_steps))
        congested_steps = np.empty(shape=(num_episodes, eval_steps))
        rejected_reqs_total = np.empty(shape=(num_episodes, eval_steps))

        # Fill the arrays with the data.
        for episode in data["scenarios"][scenario]:
            ep_idx = episode["episode"]

            # There is not "rejected_reqs_total" inside the data for each step.
            # We need to calculate this value manually step after step.
            rejected_reqs_total_tmp = 0

            for eval_step in episode["evaluation_steps"]:
                step_idx = eval_step["step"]

                rejected_reqs_total_tmp += eval_step["obs_info"]["actions"]["rejected"]

                reward_mean[ep_idx, step_idx] = eval_step["reward"]
                reward_total[ep_idx, step_idx] = eval_step["total_reward"]
                congested_steps[ep_idx, step_idx] = eval_step["obs_info"]["congested"]
                rejected_reqs_total[ep_idx, step_idx] = rejected_reqs_total_tmp

            assert step_idx == eval_steps-1, f"step_idx is {step_idx}, it should be {eval_steps}"

        assert ep_idx == num_episodes-1, f"ep_idx is {ep_idx}, it should be {num_episodes}"

        # Now make the aggregated plot. A figure with four plots: one for each
        # metrics.
        fig = plt.figure(figsize=(19.2, 14.3), dpi=300, layout="constrained")
        fig.suptitle(f"Evaluation {exp_id} with scenario {scenario!r}")
        axs = fig.subplots(ncols=2, nrows=2)

        # Reward mean plot.
        axs[0, 0].set_title("Reward mean (each step)")
        for step in range(eval_steps):
            # We need to plot the entire column, not the row! Because each
            # column is a single step in the episode for all episodes.
            axs[0, 0].plot(reward_mean[:, step], alpha=.4)
        # Print the mean of each reward mean (column by column, this is why axis=0).
        axs[0, 0].plot(np.mean(reward_mean, axis=0), color="r", label="Mean")
        axs[0, 0].legend()

        # Reward total plot.
        axs[0, 1].set_title("Reward total (cumulative)")
        for step in range(eval_steps):
            axs[0, 1].plot(reward_total[:, step], alpha=.4)
        axs[0, 1].plot(np.mean(reward_total, axis=0), color="r", label="Mean")
        axs[0, 1].legend()

        # Congested steps plot.
        axs[1, 0].set_title("Congested steps (each step)")
        for step in range(eval_steps):
            axs[1, 0].plot(congested_steps[:, step], alpha=.4)
        axs[1, 0].plot(np.mean(congested_steps, axis=0), color="r", label="Mean")
        axs[1, 0].legend()

        # Rejected requests total.
        axs[1, 1].plot(rejected_reqs_total)
        axs[1, 1].set_title("Rejected requests total (cumulative)")
        for step in range(eval_steps):
            axs[1, 1].plot(rejected_reqs_total[:, step], alpha=.4)
        axs[1, 1].plot(np.mean(rejected_reqs_total, axis=0), color="r", label="Mean")
        axs[1, 1].legend()

        for ax in axs.flat:
            ax.set_xlabel("Steps in each episode")

        # Save the plot.
        fig.savefig(Path(exp_dir, "plots", f"eval_{scenario}.pdf"))
        plt.close(fig)


def single_exp_iter(exp_dir, exp_id):
    plots_dir = Path(exp_dir, "plots")

    # Number of iteration steps when the plots will be made.
    iter_plots_each = 500

    # Each item is an iteration object, but only the steps provided by
    # iter_plots.
    iters = {}

    # Filter the "result.json" and load only the interested iterations.
    result_path = Path(exp_dir, "result.json")
    with result_path.open() as result:
        # The "result.json" file is not a valid JSON file. Each row is an
        # isolated JSON object, the result of one training iteration.
        iter_idx = 0
        while (raw_iter := result.readline()) != "":
            # We need to subtract 1 because the steps starts from zero.
            if iter_idx % (iter_plots_each - 1) == 0:
                iters[iter_idx] = json.loads(raw_iter)

            iter_idx += 1

    for (iter_idx, iter) in iters.items():
        steps = iter["hist_stats"]["episode_lengths"][0]

        input_requests = np.empty(shape=steps, dtype=np.int64)
        queue_capacity = np.empty(shape=steps, dtype=np.int64)
        forwarding_capacity = np.empty(shape=steps, dtype=np.int64)
        congested = np.empty(shape=steps, dtype=np.int64)
        actions = np.empty(shape=(steps, 3), dtype=np.int64)

        for step in range(steps):
            input_requests[step] = iter["hist_stats"]["input_requests"][step]
            queue_capacity[step] = iter["hist_stats"]["queue_capacity"][step]
            forwarding_capacity[step] = iter["hist_stats"]["forwarding_capacity"][step]
            congested[step] = iter["hist_stats"]["congested"][step]
            actions[step] = iter["hist_stats"]["actions"][step]

        assert step == steps-1, f"step is {step}, expected {steps-1}"

        # Make the plot.
        fig = plt.figure(figsize=(28.79, 21.45), dpi=600, layout="constrained")
        fig.suptitle(f"{exp_id} on iteration {iter_idx}")
        axs = fig.subplots(ncols=1, nrows=3)

        # We need to shift the lines and bars one to the right. Why? Because the
        # actions of a step refer to the previously observed environment.
        #
        # So we start from step=1 in the X axis.
        steps_x = np.arange(start=1, stop=steps)

        # Make the inverse array of congested_state.
        not_congested = np.asarray(congested, dtype=bool)
        not_congested = np.invert(not_congested)
        not_congested = np.asarray(not_congested, dtype=np.int64)

        # Do not stack the bars, because the two input array is one inverse of
        # the other, so they don't stack or overlap.
        axs[0].bar(x=steps_x, height=congested[:-1], color="r", label="Congested")
        axs[0].bar(x=steps_x, height=not_congested[:-1], color="g", label="Not congested")
        axs[0].set_title("Congested state")

        axs[1].plot(steps_x, forwarding_capacity[:-1], label="Forwarding capacity")
        axs[1].plot(steps_x, queue_capacity[:-1], label="Queue capacity")
        axs[1].set_title("Forwarding and Queue capacities")
        axs[1].legend()

        # Get the action by column and skip the first item.
        local = actions[1:, 0]
        forwarded = actions[1:, 1]
        rejected = actions[1:, 2]

        axs[2].bar(x=steps_x, height=local, label="Local")
        axs[2].bar(x=steps_x, height=forwarded, label="Forwarded", bottom=local)
        axs[2].bar(x=steps_x, height=rejected, label="Rejected", bottom=local+forwarded)
        axs[2].plot(steps_x, input_requests[:-1], linewidth=3, color="r", label="Input requests")
        axs[2].set_title("Actions")
        axs[2].legend()

        for ax in axs.flat:
            ax.set_xlabel("Step")

        # Save the plot.
        path = Path(plots_dir, f"training_iter_{iter_idx}.pdf")
        fig.savefig(path)
        plt.close(fig)
        logger.log(f"{exp_id}: {path.as_posix()!r}")


def make_plots_single_experiment(exp_dir, exp_id):
    """Makes plots related to a single experiment. The plots are stored in the
    'plots' directory within the experiment directory."""
    logger.log(f"Making plots for experiment ID {exp_id}")

    # Make sure the directory is created.
    Path(exp_dir, "plots").mkdir(parents=True, exist_ok=True)

    make_training_plot(exp_dir, exp_id)

    single_exp_iter(exp_dir, exp_id)

    # TODO: useless plots?
    # make_evaluation_plot(exp_dir, exp_id)


def get_metrics_training_exp(exp_dir):
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


def make_aggregate_plots(exp_dirs, exp_id, res_dir):
    logger.log(f"Making aggregate plots for {exp_id!r}")

    # Each value is a ndarray for each experiment.
    reward_mean = []
    reward_total = []
    congested_steps = []
    rejected_reqs_total = []
    for exp_dir in exp_dirs:
        result = get_metrics_training_exp(exp_dir)
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

    fig.savefig(Path(res_dir, "plots", "training", f"{exp_id}.pdf"))
    plt.close(fig)


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
                        logger.warn(f"Skipping experiment ID {exp['id']}")
                        continue

                    # Make plots only for finished experiments.
                    if not exp["done"]:
                        logger.warn(f"Skipping experiment {exp['id']!r} because it is not done")
                        continue

                    # Make plots for a single experiment. We only give the
                    # experiment's directory and its ID.
                    exp_directory = Path(exp_dir, exp["directory"])
                    exp_dirs.append(exp_directory)

                    task = executor.submit(make_plots_single_experiment, exp_directory, exp["id"])
                    tasks.append(task)

                    all_done = True

                # When making aggregate plots, all sub-experiments must be
                # run.
                exp_id = f"{algo}:{params}:{scenario}"
                if not all_done:
                    logger.warn(f"Skipping making plot for aggregate experiment {exp_id!r} because not all experiments are done")
                    continue
                task = executor.submit(make_aggregate_plots, exp_dirs, exp_id, exp_dir)
                tasks.append(task)

    executor.shutdown()

    for task in tasks:
        # If there is an exception, it will be thrown.
        task.result()


def make_scenario_plots(plots_dir, scenario):
    logger.log(f"Making plots for scenario {scenario!r}")

    env = TrafficManagementEnv({"scenario": scenario})
    env.reset()

    step = 0
    max_steps = env.max_steps

    input_requests = np.empty(shape=env.max_steps, dtype=np.int64)
    forward_capacity = np.empty(shape=env.max_steps, dtype=np.int64)

    while step < max_steps:
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

        input_requests[step] = obs[0]
        forward_capacity[step] = obs[1]

        step += 1

    # Make the plot.
    fig = plt.figure(figsize=(19.2, 14.3), dpi=300, layout="constrained")
    fig.suptitle(f"Scenario {scenario!r}")
    axs = fig.subplots(ncols=1, nrows=2)

    axs[0].plot(input_requests)
    axs[0].set_title("Input requests")

    axs[1].plot(forward_capacity)
    axs[1].set_title("Forwarding capacity")

    for ax in axs.flat:
        ax.set_xlabel("Step")

    # Save the plot.
    fig.savefig(Path(plots_dir, f"{scenario}.pdf"))
    plt.close(fig)


def make_environment_plots(exp_dir):
    """Makes some basic plots to show how the scenarios generate input requests
    and forwarding capacity for each scenario."""
    plots_dir = Path(exp_dir, "plots", "environment")
    plots_dir.mkdir(parents=True, exist_ok=True)

    executor = concurrent.futures.ThreadPoolExecutor()
    tasks = []

    for scenario in TrafficManagementEnv.get_scenarios():
        task = executor.submit(make_scenario_plots, plots_dir, scenario)
        tasks.append(task)

    executor.shutdown()

    for task in tasks:
        task.result()


def main(exp_dir, exp_prefix):
    plots_path = Path(exp_dir, "plots")
    plots_path.mkdir(parents=True, exist_ok=True)

    # Use only the PDF (non-interactive) backend.
    matplotlib.use("pdf", force=True)

    make_experiments_plots(exp_dir, exp_prefix)

    make_environment_plots(exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="plots",
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
