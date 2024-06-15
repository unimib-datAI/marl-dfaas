from pathlib import Path
import argparse
import json
import concurrent.futures
import traceback

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


def get_cumulated_array(data, **kwargs):
    """Returns an ndarray with the cumulated sum passing kwargs to the clip
    method.

    Used to make stacked bar charta with negative and positive values."""
    # See: https://stackoverflow.com/a/38900035
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d


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

    # For each saved iteration, make the plot.
    for (iter_idx, iter) in iters.items():
        steps = iter["hist_stats"]["episode_lengths"][0]

        # Get the data first.
        input_requests = np.empty(shape=steps, dtype=np.int64)
        queue_capacity = np.empty(shape=steps, dtype=np.int64)
        forwarding_capacity = np.empty(shape=steps, dtype=np.int64)
        congested = np.empty(shape=steps, dtype=np.int64)
        actions = np.empty(shape=(steps, 3), dtype=np.int64)
        reward = np.empty(shape=steps)
        reward_components = np.empty(shape=(steps, 4))

        for step in range(steps):
            input_requests[step] = iter["hist_stats"]["input_requests"][step]
            queue_capacity[step] = iter["hist_stats"]["queue_capacity"][step]
            forwarding_capacity[step] = iter["hist_stats"]["forwarding_capacity"][step]
            congested[step] = iter["hist_stats"]["congested"][step]
            actions[step] = iter["hist_stats"]["actions"][step]
            reward[step] = iter["hist_stats"]["reward"][step]
            reward_components[step] = iter["hist_stats"]["reward_components"][step]

        assert step == steps-1, f"step is {step}, expected {steps-1}"

        # Then make the plot.
        fig = plt.figure(figsize=(38.38, 21.45), dpi=600, layout="constrained")
        fig.suptitle(f"{exp_id} on iteration {iter_idx}")

        # The plot has four sub-plots. Only the first should have smaller
        # dimensions because it is a plot about boolean values.
        axs = fig.subplots(ncols=1,
                           nrows=4,
                           sharex="col",  # Align x-axis for all plots.
                           gridspec_kw={"height_ratios": [1, 2, 2, 2]})

        # For the first three plots, the x-axis must be moved one to the right.
        # This is because the observed data refers to the previously observed
        # state.
        steps_x = np.arange(start=1, stop=steps)

        # First plot: whether a state is in a congested state or not.
        #
        # Make the inverse array of congested_state.
        not_congested = np.asarray(congested, dtype=bool)
        not_congested = np.invert(not_congested)
        not_congested = np.asarray(not_congested, dtype=np.int64)

        # Do not stack the bars, because the two input array is one inverse of
        # the other, so they don't stack or overlap.
        axs[0].bar(x=steps_x, height=congested[:-1], color="r", label="Congested")
        axs[0].bar(x=steps_x, height=not_congested[:-1], color="g", label="Not congested")
        axs[0].set_title("Congested state")

        # Second plot: the forwarding and queue capacities as lines.
        axs[1].plot(steps_x, forwarding_capacity[:-1], label="Forwarding capacity")
        axs[1].plot(steps_x, queue_capacity[:-1], label="Queue capacity")
        axs[1].set_title("Forwarding and Queue capacities")

        # Third plot: the action chosen by the agent with the input requests.
        #
        # The actions taken from the data are organized as an array for each
        # step. We need to extract each action component as a column.
        local = actions[1:, 0]
        forwarded = actions[1:, 1]
        rejected = actions[1:, 2]

        axs[2].bar(x=steps_x, height=local, label="Local")
        axs[2].bar(x=steps_x, height=forwarded, label="Forwarded", bottom=local)
        axs[2].bar(x=steps_x, height=rejected, label="Rejected", bottom=local+forwarded)
        axs[2].plot(steps_x, input_requests[:-1], linewidth=3, color="r", label="Input requests")
        axs[2].set_title("Actions")

        # Fourth plot: the reward and its components.
        #
        # Same as the previous plot, we need to extract the columns.
        reward_local = reward_components[:, 0]
        reward_forwarded = reward_components[:, 1]
        reward_rejected = reward_components[:, 2]
        reward_malus = reward_components[:, 3]

        # We rebuild the array because we need to process it.
        reward_components = np.array([reward_local,
                                      reward_forwarded,
                                      reward_rejected,
                                      reward_malus])

        # The stacked bar is not easy to make because some values are negative
        # and some are positive. We have to calculate the offset for each value
        # in the bar to get a correct positioning in the "bottom" keyword.
        #
        # Thanks to: https://stackoverflow.com/a/38900035
        cumulated_data = get_cumulated_array(reward_components, min=0)
        cumulated_data_neg = get_cumulated_array(reward_components, max=0)

        # Re-merge negative and positive data.
        row_mask = (reward_components < 0)
        cumulated_data[row_mask] = cumulated_data_neg[row_mask]
        offset_stack = cumulated_data

        steps_x = np.arange(1, steps+1)
        labels = ["Local", "Forwarded", "Rejected", "Malus"]
        for i in np.arange(0, len(labels)):
            axs[3].bar(steps_x, reward_components[i], bottom=offset_stack[i], label=labels[i])
        axs[3].plot(steps_x, reward, linewidth=3, color="r", label="Reward")
        axs[3].set_title("Reward")

        # Common settings for all plots.
        for ax in axs.flat:
            ax.set_xlabel("Step")
            # Because the plots shares the x-axis, only the fourth plots will
            # show the ticks, but I want to write the ticks also for the first
            # three plots.
            ax.tick_params(axis="x", labelbottom=True)
            # Show x-axis ticks every 10 steps from 0 to 100.
            ax.set_xticks(np.arange(0, steps+1, 10))

            ax.grid(which="both")
            ax.set_axisbelow(True)  # Place the grid behind the lines and bars.

            ax.legend()

        # The first plots is special, it needs to overwrite some common settings.
        axs[0].tick_params(reset=True)
        axs[0].set_yticks([0, 1])
        axs[0].grid(visible=False)

        # Save the plot.
        path = Path(plots_dir, f"training_iter_{iter_idx}.pdf")
        fig.savefig(path)
        plt.close(fig)
        logger.log(f"{exp_id}: {path.as_posix()!r}")


def single_eval(exp_dir, exp_id):
    """Creates a single plot for the given experiment for evaluation across
    multiple scenarios.

    Reads data from the metrics.json file."""
    metrics = utils.json_to_dict(Path(exp_dir, "metrics.json"))

    # A figure with six sub-plots. Each row is for a metric (Total Reward, Total
    # Congested Steps, and Total Rejected Requests), with two columns: one for
    # the average total (across multiple evaluation episodes) and one for the
    # standard deviation.
    fig = plt.figure(figsize=(10, 15), dpi=600, layout="constrained")
    fig.suptitle(f"Evaluation of {exp_id} (averages across eval. episodes)")
    axs = fig.subplots(ncols=2, nrows=3)

    # Fixed scenarios to be placed on the x-axis.
    scenarios = ["Scenario 1", "Scenario 2", "Scenario 3"]

    # Data retrieved from the metrics.json file. Each list has a length of three
    # (the number of scenarios).
    reward_total_mean = []
    reward_total_std = []
    congested_total_mean = []
    congested_total_std = []
    rejected_reqs_total_mean = []
    rejected_reqs_total_std = []
    for scenario in TrafficManagementEnv.get_scenarios():
        reward_total_mean.append(metrics["scenarios"][scenario]["reward_total_mean"])
        reward_total_std.append(metrics["scenarios"][scenario]["reward_total_std"])
        congested_total_mean.append(metrics["scenarios"][scenario]["congested_total_mean"])
        congested_total_std.append(metrics["scenarios"][scenario]["congested_total_std"])
        rejected_reqs_total_mean.append(metrics["scenarios"][scenario]["rejected_reqs_total_mean"])
        rejected_reqs_total_std.append(metrics["scenarios"][scenario]["rejected_reqs_total_std"])

    axs[0, 0].bar(scenarios, reward_total_mean)
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].set_title("Average total reward")

    axs[0, 1].bar(scenarios, reward_total_std)
    axs[0, 1].set_ylabel("Reward")
    axs[0, 1].set_title("SD total reward")

    axs[1, 0].bar(scenarios, congested_total_mean, color="g")
    axs[1, 0].set_ylabel("Steps")
    axs[1, 0].set_title("Average total congested steps")

    axs[1, 1].bar(scenarios, congested_total_std, color="g")
    axs[1, 1].set_ylabel("Steps")
    axs[1, 1].set_title("SD total congested steps")

    axs[2, 0].bar(scenarios, rejected_reqs_total_mean, color="r")
    axs[2, 0].set_ylabel("Requests")
    axs[2, 0].set_title("Average total rejected requests")

    axs[2, 1].bar(scenarios, rejected_reqs_total_std, color="r")
    axs[2, 1].set_ylabel("Requests")
    axs[2, 1].set_title("SD total rejected requests")

    # Common settings for all plots.
    for ax in axs.flat:
        ax.set_xlabel("Scenarios evaluated")

        ax.grid(which="both")
        ax.set_axisbelow(True)  # Place the grid behind the lines and bars.

    path = Path(exp_dir, "plots", "evaluation.pdf")
    fig.savefig(path)
    plt.close(fig)
    logger.log(f"{exp_id}: {path.as_posix()!r}")


def make_plots_single_experiment(exp_dir, exp_id):
    """Makes plots related to a single experiment. The plots are stored in the
    'plots' directory within the experiment directory."""
    logger.log(f"Making plots for experiment ID {exp_id}")

    # Make sure the directory is created.
    Path(exp_dir, "plots").mkdir(parents=True, exist_ok=True)

    try:
        make_training_plot(exp_dir, exp_id)
    except Exception:
        logger.err(f"Failed to make plots for summary of training of experiment {exp_id!r}: {traceback.format_exc()}")

    try:
        single_exp_iter(exp_dir, exp_id)
    except Exception:
        logger.err(f"Failed to make plots for single iterations during training of experiment {exp_id!r}: {traceback.format_exc()}")

    try:
        single_eval(exp_dir, exp_id)
    except Exception:
        logger.err(f"Failed to make plots for evaluation of experiment {exp_id!r}: {traceback.format_exc()}")


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


def aggregate_train_summary(exp_dirs, exp_id, res_dir):
    logger.log(f"Making summary training aggregate plots for {exp_id!r}")

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

    path = Path(res_dir, "plots", "training", f"{exp_id}.pdf")
    fig.savefig(path)
    plt.close(fig)
    logger.log(f"{exp_id}: {path.as_posix()!r}")


def aggregate_eval_summary(exp_dirs, exp_id, res_dir):
    pass


def aggregate_scenario(exp_dirs, exp_id, res_dir):
    logger.log(f"Making aggregate plots for {exp_id!r}")

    # Make sure the directory is created.
    Path(res_dir, "plots").mkdir(parents=True, exist_ok=True)

    try:
        aggregate_train_summary(exp_dirs, exp_id, res_dir)
    except Exception:
        logger.err(f"Failed to make plots for training summary for aggregate experiment {exp_id!r}: {traceback.format_exc()}")

    try:
        aggregate_eval_summary(exp_dirs, exp_id, res_dir)
    except Exception:
        logger.err(f"Failed to make plots for evaluation summary for aggregate experiment {exp_id!r}: {traceback.format_exc()}")


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

                    task = executor.submit(make_plots_single_experiment, exp_directory, exp["id"])
                    tasks.append(task)

                    all_done = True

                # When making aggregate plots, all sub-experiments must be
                # run.
                exp_id = f"{algo}:{params}:{scenario}"
                if not all_done:
                    continue
                task = executor.submit(aggregate_scenario, exp_dirs, exp_id, exp_dir)
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
