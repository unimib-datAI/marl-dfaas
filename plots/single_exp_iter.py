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

from RL4CC.utilities.logger import Logger

logger = Logger(name="DFAAS-PLOTS", verbose=2)


def _get_cumulated_array(data, **kwargs):
    """Returns an ndarray with the cumulated sum passing kwargs to the clip
    method.

    Used to make stacked bar charta with negative and positive values."""
    # See: https://stackoverflow.com/a/38900035
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d


def make(exp_dir, exp_id):
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
        iter_idx = 1
        while (raw_iter := result.readline()) != "":
            # The first iteration is always saved.
            if iter_idx == 1 or iter_idx % iter_plots_each == 0:
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
        cumulated_data = _get_cumulated_array(reward_components, min=0)
        cumulated_data_neg = _get_cumulated_array(reward_components, max=0)

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


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser(prog="single_exp_iter")

    parser.add_argument(dest="experiment_directory",
                        help="Experiment directory")
    parser.add_argument(dest="experiment_id",
                        help="Experiment ID")

    args = parser.parse_args()

    make(Path(args.experiment_directory, args.experiment_id), args.experiment_id)
