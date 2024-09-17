# This Python script generates a graph showing the duration times of different
# experiments run by changing the GPU enable/disable and the number of rollout
# workers.
#
# The raw data is manually collected and hardcoded into this script.
from pathlib import Path
import argparse

import matplotlib
import matplotlib.pyplot as plt


def make(output_dir):
    plots_dir = output_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Each key is a type of experiment ("gpu" and "cpu"). Each value is a list
    # of 9 entries for each (from 0 rollout workers to 8). Each item is the
    # number of minutes taken from the corresponding experiment.
    data = {}
    data["gpu"] = [139, 126, 101, 89, 85, 82, 80, 78, 77]
    data["cpu"] = [301, 227, 201, 188, 188, 181, 176, 178, 173]

    # Make the plot.
    fig = plt.figure(dpi=600, layout="constrained")
    ax = fig.subplots()

    ax.plot(data["gpu"], label="With GPU", color="g", linewidth=2, marker="o")
    ax.plot(data["cpu"], label="CPU only", color="b", linewidth=2, marker="o")

    title = "Experiment duration\n" \
            "Host 24 CPU, 64 GB RAM, Tesla T4 GPU\n" \
            "Training 200 iterations, 4000 step per iteration, 100 step per episode\n"

    ax.set_title(title)
    ax.set_xlabel("Rollout workers")
    ax.set_ylabel("Minutes")

    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.

    ax.legend()

    # Save the plot.
    path = plots_dir / "experiment_duration.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="experiment_duration")

    parser.add_argument(dest="output_dir",
                        help="Where to save the plot")

    args = parser.parse_args()

    make(Path(args.output_dir))
