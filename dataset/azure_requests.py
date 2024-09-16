# This Python script generates a plot showing the invocations of a specified
# function of a given CSV file from the Azure Functions Trace 2019 dataset.
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def _get_data(data_file, function_hash):
    invocations = pd.read_csv(data_file)

    # Select the row with a specified function.
    fn_hash = function_hash
    fn = invocations[invocations["HashFunction"] == fn_hash]

    # Drop the first 3 columns (HashOwner, HashApp and HashFunction).
    fn = fn.iloc[:, 4:]

    data = {}
    data["invocations"] = fn.to_numpy(dtype=np.int32).flatten()
    data["function"] = fn_hash

    return data


def make(output_dir, data_file, function_hash):
    plots_dir = output_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Make the plot.
    fig = plt.figure(figsize=(15, 10), dpi=600, layout="constrained")
    ax = fig.subplots()

    data = _get_data(data_file, function_hash)

    ax.plot(data["invocations"])

    ax.set_title(f"Invocation (HTTP trigger) for function {data['function']}")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Invocations")

    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = plots_dir / f"requests_{function_hash}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="experiment_duration")

    parser.add_argument(dest="output_dir",
                        help="Where to save the plot")
    parser.add_argument(dest="data_file",
                        help="CSV file with function invocations")
    parser.add_argument(dest="function_hash",
                        help="Hash of the function to select in the file")

    args = parser.parse_args()

    make(Path(args.output_dir), Path(args.data_file), args.function_hash)
