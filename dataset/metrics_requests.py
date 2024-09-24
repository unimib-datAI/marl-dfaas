# This is a simple Python script that takes a list of CSV files, concatenates
# them into a single Pandas DataFrame, and calculates the metrics for the
# following columns in the dataset: 'sum', 'mean' and 'std'.
#
# This is useful to see if there are differences in the distribution between the
# pools used for training and evaluation, as well as the synthesized traces.
from pathlib import Path
import argparse

import pandas as pd


def main(csv_files):
    dataframes = []
    for csv_file in csv_files:
        print(f"selected {csv_file.as_posix()!r}")

        # Read the CSV file as Pandas DataFrame.
        dataframes.append(pd.read_csv(csv_file))

    # Join all different DataFrame(s) in a single one.
    dataframe = pd.concat(dataframes)

    for metric in ["sum", "mean", "std"]:
        result = dataframe.loc[:, metric].agg(["sum", "mean", "std"])

        # Show the result on the standard output. Maybe in the future can be
        # saved to a file.
        print(f"Metrics for {metric!r} column")
        print(f"  sum  = {result['sum']}")
        print(f"  mean = {result['mean']}")
        print(f"  std  = {result['std']}")


if __name__ == "__main__":
    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="metrics_requests")

    # A list of possible arguments, but must be at least of size one.
    parser.add_argument("csv_files",
                        help="One or more CSV files from which metrics are calculated",
                        type=Path,
                        nargs="+")

    args = parser.parse_args()

    main(args.csv_files)
