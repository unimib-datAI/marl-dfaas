# This Python script filters the Azure traces (only the invocations) by
# selecting only those whose http is triggered, excluding strange traces (by
# calculating some metrics), then reshaping the remaining traces to be suitable
# in the DFaaS environment. At each stage, the intermediate traces are saved to
# disk as CSV files.
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def _metrics(dataframe):
    # Calculate the sum, mean and std of the values, excluding the first 3
    # columns, because they are strings (owner, application and function
    # hashes).
    metrics = dataframe.iloc[:, 3:].agg(["sum", "mean", "std"], axis="columns")
    idx = 3
    for metric in metrics:
        dataframe.insert(idx, metric, metrics[metric])
        idx += 1


def filter_http(dataset_dir, overwrite):
    """Reads the function invocation datasets and extracts only the functions
    whose trigger is "http" and saves them in new datasets."""
    for src in dataset_dir.glob("invocations_per_function_md.anon.d*.csv"):
        dst = src.with_name(f"invocations_per_function_md.anon.http{src.suffixes[-2]}.csv")
        if dst.exists() and not overwrite:
            print(f"HTTP filter: file already exist, skipping {dst.as_posix()!r}")
            continue
        print(f"HTTP filter: filtering {src.as_posix()!r}")

        # Read the CSV file as Pandas DataFrame.
        data = pd.read_csv(src)

        # Get only rows with the http trigger.
        data = data[data["Trigger"] == "http"]

        # Drop the trigger column, because it contains only "http" values.
        data.drop(columns="Trigger", inplace=True)

        # Calculate and insert metrics columns.
        _metrics(data)

        # Save the new DataFrame as CSV.
        data.to_csv(dst, index=False)
        del data
        print(f"HTTP filter: filtered {dst.as_posix()!r}")


def scale(dataset_dir, overwrite):
    """Reads the filtered function call datasets and scales the values to
    respect the DFaaS environment max_steps and input_requests ranges, and saves
    them to new datasets."""
    for src in dataset_dir.glob("invocations_per_function_md.anon.http.d*.csv"):
        dst = src.with_name(f"invocations_per_function_md.anon.http.scaled{src.suffixes[-2]}.csv")
        if dst.exists() and not overwrite:
            print(f"scale: file already exist, skipping {dst.as_posix()!r}")
            continue
        print(f"scale: scaling {src.as_posix()!r}")

        # Read the CSV file as Pandas DataFrame.
        data = pd.read_csv(src)

        # Get the raw array, skipping the first non-integer columns.
        invocs = data.iloc[:, 6:].to_numpy(dtype=np.int32)

        # The original array has one row for each function, and each function
        # has 1440 columns (1 minute for 24 hours). The environment supports 288
        # steps by default (5 minutes for 24 hours). Therefore, I need to merge
        # every 5 steps from the original array into one value. This is done by
        # reshaping the original array and then summing on the correct axis.
        rows, elems = invocs.shape
        groups = elems // 5
        elems = 5
        invocs = invocs.reshape(rows, groups, elems).sum(axis=2)

        # Now the values need to be scaled in the range of the input_requests.
        # The following function is called for each row of the array and scales
        # the values.
        def interp(array):
            src_xp = (array.min(), array.max())
            dst_fp = (0, 150)  # TODO: Get dinamically.
            if (diff := np.diff(src_xp)) < 0:
                assert False, f"Values can't be negative: {src_xp = }"
            elif diff == 0:
                # This array is a flat line.
                array.fill(dst_fp[1])
                return array
            return np.interp(array, src_xp, dst_fp).astype(np.int32)

        invocs = np.apply_along_axis(interp, axis=1, arr=invocs)

        # Reconstruct the data frame with the first tree columns (owner,
        # application, and function hashes).
        data = pd.concat([data.iloc[:, :3], pd.DataFrame(invocs)], axis=1)

        # Calculate and insert metrics columns.
        _metrics(data)

        # Save the new DataFrame as CSV.
        data.to_csv(dst, index=False)
        del data
        print(f"scale: scaled {dst.as_posix()!r}")


def select(dataset_dir, overwrite):
    for src in dataset_dir.glob("invocations_per_function_md.anon.http.scaled.d*.csv"):
        dst = src.with_name(f"invocations_per_function_md.anon.http.scaled.selected{src.suffixes[-2]}.csv")
        if dst.exists() and not overwrite:
            print(f"select: file already exist, skipping {dst.as_posix()!r}")
            continue
        print(f"select: selecting {src.as_posix()!r}")

        # Read the CSV file as Pandas DataFrame.
        data = pd.read_csv(src)

        data = data[(data["mean"] >= 30) & (data["mean"] <= 130)]

        data = data[(data["std"] >= 10) & (data["std"] <= 60)]

        # Save the new DataFrame as CSV.
        data.to_csv(dst, index=False)
        del data
        print(f"select: selected {dst.as_posix()!r}")


def main(dataset_dir, overwrite):
    if not dataset_dir.exists():
        print(f"Not found {dataset_dir.as_posix()!r}")
        return

    filter_http(dataset_dir, "all" in overwrite or "http" in overwrite)

    scale(dataset_dir, "all" in overwrite or "scaled" in overwrite)

    select(dataset_dir, "all" in overwrite or "selected" in overwrite)


if __name__ == "__main__":
    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="filter_requests")

    parser.add_argument("--dataset-dir",
                        help="Directory whith the CSV files",
                        default="dataset/data")
    parser.add_argument("--overwrite-all",
                        help="Overwrite existing CSV files",
                        default=False, action="store_true")
    parser.add_argument("--overwrite-http",
                        help="Overwrite existing http filtered CSV files",
                        default=False, action="store_true")
    parser.add_argument("--overwrite-scaled",
                        help="Overwrite existing scaled CSV files",
                        default=False, action="store_true")
    parser.add_argument("--overwrite-selected",
                        help="Overwrite existing selected CSV files",
                        default=False, action="store_true")

    args = parser.parse_args()

    overwrite = []
    if args.overwrite_all:
        overwrite.append("all")
    if args.overwrite_http:
        overwrite.append("http")
    if args.overwrite_scaled:
        overwrite.append("scaled")
    if args.overwrite_selected:
        overwrite.append("selected")

    dataset_dir = Path(args.dataset_dir).resolve()

    main(dataset_dir, overwrite)
