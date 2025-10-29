from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import numpy as np
import os


def count_offloaded_processing(
    detailed_offloading: pd.DataFrame, Nn: int, Nf: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    offloaded_processing = pd.DataFrame()
    detailed_offloaded_processing = pd.DataFrame()
    for n in range(Nn):
        for f in range(Nf):
            offloaded_processing[f"n{n}_f{f}_accepted"] = detailed_offloading.loc[
                :, detailed_offloading.columns.str.endswith(f"_f{f}_n{n}")
            ].sum(axis="columns")
    return offloaded_processing, detailed_offloaded_processing


def decode_solution(
    x: np.array,
    y: np.array,
    z: np.array,
    zeta: np.array,
    complete_solution: dict,
) -> dict:
    Nn, Nf = x.shape
    # local processing
    complete_solution["local_processing"] = update_2d_variables(x, complete_solution["local_processing"])
    # offloading
    (complete_solution["offloading"], complete_solution["detailed_offloading"]) = update_3d_variables(
        y, complete_solution["offloading"], complete_solution["detailed_offloading"]
    )
    # rejections
    complete_solution["rejections"] = update_2d_variables(z, complete_solution["rejections"])
    # offloaded processing
    (complete_solution["offloaded_processing"], complete_solution["detailed_offloaded_processing"]) = (
        count_offloaded_processing(complete_solution["detailed_offloading"], Nn, Nf)
    )
    # processing reject
    complete_solution["processing_reject"] = update_2d_variables(zeta, complete_solution["processing_reject"])
    return complete_solution


def extract_solution(data: dict, solution: dict) -> Tuple[np.array, np.array, np.array, float]:
    Nn = data[None]["Nn"][None]
    Nf = data[None]["Nf"][None]
    x = np.zeros((Nn, Nf))
    y = np.zeros((Nn, Nn, Nf))
    z = np.zeros((Nn, Nf))
    # -- local processing
    if "x" in solution:
        x = np.array(solution["x"], dtype=int).reshape((Nn, Nf))
    # -- offloading
    if "y" in solution:
        y = np.array(solution["y"], dtype=int).reshape((Nn, -1, Nf))
    # -- rejections
    if "z" in solution:
        z = np.array(solution["z"], dtype=int).reshape((Nn, Nf))
    # -- processing rejects
    if "processing_reject" in solution:
        zeta = np.array(solution["processing_reject"], dtype=float).reshape((Nn, Nf))
    return x, y, z, zeta, solution["obj"]


def init_complete_solution():
    return {
        "local_processing": pd.DataFrame(),
        "offloading": pd.DataFrame(),
        "detailed_offloading": pd.DataFrame(),
        "rejections": pd.DataFrame(),
        "offloaded_processing": pd.DataFrame(),
        "detailed_offloaded_processing": pd.DataFrame(),
        "processing_reject": pd.DataFrame(),
    }


def join_complete_solution(complete_solution: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    solution = (
        complete_solution["local_processing"]
        .join(complete_solution["offloading"], lsuffix="_loc", rsuffix="_fwd")
        .join(complete_solution["rejections"])
    )
    offloaded = complete_solution["offloaded_processing"]
    detailed_fwd_solution = complete_solution["detailed_offloading"].join(
        complete_solution["detailed_offloaded_processing"], lsuffix="_tot", rsuffix="_accepted"
    )
    return solution, offloaded, detailed_fwd_solution


def plot_history(
    input_requests_traces: dict,
    solution: pd.DataFrame,
    offloaded: pd.DataFrame,
    obj_values: list,
    plot_filename: str = None,
):
    Nn = len(input_requests_traces)
    tmax = len(solution)
    _, axs = plt.subplots(nrows=Nn, ncols=3, figsize=(36, 4 * Nn))
    # loop over nodes
    for agent in range(Nn):
        # -- input workload
        axs[agent, 0].plot(range(tmax), input_requests_traces[f"node_{agent}"], ".-", color="k")
        # -- requests management
        solution.loc[:, solution.columns.str.startswith(f"n{agent}_f0")].plot.bar(stacked=True, ax=axs[agent, 0], rot=0)
        # -- received offloads
        if len(offloaded) > 0:
            offloaded.loc[:, offloaded.columns.str.startswith(f"n{agent}_f0")].plot.bar(
                stacked=True, ax=axs[agent, 1], rot=0
            )
        # -- axis properties
        axs[agent, 0].grid(axis="y")
        axs[agent, 1].grid(axis="y")
    # objective function value
    axs[0, -1].plot(range(len(obj_values)), obj_values, ".-", linewidth=2, color=mcolors.TABLEAU_COLORS["tab:red"])
    axs[0, -1].set_ylabel("Total number of rejections")
    axs[0, -1].grid(axis="y")
    # -- rate of total rejections
    total_reject_rate = [
        obj_values[t] / sum([input_requests_traces[a][t] for a in input_requests_traces]) * 100 for t in range(tmax)
    ]
    axs[1, -1].plot(
        range(len(obj_values)), total_reject_rate, ".-", linewidth=2, color=mcolors.TABLEAU_COLORS["tab:red"]
    )
    axs[1, -1].axhline(
        y=sum(total_reject_rate) / len(total_reject_rate),
        linewidth=2,
        linestyle="dashed",
        color=mcolors.CSS4_COLORS["darkred"],
    )
    axs[1, -1].set_ylabel("Total rejections rate [%]")
    axs[1, -1].grid(axis="y")
    if plot_filename is not None:
        plt.savefig(plot_filename, dpi=300, format="png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_solution(
    solution: pd.DataFrame,
    offloaded: pd.DataFrame,
    detailed_fwd_solution: pd.DataFrame,
    processing_reject: pd.DataFrame,
    obj_values: list,
    total_reject_rate: list,
    model_name: str,
    solution_folder: str,
):
    solution.to_csv(os.path.join(solution_folder, f"{model_name}_solution.csv"), index=False)
    offloaded.to_csv(os.path.join(solution_folder, f"{model_name}_offloaded.csv"), index=False)
    detailed_fwd_solution.to_csv(os.path.join(solution_folder, f"{model_name}_detailed_fwd_solution.csv"), index=False)
    processing_reject.to_csv(os.path.join(solution_folder, f"{model_name}_processing_reject.csv"), index=False)
    pd.DataFrame({f"{model_name}_obj": obj_values, f"{model_name}_rej_rate": total_reject_rate}).to_csv(
        os.path.join(solution_folder, "obj.csv"), index=False
    )


def update_2d_variables(var: np.array, res: pd.DataFrame) -> pd.DataFrame:
    Nn, Nf = var.shape
    df = var.reshape(1, -1).tolist()
    cols = [f"n{n}_f{f}" for n in range(Nn) for f in range(Nf)]
    res = pd.concat([res, pd.DataFrame(df, columns=cols)], ignore_index=True)
    return res


def update_3d_variables(y: np.array, offloading: pd.DataFrame, detailed_offloading: pd.DataFrame) -> pd.DataFrame:
    Nn, _, Nf = y.shape
    df = {f"n{n}_f{f}": [] for n in range(Nn) for f in range(Nf)}
    detailed_df = {f"n{n1}_f{f}_n{n2}": [] for n1 in range(Nn) for f in range(Nf) for n2 in range(Nn) if n2 != n1}
    for f in range(Nf):
        for n1 in range(Nn):
            df[f"n{n1}_f{f}"].append(y[n1, :, f].sum())
            for n2 in range(Nn):
                if n1 != n2:
                    detailed_df[f"n{n1}_f{f}_n{n2}"].append(y[n1, n2, f])
    offloading = pd.concat([offloading, pd.DataFrame(df)], ignore_index=True)
    detailed_offloading = pd.concat([detailed_offloading, pd.DataFrame(detailed_df)], ignore_index=True)
    return offloading, detailed_offloading
