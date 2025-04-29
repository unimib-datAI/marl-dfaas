# This Python script generates a plot of all types of input requests for on
# agent using the seeds specified on the command line.
from pathlib import Path
import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add the current directory (where Python is called) to sys.path. This is
# required to load modules in the project root directory (like dfaas_utils.py).
sys.path.append(os.getcwd())

import dfaas_env
import plot_utils


def _get_data(type, seed):
    match type:
        case "real":
            generate = dfaas_env._real_input_requests
        case "sinusoidal":
            generate = dfaas_env._synthetic_sinusoidal_input_requests
        case "normal":
            generate = dfaas_env._synthetic_normal_input_requests
        case _:
            assert False, f"Unsupported requests {type = }"

    # Set function arguments to fixed generic values.
    env = dfaas_env.DFaaS()
    limits = {}
    for agent in env.agents:
        limits[agent] = {
            "min": env.observation_space[agent]["input_requests"].low.item(),
            "max": env.observation_space[agent]["input_requests"].high.item(),
        }

    # Create the RNG used to generate input requests.
    rng = np.random.default_rng(seed=seed)

    data = {}
    if type == "real":
        # Select only the input requests, not the hashes.
        data["requests"] = generate(env.max_steps, env.agents, limits, rng, False)[0]
    else:
        data["requests"] = generate(env.max_steps, env.agents, limits, rng)
    data["agents"] = env.agents
    data["limits"] = limits

    return data


def make(output_dir, type, seed):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    data = _get_data(type, seed)

    # Make the plot.
    fig = plt.figure(figsize=(10, 7), dpi=600, layout="constrained")
    ax = fig.subplots()

    agent = "node_0"
    ax.plot(data["requests"][agent])

    bottom, top = data["limits"][agent]["min"], data["limits"][agent]["max"]
    ax.set_ylim(bottom=bottom, top=top)

    ax.set_xlabel("Steps", fontsize="large")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

    ax.set_ylabel("Requests", fontsize="large")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = plots_dir / f"input_requests_{type}_{seed}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    types = ["real", "sinusoidal", "normal"]

    # Create parser and parse command line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generic_plots"),
        help="A directory path here to save the generated plots",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Seeds used to generate (synthetic) or select (real) requests",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        type=str,
        choices=types,
        default=types,
        help="Which type of input requests to consider",
    )

    args = parser.parse_args()

    for seed in args.seeds:
        for type in args.types:
            make(args.out, type, seed)
