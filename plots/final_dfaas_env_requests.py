# This Python script generates a plot of all types of input requests for on
# agent using the seeds specified on the command line.
#
# # Note: this script assumes that the experiment directories are located in
# "results/final".
from pathlib import Path
import sys
import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add the current directory (where Python is called) to sys.path. This is
# required to load modules in the project root directory (like dfaas_utils.py).
sys.path.append(os.getcwd())

import dfaas_env
import plot_utils  # noqa: F401 (disables matplotlib debug messages)


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
    for agent in env.agent_ids:
        limits[agent] = {
                "min": env.observation_space[agent]["input_requests"].low.item(),
                "max": env.observation_space[agent]["input_requests"].high.item()
                }

    # Create the RNG used to generate input requests.
    rng = np.random.default_rng(seed=seed)

    data = {}
    if type == "real":
        # Select only the input requests, not the hashes.
        data["requests"] = generate(env.max_steps, env.agent_ids, limits, rng, False)[0]
    else:
        data["requests"] = generate(env.max_steps, env.agent_ids, limits, rng)
    data["agents"] = env.agent_ids
    data["limits"] = limits

    return data


def make(output_dir, type, seed):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    data = _get_data(type, seed)

    # Make the plot.
    fig = plt.figure(figsize=(10, 7), dpi=600, layout="constrained")
    ax = fig.subplots()

    agent = "node_0"
    ax.plot(data["requests"][agent])

    bottom, top = data["limits"][agent]["min"], data["limits"][agent]["max"]
    ax.set_ylim(bottom=bottom, top=top)

    ax.set_xlabel("Passi", fontsize="large")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

    ax.set_ylabel("Richieste", fontsize="large")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = plots_dir / f"requests_{type}_{seed}_single_agent.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    # Default size is too small.
    font = {"family": "serif", "size": 20}
    matplotlib.rc('font', **font)

    epilog = """This script assumes that the experiment directories are stored
    under the "results/final" directories with the evaluations. Produces a plot
    for each type of scenario and seed."""

    # Create parser and parse command line arguments.
    parser = argparse.ArgumentParser(epilog=epilog)

    parser.add_argument(dest="seed", nargs="+", type=int,
                        help="Seed used to generate synthetic requests")

    args = parser.parse_args()

    out = Path("results/final")
    types = ["real", "sinusoidal", "normal"]

    for seed in args.seed:
        for type in types:
            make(out, type, seed)
