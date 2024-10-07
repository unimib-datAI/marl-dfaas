# This Python script generates a plot of the synthetic input requests delivered
# to the agents using the type and seeds specified on the command line.
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
        case "sinusoidal":
            generate = dfaas_env._synthetic_sinusoidal_input_requests
        case "normal":
            generate = dfaas_env._synthetic_normal_input_requests
        case _:
            assert False, f"Unsupported synthetic {type = }"

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
    data["requests"] = generate(env.max_steps, env.agent_ids, limits, rng)
    data["agents"] = env.agent_ids
    data["limits"] = limits

    return data


def make(output_dir, type, seed):
    plots_dir = output_dir
    plots_dir.mkdir(exist_ok=True)

    data = _get_data(type, seed)

    # Make the plot.
    fig = plt.figure(figsize=(17, 7), dpi=600, layout="constrained")
    axs = fig.subplots(ncols=2)

    row = 0
    for agent in data["agents"]:
        axs[row].plot(data["requests"][agent])
        axs[row].set_title(f"Agente {agent!r}")

        bottom, top = data["limits"][agent]["min"], data["limits"][agent]["max"]
        axs[row].set_ylim(bottom=bottom, top=top)

        axs[row].set_xlabel("Passi")
        axs[row].xaxis.set_major_locator(ticker.MultipleLocator(50))

        axs[row].set_ylabel("Richieste")
        axs[row].yaxis.set_major_locator(ticker.MultipleLocator(10))

        axs[row].grid(axis="both")
        axs[row].set_axisbelow(True)  # By default the axis is over the content.

        row += 1

    # Save the plot.
    path = plots_dir / f"requests_{type}_{seed}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    epilog = """Note that the default max_steps, limits and agent IDs are fixed.
    If multiple seeds are given, for each seed a plot is produced."""

    # Create parser and parse command line arguments.
    parser = argparse.ArgumentParser(epilog=epilog)

    parser.add_argument(dest="output_dir", type=Path,
                        help="Where to save the plot")
    parser.add_argument(dest="type", choices=["sinusoidal", "normal"],
                        help="Type of syntetic requests to generate")
    parser.add_argument(dest="seed", nargs="+", type=int,
                        help="Seed used to generate synthetic requests")

    args = parser.parse_args()

    for seed in args.seed:
        make(args.output_dir.resolve(), args.type, seed)
