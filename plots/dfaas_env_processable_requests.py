# This Python script generates a plot of the given input requests type and seed
# showing the total input requests divided by the processable and non
# processable ones. The script accepts multiple seeds (multiple plots).
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

    match type:
        case "synthetic-sinusoidal":
            input_reqs = dfaas_env._synthetic_sinusoidal_input_requests(env.max_steps,
                                                                        env.agent_ids,
                                                                        limits,
                                                                        rng)
        case "synthetic-normal":
            input_reqs = dfaas_env._synthetic_normal_input_requests(env.max_steps,
                                                                    env.agent_ids,
                                                                    limits,
                                                                    rng)
        case "real":
            input_reqs = dfaas_env._real_input_requests(env.max_steps,
                                                        env.agent_ids,
                                                        limits,
                                                        rng,
                                                        True)
        case _:
            assert False, f"Unsupported synthetic {type = }"

    # Only environments with two agents are supported.
    assert env.agents == 2, "Only environments with two agents are supported for this plot"

    total_input_reqs = np.zeros(env.max_steps, dtype=np.int32)
    for agent in env.agent_ids:
        total_input_reqs += input_reqs[agent]

    # Only environments with the same queue capacity are supported for this
    # script.
    queue_capacity = np.unique(list(env.queue_capacity_max.values()))
    assert queue_capacity.size == 1, "Environments with different queue capacity size for the agents are not supported"
    total_capacity = queue_capacity.item() * env.agents

    data = {}
    data["env"] = env
    data["requests"] = total_input_reqs
    data["capacity"] = np.repeat(total_capacity, env.max_steps)

    return data


def make(output_dir, type, seed):
    plots_dir = output_dir
    plots_dir.mkdir(exist_ok=True)

    data = _get_data(type, seed)

    # Make the plot.
    fig = plt.figure(figsize=(9, 5), dpi=600, layout="constrained")
    ax = fig.subplots()

    x = np.arange(0, data["env"].max_steps)

    ax.plot(x, data["requests"], label="Richieste in ingresso")
    ax.plot(x, data["capacity"], label="Capacità totale")
    ax.fill_between(x, np.clip(data["requests"], a_min=None, a_max=data["capacity"]),
                    label="Richieste processabili", alpha=.7, color="g")
    ax.fill_between(x, data["capacity"], data["requests"],
                    where=data["requests"] > data["capacity"], interpolate=True,
                    label="Richieste non processabili", alpha=.7, color="r")

    ax.set_xlabel("Passi")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

    ax.set_ylabel("Richieste totali")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

    ax.legend()

    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = plots_dir / f"processable_requests_{type}_{seed}.pdf"
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
    parser.add_argument(dest="type", choices=["synthetic-sinusoidal",
                                              "synthetic-normal",
                                              "real"],
                        help="Type of input requests to generate")
    parser.add_argument(dest="seed", nargs="+", type=int,
                        help="Seed used to generate/select input requests")

    args = parser.parse_args()

    for seed in args.seed:
        make(args.output_dir.resolve(), args.type, seed)
