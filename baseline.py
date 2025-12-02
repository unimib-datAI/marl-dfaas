"""This module runs several DFaaS episodes using two baseline strategies:

1. All-local action: each node always attempts to process all incoming traffic
   locally.
2. Random action: each node selects its actions randomly.

The script saves the resulting data as CSV files and produces a multi-plot
figure in PDF format. Outputs are stored under: "baseline_local" for the
all-local action and "baseline_random" for the random action.

Run the script without any command-line arguments to execute a predefined set of
episodes using hardcoded random seeds (these can be modified in the main()
function).
"""

import argparse
from pathlib import Path
from typing import Tuple
import logging

# By default Ray uses DEBUG level, but I prefer the ERROR level and this must be
# set manually!
logging.basicConfig(level=logging.ERROR)

import pandas as pd
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, AutoMinorLocator, PercentFormatter

from dfaas_env_config import DFaaSConfig
from dfaas_env import DFaaS
from dfaas_utils import yaml_to_dict


def make_plots(df_agents: pd.DataFrame, df_all_data: pd.DataFrame) -> Figure:
    """Make a figure with multiple subplots, one for each agent plus a final one
    for cumulated reject rate over all agents."""
    # Extract the agents from the "reward_{agent}" columns.
    agents = [col.split("reward_")[1] for col in df_agents.columns if col.startswith("reward_")]

    steps = df_agents.shape[0]

    # We have a row for each agent with a single plot, plus a final row for the
    # totals.
    fig, axes = plt.subplots(nrows=len(agents) + 1, figsize=(10, 5 * (len(agents) + 1)), layout="constrained")

    # Loop over agents.
    for i, agent in enumerate(agents):
        ax = axes[i]

        input_rate = df_agents[f"input_rate_{agent}"]
        reject_rate = df_agents[f"reject_rate_{agent}"]
        local_rate = input_rate - reject_rate

        # Input rate.
        ax.plot(input_rate, ".-", label="Input", color="k")

        # Local rate (not rejected).
        ax.bar(range(steps), local_rate, label="Local (not rej.)")

        # Reject rate.
        ax.bar(range(steps), reject_rate, bottom=local_rate, label="Reject")

        ax.set_title(f"Agent {agent!r}")
        ax.set_ylabel("Rate")
        ax.set_xlabel("Step")

        # Ensure the start and last steps are always shown.
        major_locators = list(range(0, 288, 50)) + [288]
        ax.xaxis.set_major_locator(FixedLocator(major_locators))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(0, steps)

        ax.legend()
        ax.grid(axis="both", which="both")
        ax.set_axisbelow(True)

    # Totals.
    ax = axes[-1]
    input_rate = df_all_data["input_rate"]
    reject_rate = df_all_data["reject_rate"]

    # Show absolute reject rate.
    (line1,) = ax.plot(reject_rate, ".-", label="Reject (abs)", color="r")

    # Show percentual reject rate with average % as horizontal line.
    ax2 = ax.twinx()
    reject_rate_perc = reject_rate / input_rate
    line2 = ax2.axhline(y=reject_rate_perc.mean(), color="r", linestyle="--", label="Average reject (%)")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_ylabel("Rate (%)")

    ax.set_title("Cumulative reject rate over all agents")
    ax.set_ylabel("Rate (abs)")
    ax.set_xlabel("Step")

    # Ensure the start and last steps are always shown.
    major_locators = list(range(0, 288, 50)) + [288]
    ax.xaxis.set_major_locator(FixedLocator(major_locators))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim(0, steps)

    # Merge manually legends of both ax and ax2.
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)

    ax.grid(axis="both", which="both")
    ax.set_axisbelow(True)

    return fig


def create_dataframes(info: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the info dictionary from the DFaaS environment to Pandas
    DataFrames.

    Return two dataframes: one for agent-specific metric and one for global
    metrics."""
    agents = list(info.keys())

    agents_data = {}
    for agent in agents:
        agents_data[f"input_rate_{agent}"] = info[agent]["observation_input_rate"]
        agents_data[f"reward_{agent}"] = info[agent]["reward"]

        # Reject rate is the sum of different reject metrics.
        action_reject = np.array(info[agent]["action_reject"])
        forward_reject = np.array(info[agent]["forward_reject_rate"])
        local_reject = np.array(info[agent]["incoming_rate_local_reject"])

        agents_data[f"reject_rate_{agent}"] = action_reject + forward_reject + local_reject

        # Latency
        agents_data[f"response_time_avg_local_{agent}"] = np.array(info[agent]["response_time_avg_local"])
        agents_data[f"exec_time_avg_forwarded_{agent}"] = np.array(info[agent]["response_time_avg_forwarded"])
        agents_data[f"network_forward_delay_avg_{agent}"] = np.array(info[agent]["network_forward_delay_avg"])
        agents_data[f"network_return_delay_avg_{agent}"] = np.array(info[agent]["network_return_delay_avg"])
        agents_data[f"response_time_avg_forwarded_{agent}"] = (
            agents_data[f"exec_time_avg_forwarded_{agent}"] + 
                agents_data[f"network_forward_delay_avg_{agent}"] +
                    agents_data[f"network_return_delay_avg_{agent}"]
        )

    df_agents = pd.DataFrame(agents_data, columns=sorted(agents_data))

    # Cumulate metrics for all agent under the "all" agent.
    all_data = {}
    all_data["input_rate"] = df_agents.filter(regex="^input_rate_").sum(axis=1)
    all_data["reject_rate"] = df_agents.filter(regex="^reject_rate_").sum(axis=1)
    all_data["reward"] = df_agents.filter(regex="^reward").sum(axis=1)
    all_data["response_time_avg_local"] = df_agents.filter(regex="^response_time_avg_local_").sum(axis=1) / len(agents)
    all_data["exec_time_avg_forwarded"] = df_agents.filter(regex="^exec_time_avg_forwarded_").sum(axis=1) / len(agents)
    all_data["response_time_avg_forwarded"] = df_agents.filter(regex="^response_time_avg_forwarded_").sum(axis=1) / len(agents)
    all_data["network_forward_delay_avg"] = df_agents.filter(regex="^network_forward_delay_avg_").sum(axis=1) / len(agents)
    all_data["network_return_delay_avg"] = df_agents.filter(regex="^network_return_delay_avg_").sum(axis=1) / len(agents)

    df_all_data = pd.DataFrame(all_data)

    return df_agents, df_all_data


def run_episode(
    env: DFaaS,
    exp_iter: int,
    base_results_folder: str | Path,
    seed: int | None,
    agent_policy: str = "local",
):
    # By default, seed=seed sets the master seed, which is then used to
    # generate a sequence of seeds, one for each episode. However, in this
    # case, we want to directly control the seed for a single episode, so we
    # override it individually for each one.
    _ = env.reset(options={"override_seed": seed})

    action = {}
    if agent_policy == "local":
        for agent in env.agents:
            # Force all local processing by setting only the first value.
            agent_action = np.zeros(shape=env.action_space[agent].shape)
            agent_action[0] = 1.0
            action[agent] = agent_action

    for t in trange(env.max_steps, desc=f"Episode {exp_iter} seed {env.seed} ({agent_policy})"):
        if agent_policy == "random":
            action = {agent: env.action_space[agent].sample() for agent in env.agents}

        env.step(action)

    # Create the result dataframes from the environment's info dictionary. See
    # the DFaaS agent for more information.
    df_agents, df_all_data = create_dataframes(env.info)

    results_folder = Path(base_results_folder) / f"{exp_iter}_seed_{env.seed}"
    results_folder.mkdir(parents=True, exist_ok=True)

    df_agents.to_csv(results_folder / "results_by_agent.csv", index=False)
    df_all_data.to_csv(results_folder / "results_all.csv", index=False)

    fig = make_plots(df_agents, df_all_data)
    fig.suptitle(f"Episode with seed {env.seed} ({agent_policy})")
    # Trim extra whitespace with bbox_inches=thight.
    fig.savefig(results_folder / "plots.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(
    env_config: dict,
    seeds: list[int],
    base_results_folder: str,
    agent_policy: str = "local",
):
    print(f"Agent policy: {agent_policy}")
    print(f"Environment config.: {env_config}\n")

    # Initialize environment.
    env = DFaaSConfig.from_dict(env_config).build()

    # Dummy seed, we need to call reset at least once to set up the env.
    env.reset(seed=seeds[0])

    for exp in trange(len(seeds), desc=f"Running episodes ({agent_policy})"):
        run_episode(
            env,
            exp,
            base_results_folder,
            seeds[exp],
            agent_policy=agent_policy,
        )
    print()


def main():
    """Main entry point of the script."""
    parser = argparse.ArgumentParser(
        description="Run baseline experiments for DFaaS environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default="configs/env/default.yaml",
        help="Path to the environment configuration file.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            1114399290,
            586248983,
            1296339178,
            980462265,
            2807237418,
            3153669498,
            1573623524,
            1657272726,
            409898216,
            2730495449,
        ],
        help="List of seeds for the experiments.",
    )
    parser.add_argument(
        "--results-folder",
        type=Path,
        default="results/baseline",
        help="Folder to store results for 'local' and 'random' policies",
    )
    args = parser.parse_args()

    env_config = yaml_to_dict(args.env_config)

    # First run: local actions.
    run(
        env_config,
        args.seeds,
        args.results_folder / "local",
        agent_policy="local",
    )

    # Second run: random actions.
    run(
        env_config,
        args.seeds,
        args.results_folder / "random",
        agent_policy="random",
    )


if __name__ == "__main__":
    main()
