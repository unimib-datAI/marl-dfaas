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

from dfaas_env import DFaaS as DFaaSEnv
from dfaas_utils import toml_to_dict


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
        agents_data[f"reject_rate_{agent}"] = info[agent]["incoming_rate_local_reject"]

    df_agents = pd.DataFrame(agents_data, columns=sorted(agents_data))

    # Cumulate metrics for all agent under the "all" agent.
    all_data = {}
    all_data["input_rate"] = df_agents.filter(regex="^input_rate_").sum(axis=1)
    all_data["reject_rate"] = df_agents.filter(regex="^reject_rate_").sum(axis=1)
    all_data["reward"] = df_agents.filter(regex="^reward").sum(axis=1)

    df_all_data = pd.DataFrame(all_data)

    return df_agents, df_all_data


def run_episode(
    env: DFaaSEnv,
    exp_iter: int,
    warm_service_time: float,
    cold_service_time: float,
    idle_time_before_kill: int,
    maximum_concurrency: int,
    base_results_folder: str | Path,
    seed: int | None,
):
    # By default, seed=seed sets the master seed, which is then used to
    # generate a sequence of seeds, one for each episode. However, in this
    # case, we want to directly control the seed for a single episode, so we
    # override it individually for each one.
    _ = env.reset(options={"override_seed": seed})

    action = {}
    for agent in env.agents:
        # Force all local processing by setting only the first value.
        agent_action = np.zeros(shape=env.action_space[agent].shape)
        agent_action[0] = 1.0

        action[agent] = agent_action

    for t in trange(env.max_steps, desc=f"Episode {exp_iter} seed {env.seed}"):
        env.step(action)

    # Create the result dataframes from the environment's info dictionary. See
    # the DFaaS agent for more information.
    df_agents, df_all_data = create_dataframes(env.info)

    results_folder = Path(base_results_folder) / f"{exp_iter}_seed_{env.seed}"
    results_folder.mkdir(parents=True, exist_ok=True)

    df_agents.to_csv(results_folder / "results_by_agent.csv", index=False)
    df_agents.to_csv(results_folder / "results_all.csv", index=False)

    fig = make_plots(df_agents, df_all_data)
    fig.suptitle(f"Episode with seed {env.seed}")
    # Trim extra whitespace with bbox_inches=thight.
    fig.savefig(results_folder / "plots.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(
    env_config_file: str,
    seeds: list[int],
    warm_service_time: float,
    cold_service_time: float,
    idle_time_before_kill: int,
    maximum_concurrency: int,
    base_results_folder: str,
):
    env_config = toml_to_dict(env_config_file)

    print(f"Environment config.: {env_config}\n")

    # Initialize environment.
    env = DFaaSEnv(config=env_config)

    # Dummy seed, we need to call reset at least once to set up the env.
    env.reset(seed=seeds[0])

    for exp in trange(len(seeds), desc="Running episodes"):
        run_episode(
            env,
            exp,
            warm_service_time,
            cold_service_time,
            idle_time_before_kill,
            maximum_concurrency,
            base_results_folder,
            seeds[exp],
        )


def main():
    env_config_file = "configs/env/five_agents.toml"

    seeds = [
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
    ]

    # Values taken from dfaas_env.py.
    warm_service_time = 15
    cold_service_time = 30
    idle_time_before_kill = 600
    maximum_concurrency = 1000
    base_results_folder = "baseline_local"

    run(
        env_config_file,
        seeds,
        warm_service_time,
        cold_service_time,
        idle_time_before_kill,
        maximum_concurrency,
        base_results_folder,
    )


if __name__ == "__main__":
    main()
