# This Python script generates a plot for each environment type showing the
# average total reward for all evaluations.
#
# Note: this script assumes that the experiment directories are located in
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
import dfaas_utils
import plot_utils


def _get_exp_env_type(exp_name):
    if "BASE" in exp_name:
        return "BASE"
    if "ASYM" in exp_name:
        return "ASYM"

    return "SYM"


def _get_exp_train_scenario(exp_name):
    if "real" in exp_name:
        return "R"  # Real scenario.
    if "synt_norm" in exp_name:
        return "S"  # Sinthetic sinusoidal scenario.

    return "G"  # Sinthetic gaussian scenario.


def _avg_reward_exp(exp_dir):
    evals_dir = list(exp_dir.glob("evaluation_*"))

    reward = reward_min = reward_max = 0
    for eval_dir in evals_dir:
        # Read data from the given evaluation directory
        eval = dfaas_utils.parse_result_file(eval_dir / "evaluation.json")
        eval = eval[0]["evaluation"]

        reward_min += eval["episode_reward_min"]
        reward_max += eval["episode_reward_max"]
        reward += eval["episode_reward_mean"]

    reward_avg_min = reward_min / len(evals_dir)
    reward_avg_max = reward_max / len(evals_dir)
    reward_avg = reward / len(evals_dir)

    # Because matplotlib wants the errors (min, max) as offsets, I need to
    # process the values.
    assert reward_avg_min <= reward_avg and reward_avg_max >= reward_avg
    reward_avg_min = reward_avg - reward_avg_min
    reward_avg_max = reward_avg_max - reward_avg

    return reward_avg, reward_avg_min, reward_avg_max


def _get_data(exps):
    data = {}

    data = {}
    for algo in ["PPO", "PPO-CC"]:
        data[algo] = {}
        data[algo]["values"] = []
        data[algo]["min"] = []
        data[algo]["max"] = []
        data[algo]["labels"] = []
        for exp_dir in exps:
            if algo == "PPO" and "cc" in exp_dir.name:
                continue
            elif algo == "PPO-CC" and "cc" not in exp_dir.name:
                continue

            scenario = _get_exp_train_scenario(exp_dir.name)

            reward_avg, reward_min_avg, reward_max_avg = _avg_reward_exp(exp_dir)

            data[algo]["values"].append(reward_avg)
            data[algo]["min"].append(reward_min_avg)
            data[algo]["max"].append(reward_max_avg)
            data[algo]["labels"].append(scenario)

    return data


def make(env_type, exps, out):
    plots_dir = out
    plots_dir.mkdir(exist_ok=True)

    data = _get_data(exps)
    env = dfaas_env.DFaaS()  # Dummy env.

    fig = plt.figure(figsize=(7, 5), dpi=600, layout="constrained")
    ax = fig.subplots()

    # Limits for the y axis, both for total and single step.
    # Note the reward for each step is in [0, 1].
    _, top = env.reward_range
    max_reward = top * env.max_steps * env.agents
    bottom = 250  # Heuristically selected.
    ax.set_ylim(bottom=250, top=max_reward+10)

    # Set bar width and colors.
    bar_width = 0.30
    ppo_bar_color = '#99ccff'
    ppo_cc_bar_color = '#ff9999'
    max_reward_line_color = "#7a49a5"
    capsize = 10  # Length of the error horizontal bar.

    x_ticks = np.arange(len(data["PPO"]["values"]))
    ppo_bars = ax.bar(x_ticks - bar_width/2, data["PPO"]["values"], bar_width,
                      yerr=[data["PPO"]["min"], data["PPO"]["max"]],
                      capsize=capsize, color=ppo_bar_color)
    ppo_cc_bars = ax.bar(x_ticks + bar_width/2, data["PPO-CC"]["values"], bar_width,
                         yerr=[data["PPO-CC"]["min"], data["PPO-CC"]["max"]],
                         capsize=capsize, color=ppo_cc_bar_color)

    # Draw an horizontal line with the max reward value.
    ax.axhline(y=max_reward, color=max_reward_line_color, linewidth=2)

    # Set label for both axis.
    ax.set_ylabel("Ricompensa media", fontsize="x-large")
    ax.set_xlabel("Scenario di addestramento", fontsize="x-large")

    # In the Y axis show ticks with interval of 50 reward points.
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    # Set and show legend.
    ax.legend((ppo_bars, ppo_cc_bars), ('PPO', 'PPO-CC'))

    # Set X axis labels (one for each pair of PPO and PPO-CC experiments).
    labels = ["Reale",
              "Sintetico sinusoidale",
              "Sintetico gaussiano"]
    ax.set_xticks(x_ticks, labels=labels, fontsize='x-large', fontstretch='condensed')

    # Set background.
    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = plots_dir / f"eval_{env_type}_summary_reward.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


def _get_data_specific(exp):
    evals_dir = list(exp.glob("evaluation_*"))

    data = {"values": [], "max": [], "min": [], "types": []}

    reward = reward_min = reward_max = 0
    for eval_dir in evals_dir:
        # Read data from the given evaluation directory
        eval_type = dfaas_utils.json_to_dict(eval_dir / "env_config.json")
        eval_type = eval_type["input_requests_type"]

        eval = dfaas_utils.parse_result_file(eval_dir / "evaluation.json")
        eval = eval[0]["evaluation"]

        reward_min = eval["episode_reward_min"]
        reward_max = eval["episode_reward_max"]
        reward_avg = eval["episode_reward_mean"]

        # Because matplotlib wants the errors (min, max) as offsets, I need to
        # process the values.
        reward_avg_min = reward_avg - reward_min
        reward_avg_max = reward_max - reward_avg

        data["values"].append(reward_avg)
        data["max"].append(reward_avg_max)
        data["min"].append(reward_avg_min)
        data["types"].append(eval_type)

    return data


def make_specific(env_type, exps, out):
    plots_dir = out
    plots_dir.mkdir(exist_ok=True)

    train_type = dfaas_utils.json_to_dict(exps[0] / "env_config.json")
    train_type = train_type["input_requests_type"]
    train_type_2 = dfaas_utils.json_to_dict(exps[1] / "env_config.json")
    assert train_type == train_type_2["input_requests_type"]

    data_ppo = _get_data_specific(exps[0])
    data_ppo_cc = _get_data_specific(exps[1])
    env = dfaas_env.DFaaS()  # Dummy env.

    fig = plt.figure(figsize=(12, 7), dpi=600, layout="constrained")
    ax = fig.subplots()

    # Limits for the y axis, both for total and single step.
    # Note the reward for each step is in [0, 1].
    _, top = env.reward_range
    max_reward = top * env.max_steps * env.agents
    bottom = 160  # Heuristically selected.
    ax.set_ylim(bottom=160, top=max_reward+10)

    # Set bar width and colors.
    bar_width = 0.30
    ppo_bar_color = '#99ccff'
    ppo_cc_bar_color = '#ff9999'
    max_reward_line_color = "#7a49a5"
    capsize = 10  # Length of the error horizontal bar.

    x_ticks = np.arange(len(data_ppo["values"]))
    ppo_bars = ax.bar(x_ticks - bar_width/2, data_ppo["values"], bar_width,
                      yerr=[data_ppo["min"], data_ppo["max"]],
                      capsize=capsize, color=ppo_bar_color)
    ppo_cc_bars = ax.bar(x_ticks + bar_width/2, data_ppo_cc["values"], bar_width,
                         yerr=[data_ppo_cc["min"], data_ppo_cc["max"]],
                         capsize=capsize, color=ppo_cc_bar_color)

    # Draw an horizontal line with the max reward value.
    ax.axhline(y=max_reward, color=max_reward_line_color, linewidth=2)

    # Set label for both axis.
    ax.set_ylabel("Ricompensa media", fontsize="large")
    ax.set_xlabel("Scenario di valutazione", fontsize="large")

    # In the Y axis show ticks with interval of 50 reward points.
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    # Set and show legend.
    ax.legend((ppo_bars, ppo_cc_bars), ('PPO', 'PPO-CC'))

    # Set X axis labels (one for each pair of PPO and PPO-CC experiments).
    assert data_ppo["types"] == data_ppo_cc["types"]
    assert data_ppo["types"][0] == "synthetic-sinusoidal"
    assert data_ppo["types"][1] == "synthetic-normal"
    assert data_ppo["types"][2] == "real"
    labels = ["Sintetico sinusoidale",
              "Sintetico gaussiano",
              "Reale"]
    ax.set_xticks(x_ticks, labels=labels, fontsize='large', fontstretch='condensed')

    # Set background.
    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = plots_dir / f"eval_{env_type}_train_{train_type}_summary_reward.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")

if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    exps = {}
    exps["SYM"] = [Path(f"results/final/DFAAS-MA_SYM_500_real"),
                   Path(f"results/final/DFAAS-MA_SYM_500_synt_sin"),
                   Path(f"results/final/DFAAS-MA_SYM_500_synt_norm"),
                   Path(f"results/final/DFAAS-MA_SYM_500_cc_real"),
                   Path(f"results/final/DFAAS-MA_SYM_500_cc_synt_sin"),
                   Path(f"results/final/DFAAS-MA_SYM_500_cc_synt_norm")]
    exps["ASYM"] = [Path(f"results/final/DFAAS-MA_ASYM_500_real"),
                    Path(f"results/final/DFAAS-MA_ASYM_500_synt_sin"),
                    Path(f"results/final/DFAAS-MA_ASYM_500_synt_norm"),
                    Path(f"results/final/DFAAS-MA_ASYM_500_cc_real"),
                    Path(f"results/final/DFAAS-MA_ASYM_500_cc_synt_sin"),
                    Path(f"results/final/DFAAS-MA_ASYM_500_cc_synt_norm")]
    exps["BASE"] = [Path(f"results/final/DFAAS-MA_BASE_500_real"),
                    Path(f"results/final/DFAAS-MA_BASE_500_synt_sin"),
                    Path(f"results/final/DFAAS-MA_BASE_500_synt_norm"),
                    Path(f"results/final/DFAAS-MA_BASE_500_cc_real"),
                    Path(f"results/final/DFAAS-MA_BASE_500_cc_synt_sin"),
                    Path(f"results/final/DFAAS-MA_BASE_500_cc_synt_norm")]

    out = Path("results/final/plots")

    for env_type in ["BASE", "ASYM", "SYM"]:
        make(env_type, exps[env_type], out)

    base_norm = [Path(f"results/final/DFAAS-MA_BASE_500_synt_norm"),
                 Path(f"results/final/DFAAS-MA_BASE_500_cc_synt_norm")]
    make_specific("BASE", base_norm, out)
